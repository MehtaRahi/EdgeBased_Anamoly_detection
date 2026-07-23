import sys
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

# Fix import path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.autoencoder import improved_cnn_lstm
from src.data.skab_loader import load_skab
from src.evaluation.evaluator import evaluate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = "skab"
SEQ_LEN = 50

# Wrapper for TFLite inference that exposes predict and predict_latent to evaluator.py
class TFLiteWrapper:
    def __init__(self, tflite_model_bytes):
        self.interpreter = tf.lite.Interpreter(model_content=tflite_model_bytes)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def fit(self, X_train_noisy, X_train, **kwargs):
        # TFLite models are already quantized and cannot be retrained.
        # This is a no-op to satisfy the personalization step in evaluate().
        pass

    def predict(self, X, verbose=0):
        preds = []
        for i in range(len(X)):
            inp = X[i:i+1].astype(np.float32)
            self.interpreter.set_tensor(self.input_details[0]['index'], inp)
            self.interpreter.invoke()
            
            out0 = self.interpreter.get_tensor(self.output_details[0]['index'])
            out1 = self.interpreter.get_tensor(self.output_details[1]['index'])
            
            # Reconstruction output is 3D: (1, seq_len, features)
            if len(out0.shape) == 3:
                preds.append(out0[0])
            else:
                preds.append(out1[0])
        return np.array(preds)

    def predict_latent(self, X):
        latents = []
        for i in range(len(X)):
            inp = X[i:i+1].astype(np.float32)
            self.interpreter.set_tensor(self.input_details[0]['index'], inp)
            self.interpreter.invoke()
            
            out0 = self.interpreter.get_tensor(self.output_details[0]['index'])
            out1 = self.interpreter.get_tensor(self.output_details[1]['index'])
            
            # Latent output is 2D: (1, latent_dim)
            if len(out0.shape) == 2:
                latents.append(out0[0])
            else:
                latents.append(out1[0])
        return np.array(latents)

def benchmark_tflite_latency(model_bytes, num_runs=50):
    interpreter = tf.lite.Interpreter(model_content=model_bytes)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    
    # Random sample matching model input shape
    sample = np.random.rand(*input_details[0]['shape']).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        interpreter.set_tensor(input_details[0]['index'], sample)
        interpreter.invoke()
        
    # Measure
    latencies = []
    for _ in range(num_runs):
        start = time.time()
        interpreter.set_tensor(input_details[0]['index'], sample)
        interpreter.invoke()
        _ = interpreter.get_output_details() # touch output
        end = time.time()
        latencies.append((end - start) * 1000) # ms
        
    return np.mean(latencies)

def main():
    print("--- Running Edge Quantization & Profiling ---")

    # Load Data
    X_train, X_test, y_test = load_skab()
    num_features = X_train.shape[2]

    # Load original keras model
    keras_model_path = PROJECT_ROOT / "models/federated_skab_model.keras"
    if not keras_model_path.exists():
        print(f"[ERROR] Keras model weights not found at {keras_model_path}. Train model first.")
        return
        
    keras_model = improved_cnn_lstm(seq_len=SEQ_LEN, num_features=num_features)
    keras_model.load_weights(keras_model_path)
    
    # Create multi-output Keras model (reconstruction & latent space output)
    inputs = keras_model.input
    reconstruction = keras_model.output
    latent = keras_model.get_layer("latent").output
    multi_output_model = tf.keras.Model(inputs=inputs, outputs=[reconstruction, latent])
    
    # Prepare calibrations representative generator for INT8 Full Integer Quantization
    def representative_data_gen():
        for i in range(100):
            yield [X_train[i:i+1].astype(np.float32)]

    models_dir = PROJECT_ROOT / "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # ------------------ 1. CONVERT TFLITE ------------------
    conversions = {}
    
    # FP32
    print("Converting: FP32 Standard TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(multi_output_model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_fp32 = converter.convert()
    conversions["FP32"] = tflite_fp32
    
    # FP16
    print("Converting: FP16 Quantized TFLite...")
    converter_fp16 = tf.lite.TFLiteConverter.from_keras_model(multi_output_model)
    converter_fp16.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_fp16.target_spec.supported_types = [tf.float16]
    converter_fp16.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_fp16 = converter_fp16.convert()
    conversions["FP16"] = tflite_fp16
    
    # INT8 Dynamic Range
    print("Converting: INT8 Dynamic Range Quantized TFLite...")
    converter_dyn = tf.lite.TFLiteConverter.from_keras_model(multi_output_model)
    converter_dyn.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_dyn.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_dyn = converter_dyn.convert()
    conversions["INT8_Dynamic"] = tflite_dyn
    
    # INT8 Full Integer
    print("Converting: INT8 Full Integer Quantized TFLite...")
    converter_int8 = tf.lite.TFLiteConverter.from_keras_model(multi_output_model)
    converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_int8.representative_dataset = representative_data_gen
    # fallback to float ops for unsupported layers (e.g. LSTM unrolled ops compatibility)
    converter_int8.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS
    ]
    tflite_int8 = converter_int8.convert()
    conversions["INT8_Full"] = tflite_int8

    # Save files
    for name, data in conversions.items():
        path = models_dir / f"skab_{name.lower()}.tflite"
        with open(path, "wb") as f:
            f.write(data)
        print(f"[OK] Saved model to {path}")

    # ------------------ 2. PROFILE SIZE, LATENCY, AND ACCURACY ------------------
    results = []
    
    # First profile Keras Baseline (as reference)
    print("\nProfiling Keras Baseline...")
    keras_size_kb = os.path.getsize(keras_model_path) / 1024.0
    
    # Latency profile for Keras
    keras_latencies = []
    sample_keras = np.random.rand(1, SEQ_LEN, num_features).astype(np.float32)
    for _ in range(10):
        keras_model.predict(sample_keras, verbose=0)
    for _ in range(50):
        start = time.time()
        keras_model.predict(sample_keras, verbose=0)
        end = time.time()
        keras_latencies.append((end - start) * 1000)
    keras_latency = np.mean(keras_latencies)
    
    # F1-score evaluation
    p_k, r_k, f1_k, _ = evaluate(keras_model, X_train, X_test, y_test)
    results.append({
        "Model Type": "Keras (Reference)",
        "Size (KB)": keras_size_kb,
        "Avg Latency (ms)": keras_latency,
        "Precision": p_k,
        "Recall": r_k,
        "F1 Score": f1_k
    })
    print(f"Keras -> Size: {keras_size_kb:.1f} KB | Latency: {keras_latency:.2f} ms | F1: {f1_k:.4f}")

    for name, data in conversions.items():
        print(f"\nProfiling {name} TFLite...")
        # Get Size
        path = models_dir / f"skab_{name.lower()}.tflite"
        size_kb = os.path.getsize(path) / 1024.0
        
        # Get Latency
        latency = benchmark_tflite_latency(data)
        
        # Get Accuracy
        wrapper = TFLiteWrapper(data)
        p, r, f1, _ = evaluate(wrapper, X_train, X_test, y_test)
        
        results.append({
            "Model Type": f"TFLite ({name})",
            "Size (KB)": size_kb,
            "Avg Latency (ms)": latency,
            "Precision": p,
            "Recall": r,
            "F1 Score": f1
        })
        print(f"{name} -> Size: {size_kb:.1f} KB | Latency: {latency:.2f} ms | F1: {f1:.4f}")

    # Save metrics summary
    df = pd.DataFrame(results)
    df.to_csv(PROJECT_ROOT / f"results/{DATASET}_quantization_profile.csv", index=False)
    print(f"\n[OK] Profiling Complete! Saved to results/{DATASET}_quantization_profile.csv")

if __name__ == "__main__":
    main()
