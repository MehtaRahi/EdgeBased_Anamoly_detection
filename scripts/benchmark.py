import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score
from scipy.signal import medfilt
import tensorflow as tf
import json

# Fix import path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.autoencoder import improved_cnn_lstm
from src.data.data_loader import load_data
from src.data.skab_loader import load_skab
from src.evaluation.evaluator import evaluate

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Choose datasets to run baselines for
DATASETS = ["skab", "nab"]  # Can add "smd" too
SEQ_LEN = 50

MACHINES = [
    "machine-1-1",
    "machine-1-2",
    "machine-1-3",
    "machine-2-1",
    "machine-3-6"
]

def optimize_threshold_f1(y_true, anomaly_scores):
    precision, recall, thresholds = precision_recall_curve(y_true, anomaly_scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[max(best_idx - 1, 0)]
    y_pred = (anomaly_scores >= best_threshold).astype(int)
    y_pred = medfilt(y_pred, kernel_size=5)
    
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return p, r, f1, best_threshold

def run_baselines(dataset_name):
    print(f"\n--- Running Baselines for {dataset_name.upper()} ---")
    results = []

    # Load Data
    print(f"Loading {dataset_name.upper()} data...")
    if dataset_name == "skab":
        X_train, X_test, y_test = load_skab()
    elif dataset_name == "smd":
        X_train, X_test, y_test = load_data("machine-1-1", dataset="smd")
    elif dataset_name == "nab":
        from src.data.nab_loader import load_nab
        X_train, X_test, y_test = load_nab()
    else:
        raise ValueError("Unknown dataset")
        
    num_features = X_train.shape[2]

    # Load Federated Model
    fed_model_path = PROJECT_ROOT / f"models/federated_{dataset_name}_model.keras"
    if not fed_model_path.exists():
        print(f"[WARN] Federated model weights not found at {fed_model_path}. Please run training first.")
        return

    fed_model = improved_cnn_lstm(seq_len=SEQ_LEN, num_features=num_features)
    try:
        fed_model.load_weights(fed_model_path)
    except ValueError as e:
        print(f"[WARN] Failed to load weights for {dataset_name} due to architecture mismatch. {e}")
        return

    # 1. Baseline: Pure Reconstruction MSE Thresholding
    print("[1/5] Baseline: Pure Reconstruction MSE Thresholding...")
    test_pred = fed_model.predict(X_test, verbose=0)
    test_mse = np.mean((X_test - test_pred) ** 2, axis=(1, 2))
    p_mse, r_mse, f1_mse, _ = optimize_threshold_f1(y_test, test_mse)
    print(f"      F1: {f1_mse:.4f} | Precision: {p_mse:.4f} | Recall: {r_mse:.4f}")
    results.append({"Method": "Reconstruction MSE Only", "Precision": p_mse, "Recall": r_mse, "F1": f1_mse})

    # 2. Baseline: OneClassSVM on Raw features (Flattened sequences)
    print("[2/5] Baseline: OneClassSVM on Raw Sensor Data (No Deep Learning)...")
    X_train_raw = X_train.reshape(X_train.shape[0], -1)
    X_test_raw = X_test.reshape(X_test.shape[0], -1)
    clf_raw = OneClassSVM(nu=0.05, kernel="rbf", gamma="scale")
    clf_raw.fit(X_train_raw)
    scores_raw = -clf_raw.decision_function(X_test_raw)
    p_raw, r_raw, f1_raw, _ = optimize_threshold_f1(y_test, scores_raw)
    print(f"      F1: {f1_raw:.4f} | Precision: {p_raw:.4f} | Recall: {r_raw:.4f}")
    results.append({"Method": "Raw One-Class SVM", "Precision": p_raw, "Recall": r_raw, "F1": f1_raw})

    # 3. Ablation Study: Hybrid + Classifier with feature combinations
    print("[3/5] Ablation: Hybrid + Isolation Forest (MSE Only)")
    p_a1, r_a1, f1_a1, _ = evaluate(fed_model, X_train, X_test, y_test, feature_indices=[0], classifier="isolation_forest")
    print(f"      F1: {f1_a1:.4f} | Precision: {p_a1:.4f} | Recall: {r_a1:.4f}")
    results.append({"Method": "Ablation (MSE only)", "Precision": p_a1, "Recall": r_a1, "F1": f1_a1})

    print("[4/5] Ablation: Hybrid + Isolation Forest (MSE + Temporal Diff)")
    p_a2, r_a2, f1_a2, _ = evaluate(fed_model, X_train, X_test, y_test, feature_indices=[0, 1], classifier="isolation_forest")
    print(f"      F1: {f1_a2:.4f} | Precision: {p_a2:.4f} | Recall: {r_a2:.4f}")
    results.append({"Method": "Ablation (MSE + Diff)", "Precision": p_a2, "Recall": r_a2, "F1": f1_a2})

    # 4. Our Full Model: Federated CNN-LSTM + Classifier (4 features)
    print("[5/5] Full Hybrid Model (MSE + Diff + Latent + Var)")
    p_if, r_if, f1_if, _ = evaluate(fed_model, X_train, X_test, y_test, classifier="isolation_forest")
    results.append({"Method": "Full Proposed Hybrid Model (Isolation Forest)", "Precision": p_if, "Recall": r_if, "F1": f1_if})
    
    p_ocsvm, r_ocsvm, f1_ocsvm, _ = evaluate(fed_model, X_train, X_test, y_test, classifier="ocsvm")
    results.append({"Method": "Full Proposed Hybrid Model (OCSVM)", "Precision": p_ocsvm, "Recall": r_ocsvm, "F1": f1_ocsvm})

    # Save to CSV
    os.makedirs(str(PROJECT_ROOT / "results"), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(PROJECT_ROOT / f"results/{dataset_name}_baselines_summary.csv", index=False)
    print(f"\n[OK] Saved results to results/{dataset_name}_baselines_summary.csv")

def main():
    for dataset in DATASETS:
        run_baselines(dataset)

if __name__ == "__main__":
    main()
