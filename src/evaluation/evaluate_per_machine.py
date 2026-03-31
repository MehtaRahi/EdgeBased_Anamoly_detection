import json
import numpy as np
import tensorflow as tf
from pathlib import Path

from sklearn.metrics import precision_score, recall_score, f1_score

from src.data.data_loader import load_data
from src.evaluation.threshold import compute_threshold


BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MODEL_PATH = RESULTS_DIR / "global_model.h5"


MACHINES = [
    "machine-1-1",
    "machine-1-2",
    "machine-1-3",
    "machine-2-1",
    "machine-3-6",
]


print("[INFO] Loading Global Federated Model...")
model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)

federated_results = {}

print("\n[INFO] Running per-machine federated evaluation...\n")


for machine in MACHINES:
    print(f"[INFO] Evaluating on: {machine}")

    _, X_test, y_test = load_data(machine)

    preds = model.predict(X_test, verbose=0)
    mse = ((X_test - preds) ** 2).mean(axis=2).mean(axis=1)

    threshold = compute_threshold(mse)

    y_pred = (mse > threshold).astype(int)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    federated_results[machine] = {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "threshold": float(threshold),
    }


print("\n========== FEDERATED RESULTS (PER MACHINE) ==========")
for m, stats in federated_results.items():
    print(f"\nMachine: {m}")
    print(f"Precision: {stats['precision']:.4f}")
    print(f"Recall:    {stats['recall']:.4f}")
    print(f"F1-Score:  {stats['f1_score']:.4f}")
    print(f"Threshold: {stats['threshold']:.6f}")
print("=====================================================\n")


save_path = RESULTS_DIR / "federated_scores.json"

with open(save_path, "w") as f:
    json.dump(federated_results, f, indent=4)

print(f"[SAVED] Federated per-machine metrics saved → {save_path}")