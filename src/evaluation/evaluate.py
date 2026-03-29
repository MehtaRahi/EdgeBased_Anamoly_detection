import numpy as np
import tensorflow as tf
from pathlib import Path

from sklearn.metrics import precision_score, recall_score, f1_score

from src.data.data_loader import load_data
from src.evaluation.threshold import compute_threshold

# -----------------------------------------------------------
# PATH SETUP
# -----------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MODEL_PATH = RESULTS_DIR / "global_model.h5"

# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------
SEQ_LEN = 64
NUM_FEATURES = 38

MACHINES = [
    "machine-1-1",
    "machine-1-2",
    "machine-1-3",
    "machine-2-1",
    "machine-3-6"
]

# -----------------------------------------------------------
# LOAD GLOBAL MODEL
# -----------------------------------------------------------
def load_global_model():
    print(f"[INFO] Loading global FL model → {MODEL_PATH}")

    model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    model.compile(optimizer="adam", loss="mse")

    return model

# -----------------------------------------------------------
# EVALUATE GLOBAL MODEL
# -----------------------------------------------------------
def evaluate_global_model():
    model = load_global_model()

    all_mse = []
    all_labels = []

    print("\n[INFO] Running global evaluation on all machines...\n")

    for machine_id in MACHINES:
        print(f"[INFO] Evaluating on: {machine_id}")

        _, X_test, y_test = load_data(machine_id)

        preds = model.predict(X_test, verbose=0)

        mse = np.mean((X_test - preds) ** 2, axis=(1, 2))

        all_mse.extend(mse)
        all_labels.extend(y_test)

    all_mse = np.array(all_mse)
    all_labels = np.array(all_labels)

    # Threshold (IQR)
    threshold = compute_threshold(all_mse)
    print(f"\n[INFO] Computed anomaly threshold (IQR): {threshold:.6f}")

    y_pred = (all_mse > threshold).astype(int)

    # Metrics
    precision = precision_score(all_labels, y_pred, zero_division=0)
    recall = recall_score(all_labels, y_pred, zero_division=0)
    f1 = f1_score(all_labels, y_pred, zero_division=0)

    print("\n========== GLOBAL EVALUATION RESULTS ==========")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("==============================================\n")

    return precision, recall, f1

# -----------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------
if __name__ == "__main__":
    evaluate_global_model()