import json
import numpy as np
import pandas as pd
from pathlib import Path

from src.models.autoencoder import improved_cnn_lstm
from src.data.data_loader import load_data
from src.evaluation.threshold import compute_threshold

# ---------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
SEQ_LEN = 64
NUM_FEATURES = 38
EPOCHS = 5
BATCH_SIZE = 64

MACHINES = [
    "machine-1-1",
    "machine-1-2",
    "machine-1-3",
    "machine-2-1",
    "machine-3-6"
]

# ---------------------------------------------------------
# TRAIN
# ---------------------------------------------------------
def train_one_machine(machine_id):
    print(f"\n[TRAINING] {machine_id}")

    X_train, X_test, y_test = load_data(machine_id)

    model = improved_cnn_lstm(seq_len=SEQ_LEN, num_features=NUM_FEATURES)

    model.fit(
        X_train, X_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    save_path = RESULTS_DIR / f"{machine_id}_model.h5"
    model.save(save_path)

    print(f"[SAVED] Model saved → {save_path}")

    return model, X_test, y_test


# ---------------------------------------------------------
# EVALUATE
# ---------------------------------------------------------
def evaluate_one_machine(model, X_test, y_test, machine_id):

    preds = model.predict(X_test, verbose=0)

    mse = np.mean((X_test - preds) ** 2, axis=(1, 2))

    threshold = compute_threshold(mse)

    y_pred = (mse > threshold).astype(int)

    from sklearn.metrics import precision_score, recall_score, f1_score

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n======== LOCAL EVALUATION ========")
    print(f"Machine   : {machine_id}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"Threshold : {threshold:.6f}")
    print("===================================\n")

    return precision, recall, f1, float(threshold)


# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------
def run_all():
    summary = []
    thresholds = {}

    for machine_id in MACHINES:

        model, X_test, y_test = train_one_machine(machine_id)

        precision, recall, f1, threshold = evaluate_one_machine(
            model, X_test, y_test, machine_id
        )

        summary.append({
            "machine": machine_id,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "threshold": threshold
        })

        thresholds[machine_id] = threshold

    # Save CSV
    df = pd.DataFrame(summary)
    summary_path = RESULTS_DIR / "local_summary.csv"
    df.to_csv(summary_path, index=False)

    print(f"[SAVED] Local training summary → {summary_path}")

    # Save JSON
    json_path = RESULTS_DIR / "thresholds.json"
    with open(json_path, "w") as f:
        json.dump(thresholds, f, indent=4)

    print(f"[SAVED] Thresholds JSON → {json_path}")

    print("\n=========== COMPLETE ===========")
    print("Local training + evaluation done for ALL machines!")
    print("================================\n")


if __name__ == "__main__":
    run_all()