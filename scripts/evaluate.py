import sys
from pathlib import Path

# Fix import path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import json

from src.models.autoencoder import improved_cnn_lstm
from src.data.data_loader import load_data
from src.data.skab_loader import load_skab
from src.evaluation.evaluator import evaluate


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET = "skab"   # "smd" or "skab"

SEQ_LEN = 50

MACHINES = [
    "machine-1-1",
    "machine-1-2",
    "machine-1-3",
    "machine-2-1",
    "machine-3-6"
]


# -----------------------
# SMD evaluation
# -----------------------
def run_smd():
    summary = []

    for machine_id in MACHINES:
        print(f"Evaluating {machine_id}")

        X_train, X_test, y_test = load_data(machine_id, dataset="smd")

        num_features = X_train.shape[2]

        model = improved_cnn_lstm(seq_len=SEQ_LEN, num_features=num_features)

        # 🔥 Fix model path
        model.load_weights(PROJECT_ROOT / "models/federated_smd_model.keras")

        precision, recall, f1, threshold = evaluate(
            model, X_train, X_test, y_test
        )

        print(f"{machine_id} → Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")
        print(f"Threshold: {threshold:.4f}\n")

        summary.append({
            "machine": machine_id,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "threshold": threshold
        })

    return pd.DataFrame(summary)


# -----------------------
# SKAB evaluation (FIXED)
# -----------------------
def run_skab():

    print("Evaluating SKAB dataset")

    # 🔥 Correct loader usage
    X_train, X_test, y_test = load_skab()

    num_features = X_train.shape[2]

    model = improved_cnn_lstm(seq_len=SEQ_LEN, num_features=num_features)

    # 🔥 Load correct federated model
    model.load_weights(PROJECT_ROOT / "models/federated_skab_model.keras")

    # 🔥 No manual split anymore
    precision, recall, f1, threshold = evaluate(
        model, X_train, X_test, y_test
    )

    print("Using: AE + RandomForest\n")
    print("Mode: Classifier-based detection\n")

    summary = [{
        "dataset": "skab",
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "mode": "classifier"
    }]

    return pd.DataFrame(summary)


# -----------------------
# MAIN
# -----------------------
def main():

    print(f"[INFO] Starting evaluation on {DATASET.upper()}...\n")

    if DATASET == "smd":
        df = run_smd()

    elif DATASET == "skab":
        df = run_skab()

    else:
        raise ValueError("Unsupported dataset")

    print("\n=== MODEL SUMMARY ===")
    print("Avg Precision:", df["precision"].mean())
    print("Avg Recall   :", df["recall"].mean())
    print("Avg F1       :", df["f1_score"].mean())

    # Save results
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    df.to_csv(results_dir / f"{DATASET}_summary.csv", index=False)

    metrics = {
        "avg_precision": float(df["precision"].mean()),
        "avg_recall": float(df["recall"].mean()),
        "avg_f1": float(df["f1_score"].mean())
    }

    with open(results_dir / f"{DATASET}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("\n[SAVED] Results + metrics")


if __name__ == "__main__":
    main()