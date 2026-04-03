import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.models.autoencoder import improved_cnn_lstm_prob
from src.data.data_loader import load_data
from src.evaluation.evaluator import evaluate


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEQ_LEN = 64
NUM_FEATURES = 38

MACHINES = [
    "machine-1-1",
    "machine-1-2",
    "machine-1-3",
    "machine-2-1",
    "machine-3-6"
]


def main():

    print("[INFO] Starting evaluation...\n")

    summary = []

    for machine_id in MACHINES:
        print(f"Evaluating {machine_id}")

        # 🔥 reload model per machine (important)
        model = improved_cnn_lstm_prob(seq_len=SEQ_LEN, num_features=NUM_FEATURES)
        model.load_weights(PROJECT_ROOT / "models/global_model.keras")

        X_train, X_test, y_test = load_data(machine_id)

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

    df = pd.DataFrame(summary)

    print("\n=== FEDERATED MODEL SUMMARY ===")
    print("Avg Precision:", df["precision"].mean())
    print("Avg Recall   :", df["recall"].mean())
    print("Avg F1       :", df["f1_score"].mean())

    # Save results
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    dataset_name = "smd"
    df.to_csv(results_dir / f"{dataset_name}_federated_summary.csv", index=False)


    import json
    metrics = {
        "avg_precision": float(df["precision"].mean()),
        "avg_recall": float(df["recall"].mean()),
        "avg_f1": float(df["f1_score"].mean())
    }

    with open(results_dir / f"{dataset_name}_federated_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("\n[SAVED] Results + metrics")


if __name__ == "__main__":
    main()