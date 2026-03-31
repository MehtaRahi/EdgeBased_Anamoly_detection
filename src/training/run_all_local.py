import json
import numpy as np
import pandas as pd
from pathlib import Path

from src.models.autoencoder import improved_cnn_lstm
from src.data.data_loader import load_data

np.random.seed(42)
BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SEQ_LEN = 32
NUM_FEATURES = 38
EPOCHS = 10
BATCH_SIZE = 64

MACHINES = [
    "machine-1-1",
    "machine-1-2",
    "machine-1-3",
    "machine-2-1",
    "machine-3-6"
]


def train_one_machine(machine_id):
    print(f"\n[TRAINING] {machine_id}")

    X_train, X_test, y_test = load_data(machine_id)

    model = improved_cnn_lstm(seq_len=SEQ_LEN, num_features=NUM_FEATURES)

    noise_factor = 0.08
    X_train_noisy = X_train + noise_factor * np.random.normal(size=X_train.shape)
    X_train_noisy = np.clip(X_train_noisy, 0., 1.)

    model.fit(
        X_train_noisy, X_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=1
    )

    save_path = RESULTS_DIR / f"{machine_id}_model.h5"
    model.save(str(save_path).replace(".h5", ".keras"))

    print(f"[SAVED] Model saved → {save_path}")

    return model, X_train, X_test, y_test


def evaluate_one_machine(model, X_train, X_test, y_test, machine_id):

    from sklearn.metrics import precision_score, recall_score, f1_score

    train_preds = model.predict(X_train, verbose=0)

    train_error = (X_train - train_preds) ** 2
    feature_std = np.std(X_train, axis=0) + 1e-8
    train_error = train_error / feature_std
    train_mse = train_error.mean(axis=2).mean(axis=1)

    mean = train_mse.mean()
    std = train_mse.std() + 1e-8
    train_mse_norm = (train_mse - mean) / std

    test_preds = model.predict(X_test, verbose=0)

    test_error = (X_test - test_preds) ** 2
    test_error = test_error / feature_std
    test_mse = test_error.mean(axis=2).mean(axis=1)

    test_mse_norm = (test_mse - mean) / std

    percentiles = [90, 92, 94, 95, 96, 97, 98, 99]

    best_f1 = 0
    best_threshold = None

    for p in percentiles:
        threshold = np.percentile(train_mse_norm, p)
        y_pred = (test_mse_norm > threshold).astype(int)

        f1 = f1_score(y_test, y_pred, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    threshold = best_threshold if best_threshold is not None else np.percentile(train_mse_norm, 98)

    y_pred = (test_mse_norm > threshold).astype(int)

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


def main():
    summary = []
    thresholds = {}

    for machine_id in MACHINES:

        model, X_train, X_test, y_test = train_one_machine(machine_id)

        precision, recall, f1, threshold = evaluate_one_machine(
            model, X_train, X_test, y_test, machine_id
        )

        summary.append({
            "machine": machine_id,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "threshold": threshold
        })

        thresholds[machine_id] = threshold

    df = pd.DataFrame(summary)
    summary_path = RESULTS_DIR / "local_summary.csv"
    df.to_csv(summary_path, index=False)

    print(f"[SAVED] Local training summary → {summary_path}")

    json_path = RESULTS_DIR / "thresholds.json"
    with open(json_path, "w") as f:
        json.dump(thresholds, f, indent=4)

    print(f"[SAVED] Thresholds JSON → {json_path}")

    print("\n======= FINAL SUMMARY =======")
    print("Average Precision:", df["precision"].mean())
    print("Average Recall   :", df["recall"].mean())
    print("Average F1       :", df["f1_score"].mean())
    print("============================\n")


if __name__ == "__main__":
    main()