import json
import numpy as np
from pathlib import Path

from sklearn.metrics import precision_score, recall_score, f1_score

from src.models.autoencoder import autoencoder
from src.data.data_loader import load_data
from src.evaluation.threshold import compute_threshold


BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


MACHINES = [
    "machine-1-1",
    "machine-1-2",
    "machine-1-3",
    "machine-2-1",
    "machine-3-6",
]


for machine in MACHINES:
    print("\n[TRAINING LOCAL AE] =", machine)


    X_train, X_test, y_test = load_data(machine)


    model = autoencoder(seq_len=64, n_features=38)


    model.fit(
        X_train, X_train,
        epochs=5,
        batch_size=64,
        verbose=1,
    )


    model_path = RESULTS_DIR / f"{machine}_model.h5"
    model.save(model_path)
    print(f"[SAVED] Model → {model_path}")


    preds = model.predict(X_test, verbose=0)
    mse = ((X_test - preds) ** 2).mean(axis=2).mean(axis=1)


    threshold = compute_threshold(mse)
    y_pred = (mse > threshold).astype(int)


    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "threshold": float(threshold)
    }


    metrics_path = RESULTS_DIR / f"{machine}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)


    print("\n===== LOCAL AE EVALUATION =====")
    print(f"Machine   : {machine}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"Threshold : {threshold:.6f}")
    print("=================================\n")