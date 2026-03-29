import json
import numpy as np
from pathlib import Path

from sklearn.metrics import precision_score, recall_score, f1_score

from src.models.autoencoder import autoencoder
from src.data.data_loader import load_data
from src.evaluation.threshold import compute_threshold

# ---------------------------------
# PATH SETUP (MODERN + PORTABLE)
# ---------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------
# CONFIG
# ---------------------------------
MACHINES = [
    "machine-1-1",
    "machine-1-2",
    "machine-1-3",
    "machine-2-1",
    "machine-3-6",
]

# ---------------------------------
# TRAIN AND EVALUATE LOCAL AE
# ---------------------------------
for machine in MACHINES:
    print("\n[TRAINING LOCAL AE] =", machine)

    # Load data
    X_train, X_test, y_test = load_data(machine)

    # Build model
    model = autoencoder(seq_len=64, n_features=38)

    # Train
    model.fit(
        X_train, X_train,
        epochs=5,
        batch_size=64,
        verbose=1,
    )

    # Save model
    model_path = RESULTS_DIR / f"{machine}_model.h5"
    model.save(model_path)
    print(f"[SAVED] Model → {model_path}")

    # Predict
    preds = model.predict(X_test, verbose=0)
    mse = ((X_test - preds) ** 2).mean(axis=2).mean(axis=1)

    # Threshold
    threshold = compute_threshold(mse)
    y_pred = (mse > threshold).astype(int)

    # Metrics
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "threshold": float(threshold)
    }

    # Save metrics
    metrics_path = RESULTS_DIR / f"{machine}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # Print
    print("\n===== LOCAL AE EVALUATION =====")
    print(f"Machine   : {machine}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"Threshold : {threshold:.6f}")
    print("=================================\n")