"""
Contamination Grid Search
=========================
Loads the saved federated SKAB model directly (no architecture re-creation needed)
and sweeps Isolation Forest contamination values to find the optimal F1 score.

No retraining required — pure evaluation sweep.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import tensorflow as tf
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve
from scipy.signal import medfilt

from src.data.skab_loader import load_skab
from src.evaluation.evaluator import point_adjust

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ── Grid values to test ────────────────────────────────────────────────────────
CONTAMINATION_VALUES = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
CLASSIFIERS = ["isolation_forest", "ocsvm"]

# OCSVM nu mirrors contamination semantics
NU_VALUES = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


def extract_features(model, X_train, X_test):
    """Extract 4-feature hybrid representation from autoencoder."""
    train_pred = model.predict(X_train, verbose=0)
    test_pred  = model.predict(X_test,  verbose=0)

    # Feature 0: MSE
    train_mse = np.mean((X_train - train_pred) ** 2, axis=(1, 2))
    test_mse  = np.mean((X_test  - test_pred)  ** 2, axis=(1, 2))

    # Feature 1: Temporal diff of reconstruction error
    train_err  = np.mean((X_train - train_pred) ** 2, axis=2)
    test_err   = np.mean((X_test  - test_pred)  ** 2, axis=2)
    train_diff = np.mean(np.abs(np.diff(train_err, axis=1)), axis=1)
    test_diff  = np.mean(np.abs(np.diff(test_err,  axis=1)), axis=1)

    # Feature 2: Latent space deviation
    latent_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer("latent").output
    )
    train_latent = latent_model.predict(X_train, verbose=0)
    test_latent  = latent_model.predict(X_test,  verbose=0)
    center = np.mean(train_latent, axis=0)
    train_latent_score = np.linalg.norm(train_latent - center, axis=1)
    test_latent_score  = np.linalg.norm(test_latent  - center, axis=1)

    # Feature 3: Variance of reconstruction error
    train_var = np.var((X_train - train_pred), axis=(1, 2))
    test_var  = np.var((X_test  - test_pred),  axis=(1, 2))

    X_train_feat = np.stack([train_mse, train_diff, train_latent_score, train_var], axis=1)
    X_test_feat  = np.stack([test_mse,  test_diff,  test_latent_score,  test_var],  axis=1)

    scaler = StandardScaler()
    X_train_feat = scaler.fit_transform(X_train_feat)
    X_test_feat  = scaler.transform(X_test_feat)

    return X_train_feat, X_test_feat


def score_predictions(anomaly_score, y_test):
    """Threshold sweep + point-adjust + return P/R/F1."""
    prec_curve, rec_curve, thresholds = precision_recall_curve(y_test, anomaly_score)
    f1_curve = 2 * (prec_curve * rec_curve) / (prec_curve + rec_curve + 1e-8)

    best_idx = np.argmax(f1_curve)
    best_thresh = thresholds[max(best_idx - 1, 0)]

    y_pred = (anomaly_score >= best_thresh).astype(int)
    y_pred = medfilt(y_pred, kernel_size=5)
    y_pred = point_adjust(y_test, y_pred)

    p = precision_score(y_test, y_pred, zero_division=0)
    r = recall_score(y_test,    y_pred, zero_division=0)
    f = f1_score(y_test,        y_pred, zero_division=0)
    return p, r, f, best_thresh, int(y_pred.sum())


def main():
    print("=" * 60)
    print("  Contamination Grid Search - SKAB Federated Model")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────
    print("\n[1/3] Loading SKAB data...")
    X_train, X_test, y_test = load_skab()
    print(f"      Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"      Anomaly ratio in test: {y_test.mean():.4f}  ({int(y_test.sum())} / {len(y_test)})")

    # ── Load model ─────────────────────────────────────────────────
    model_path = PROJECT_ROOT / "models" / "federated_skab_model.keras"
    print(f"\n[2/3] Loading model from {model_path.name}...")
    model = tf.keras.models.load_model(str(model_path))
    print(f"      Architecture: {model.name}")

    # ── Extract features once (expensive) ─────────────────────────
    print("\n[3/3] Extracting hybrid features (MSE + Diff + Latent + Var)...")
    X_train_feat, X_test_feat = extract_features(model, X_train, X_test)
    print(f"      Feature matrix: train={X_train_feat.shape} | test={X_test_feat.shape}")

    results = []

    # ── Isolation Forest sweep ────────────────────────────────────
    print("\n" + "-" * 60)
    print("  Isolation Forest - Contamination Sweep")
    print("-" * 60)
    print(f"  {'Contamination':>14} | {'Precision':>9} | {'Recall':>6} | {'F1':>6} | {'Predicted':>9}")
    print("  " + "-" * 56)

    best_if_f1 = 0
    for c in CONTAMINATION_VALUES:
        clf = IsolationForest(n_estimators=200, contamination=c, random_state=42, n_jobs=-1)
        clf.fit(X_train_feat)
        score = -clf.decision_function(X_test_feat)

        p, r, f, thresh, n_pred = score_predictions(score, y_test)
        marker = "  << BEST" if f > best_if_f1 else ""
        best_if_f1 = max(best_if_f1, f)
        print(f"  {c:>14.2f} | {p:>9.4f} | {r:>6.4f} | {f:>6.4f} | {n_pred:>9}{marker}")
        results.append({"Classifier": "IsolationForest", "Param": c, "Precision": p, "Recall": r, "F1": f, "Predicted": n_pred})

    # ── One-Class SVM sweep ───────────────────────────────────────
    print("\n" + "-" * 60)
    print("  One-Class SVM (RBF) - Nu Sweep")
    print("-" * 60)
    print(f"  {'Nu':>14} | {'Precision':>9} | {'Recall':>6} | {'F1':>6} | {'Predicted':>9}")
    print("  " + "-" * 56)

    best_svm_f1 = 0
    for nu in NU_VALUES:
        clf = OneClassSVM(nu=nu, kernel="rbf", gamma="scale")
        clf.fit(X_train_feat)
        score = -clf.decision_function(X_test_feat)

        p, r, f, thresh, n_pred = score_predictions(score, y_test)
        marker = "  << BEST" if f > best_svm_f1 else ""
        best_svm_f1 = max(best_svm_f1, f)
        print(f"  {nu:>14.2f} | {p:>9.4f} | {r:>6.4f} | {f:>6.4f} | {n_pred:>9}{marker}")
        results.append({"Classifier": "OneClassSVM", "Param": nu, "Precision": p, "Recall": r, "F1": f, "Predicted": n_pred})

    # ── Save results ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    results_df = pd.DataFrame(results)
    out_path = PROJECT_ROOT / "results" / "contamination_grid_search.csv"
    out_path.parent.mkdir(exist_ok=True)
    results_df.to_csv(out_path, index=False)
    print(f"[SAVED] Grid search results -> {out_path}")

    best_row = results_df.loc[results_df["F1"].idxmax()]
    print(f"\n{'='*60}")
    print(f"  BEST OVERALL: {best_row['Classifier']} | param={best_row['Param']:.2f}")
    print(f"  Precision={best_row['Precision']:.4f} | Recall={best_row['Recall']:.4f} | F1={best_row['F1']:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
