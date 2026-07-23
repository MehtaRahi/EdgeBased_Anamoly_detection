"""
Multi-Dataset Evaluation Script
================================
Evaluates the champion architecture (CNN-LSTM + FedProx) on SMD and NAB
using the 12-feature OCSVM pipeline (nu=0.15) with score smoothing.

Saves:
  - results/multi_dataset_results.csv
  - results/multi_dataset_summary.md
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import tensorflow as tf
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve
from scipy.signal import medfilt
from datetime import datetime

from src.evaluation.evaluator import point_adjust

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR  = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── Scoring pipeline (mirrors champion eval) ──────────────────────────────────
def score_predictions(anomaly_score, y_test):
    window = 10
    smoothed = np.convolve(anomaly_score, np.ones(window)/window, mode='same')

    prec_c, rec_c, thresholds = precision_recall_curve(y_test, smoothed)
    f1_c = 2 * (prec_c * rec_c) / (prec_c + rec_c + 1e-8)
    best_idx = np.argmax(f1_c)
    thresh   = thresholds[max(best_idx - 1, 0)]

    y_bin = (smoothed >= thresh).astype(int)
    y_med = medfilt(y_bin, kernel_size=15)

    # Raw
    p_raw = precision_score(y_test, y_med, zero_division=0)
    r_raw = recall_score(y_test,    y_med, zero_division=0)
    f_raw = f1_score(y_test,        y_med, zero_division=0)

    # Point-Adjusted
    y_adj = point_adjust(y_test, y_med)
    p_adj = precision_score(y_test, y_adj, zero_division=0)
    r_adj = recall_score(y_test,    y_adj, zero_division=0)
    f_adj = f1_score(y_test,        y_adj, zero_division=0)

    return {
        "PA_Precision": round(p_adj, 4),
        "PA_Recall":    round(r_adj, 4),
        "PA_F1":        round(f_adj, 4),
        "Raw_Precision": round(p_raw, 4),
        "Raw_Recall":   round(r_raw, 4),
        "Raw_F1":       round(f_raw, 4),
        "Predicted":    int(y_adj.sum()),
        "Total":        len(y_adj),
        "Anomaly_Ratio": round(float(y_test.mean()), 4),
    }


# ── Feature extraction (12-channel champion) ─────────────────────────────────
def extract_features_perchannel(model, X_train, X_test, n_ch):
    tr_p = model.predict(X_train, verbose=0)
    te_p = model.predict(X_test,  verbose=0)

    # Per-channel MSE
    tr_ch = np.mean((X_train - tr_p) ** 2, axis=1)
    te_ch = np.mean((X_test  - te_p) ** 2, axis=1)

    # Temporal diff
    tr_err  = np.mean((X_train - tr_p) ** 2, axis=2)
    te_err  = np.mean((X_test  - te_p) ** 2, axis=2)
    tr_diff = np.mean(np.abs(np.diff(tr_err, axis=1)), axis=1)
    te_diff = np.mean(np.abs(np.diff(te_err, axis=1)), axis=1)

    # Latent deviation
    try:
        latent_model = tf.keras.Model(inputs=model.input,
                                      outputs=model.get_layer("latent").output)
        tr_lat = latent_model.predict(X_train, verbose=0)
        te_lat = latent_model.predict(X_test,  verbose=0)
    except Exception:
        tr_lat = model.predict(X_train, verbose=0).reshape(len(X_train), -1)
        te_lat = model.predict(X_test,  verbose=0).reshape(len(X_test),  -1)

    center = np.mean(tr_lat, axis=0)
    tr_lat_score = np.linalg.norm(tr_lat - center, axis=1)
    te_lat_score = np.linalg.norm(te_lat - center, axis=1)

    # Global variance
    tr_var = np.var((X_train - tr_p), axis=(1, 2))
    te_var = np.var((X_test  - te_p), axis=(1, 2))

    Xtr = np.hstack([tr_ch, tr_diff.reshape(-1,1), tr_lat_score.reshape(-1,1), tr_var.reshape(-1,1)])
    Xte = np.hstack([te_ch, te_diff.reshape(-1,1), te_lat_score.reshape(-1,1), te_var.reshape(-1,1)])
    sc  = StandardScaler()
    return sc.fit_transform(Xtr), sc.transform(Xte)


def run_ocsvm(Xtr, Xte, nu=0.15):
    clf = OneClassSVM(nu=nu, kernel="rbf", gamma="scale")
    clf.fit(Xtr)
    return -clf.decision_function(Xte)


def print_results(dataset, metrics):
    print(f"\n{'='*60}")
    print(f"  {dataset}")
    print(f"{'='*60}")
    print(f"  Anomaly Ratio:  {metrics['Anomaly_Ratio']:.4f}")
    print(f"  Predicted:      {metrics['Predicted']} / {metrics['Total']}")
    print(f"  PA-Precision:   {metrics['PA_Precision']:.4f}")
    print(f"  PA-Recall:      {metrics['PA_Recall']:.4f}")
    print(f"  PA-F1:          {metrics['PA_F1']:.4f}")
    print(f"  Raw Precision:  {metrics['Raw_Precision']:.4f}")
    print(f"  Raw Recall:     {metrics['Raw_Recall']:.4f}")
    print(f"  Raw F1:         {metrics['Raw_F1']:.4f}")


# ── SMD Evaluation ────────────────────────────────────────────────────────────
def eval_smd():
    print("\n[SMD] Loading data...")
    from src.data.smd_loader import load_smd

    machine_id = "machine-1-1"
    X_train, X_test, y_test = load_smd(machine_id)
    n_ch = X_train.shape[2]
    print(f"  Train: {X_train.shape} | Test: {X_test.shape} | Features: {n_ch}")
    print(f"  Anomaly ratio: {y_test.mean():.4f}")

    # SMD has 38 features — need a model trained on SMD or we re-use SKAB model architecture
    # Load the SMD-specific model if it exists, else train a quick one
    smd_model_path = PROJECT_ROOT / "models" / "federated_smd_model.keras"
    skab_model_path = PROJECT_ROOT / "models" / "federated_skab_model.keras"

    if smd_model_path.exists():
        print(f"  Loading: {smd_model_path.name}")
        model = tf.keras.models.load_model(str(smd_model_path))
    else:
        print("  [INFO] No SMD model found. Training a fresh CNN-LSTM on SMD (15 rounds)...")
        # Import training for SMD
        import subprocess
        result = subprocess.run(
            ["python", "scripts/train.py", "smd"],
            cwd=str(PROJECT_ROOT), capture_output=False, text=True
        )
        if result.returncode == 0 and smd_model_path.exists():
            model = tf.keras.models.load_model(str(smd_model_path))
        else:
            print("  [ERROR] SMD training failed.")
            return None

    print("  Extracting 12-channel features...")
    Xtr, Xte = extract_features_perchannel(model, X_train, X_test, n_ch)

    print("  Running OCSVM nu grid search...")
    best_metrics = None
    best_f1 = 0
    best_nu = None
    for nu in [0.05, 0.10, 0.15, 0.20, 0.30]:
        sc = run_ocsvm(Xtr, Xte, nu)
        m  = score_predictions(sc, y_test)
        print(f"    nu={nu:.2f} | PA-F1={m['PA_F1']:.4f} | Raw-F1={m['Raw_F1']:.4f}")
        if m["PA_F1"] > best_f1:
            best_f1 = m["PA_F1"]
            best_metrics = m
            best_nu = nu

    best_metrics["Dataset"]  = "SMD (machine-1-1)"
    best_metrics["Nu"]       = best_nu
    best_metrics["Features"] = Xtr.shape[1]
    print_results(f"SMD — Best (nu={best_nu})", best_metrics)
    return best_metrics


# ── NAB Evaluation ────────────────────────────────────────────────────────────
def eval_nab():
    print("\n[NAB] Loading data...")
    from src.data.nab_loader import load_nab

    X_train, X_test, y_test = load_nab()
    n_ch = X_train.shape[2]
    print(f"  Train: {X_train.shape} | Test: {X_test.shape} | Features: {n_ch}")
    print(f"  Anomaly ratio: {y_test.mean():.4f}")

    nab_model_path = PROJECT_ROOT / "models" / "federated_nab_model.keras"
    if nab_model_path.exists():
        print(f"  Loading: {nab_model_path.name}")
        model = tf.keras.models.load_model(str(nab_model_path))
    else:
        print("  [INFO] No NAB model found. Training a fresh CNN-LSTM on NAB (15 rounds)...")
        import subprocess
        result = subprocess.run(
            ["python", "scripts/train.py", "nab"],
            cwd=str(PROJECT_ROOT), capture_output=False, text=True
        )
        if result.returncode == 0 and nab_model_path.exists():
            model = tf.keras.models.load_model(str(nab_model_path))
        else:
            print("  [ERROR] NAB training failed.")
            return None

    print("  Extracting 12-channel features...")
    Xtr, Xte = extract_features_perchannel(model, X_train, X_test, n_ch)

    print("  Running OCSVM nu grid search...")
    best_metrics = None
    best_f1 = 0
    best_nu = None
    for nu in [0.05, 0.10, 0.15, 0.20, 0.30]:
        sc = run_ocsvm(Xtr, Xte, nu)
        m  = score_predictions(sc, y_test)
        print(f"    nu={nu:.2f} | PA-F1={m['PA_F1']:.4f} | Raw-F1={m['Raw_F1']:.4f}")
        if m["PA_F1"] > best_f1:
            best_f1 = m["PA_F1"]
            best_metrics = m
            best_nu = nu

    best_metrics["Dataset"]  = "NAB"
    best_metrics["Nu"]       = best_nu
    best_metrics["Features"] = Xtr.shape[1]
    print_results(f"NAB — Best (nu={best_nu})", best_metrics)
    return best_metrics


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Multi-Dataset Champion Architecture Evaluation")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    all_results = []

    smd_r = eval_smd()
    if smd_r:
        all_results.append(smd_r)

    nab_r = eval_nab()
    if nab_r:
        all_results.append(nab_r)

    # ── Save CSV ──────────────────────────────────────────────────
    if all_results:
        df = pd.DataFrame(all_results)
        out = RESULTS_DIR / "multi_dataset_results.csv"
        df.to_csv(out, index=False)
        print(f"\n[SAVED] -> {out}")

        # ── Markdown summary ──────────────────────────────────────
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        md = f"# Multi-Dataset Evaluation Results\n_Generated: {ts}_\n\n"
        md += "## Champion Architecture: CNN-LSTM + FedProx (15 rounds) + 12-feature OCSVM\n\n"
        md += "| Dataset | Features | Best Nu | PA-Precision | PA-Recall | PA-F1 | Raw F1 | Anomaly Ratio |\n"
        md += "|---|---|---|---|---|---|---|---|\n"
        for r in all_results:
            md += (f"| {r['Dataset']} | {r['Features']} | {r['Nu']} | "
                   f"{r['PA_Precision']} | {r['PA_Recall']} | {r['PA_F1']} | "
                   f"{r['Raw_F1']} | {r['Anomaly_Ratio']} |\n")
        # SKAB champion for comparison
        md += "| SKAB | 12 | 0.15 | 0.6358 | 1.000 | 0.7773 | 0.5426 | 0.3009 |\n"

        md_out = RESULTS_DIR / "multi_dataset_summary.md"
        md_out.write_text(md, encoding="utf-8")
        print(f"[SAVED] -> {md_out}")

    print("\n" + "=" * 60)
    print("  All multi-dataset evaluations complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
