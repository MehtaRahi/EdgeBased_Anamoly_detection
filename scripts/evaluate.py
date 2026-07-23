"""
Post-Training Experiment Runner
================================
Runs after federated training completes. Chains:
  Exp A: OCSVM nu grid search (eval only)
  Exp B: Per-channel feature engineering (eval only)
  Exp C: ECOD classifier from pyod (eval only)

Saves all results to results/experiment_results.csv
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
from datetime import datetime

from src.data.data_loader import load_data
from src.data.skab_loader import load_skab
from src.data.nab_loader import load_nab
from src.data.smap_loader import load_smap
from src.evaluation.evaluator import point_adjust

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR  = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DATASET = sys.argv[1] if len(sys.argv) > 1 else "skab"


# ─────────────────────────────────────────────────────────────────────────────
def score(anomaly_score, y_test):
    # 1. Score Smoothing (Moving average over 10 timestamps)
    window = 10
    smoothed_score = np.convolve(anomaly_score, np.ones(window)/window, mode='same')
    
    prec_c, rec_c, thresholds = precision_recall_curve(y_test, smoothed_score)
    f1_c = 2 * (prec_c * rec_c) / (prec_c + rec_c + 1e-8)
    best_idx   = np.argmax(f1_c)
    best_thresh = thresholds[max(best_idx - 1, 0)]
    
    y_pred_raw = (smoothed_score >= best_thresh).astype(int)
    
    # 2. Aggressive Median Filter (kill scattered noise)
    y_pred_med = medfilt(y_pred_raw, kernel_size=15)
    
    # 3. Raw Metrics (Before point-adjust)
    p_raw = precision_score(y_test, y_pred_med, zero_division=0)
    r_raw = recall_score(y_test,    y_pred_med, zero_division=0)
    f_raw = f1_score(y_test,        y_pred_med, zero_division=0)
    
    # 4. Point-Adjusted Metrics (Literature standard)
    y_pred_adj = point_adjust(y_test, y_pred_med)
    p_adj = precision_score(y_test, y_pred_adj, zero_division=0)
    r_adj = recall_score(y_test,    y_pred_adj, zero_division=0)
    f_adj = f1_score(y_test,        y_pred_adj, zero_division=0)
    
    return (round(p_adj, 4), round(r_adj, 4), round(f_adj, 4), int(y_pred_adj.sum()),
            round(p_raw, 4), round(r_raw, 4), round(f_raw, 4))


def extract_standard(model, X_train, X_test):
    """4-feature hybrid extraction (baseline)."""
    tr_p = model.predict(X_train, verbose=0)
    te_p = model.predict(X_test,  verbose=0)

    tr_mse  = np.mean((X_train - tr_p) ** 2, axis=(1, 2))
    te_mse  = np.mean((X_test  - te_p) ** 2, axis=(1, 2))

    tr_err  = np.mean((X_train - tr_p) ** 2, axis=2)
    te_err  = np.mean((X_test  - te_p) ** 2, axis=2)
    tr_diff = np.mean(np.abs(np.diff(tr_err, axis=1)), axis=1)
    te_diff = np.mean(np.abs(np.diff(te_err, axis=1)), axis=1)

    latent_model = tf.keras.Model(inputs=model.input,
                                  outputs=model.get_layer("latent").output)
    tr_lat = latent_model.predict(X_train, verbose=0)
    te_lat = latent_model.predict(X_test,  verbose=0)
    center = np.mean(tr_lat, axis=0)
    tr_lat_score = np.linalg.norm(tr_lat - center, axis=1)
    te_lat_score = np.linalg.norm(te_lat - center, axis=1)

    tr_var = np.var((X_train - tr_p), axis=(1, 2))
    te_var = np.var((X_test  - te_p), axis=(1, 2))

    Xtr = np.stack([tr_mse, tr_diff, tr_lat_score, tr_var], axis=1)
    Xte = np.stack([te_mse, te_diff, te_lat_score, te_var], axis=1)
    sc  = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
    return Xtr, Xte


def extract_perchannel(model, X_train, X_test):
    """12-feature hybrid: per-channel MSE + diff + latent + global var."""
    tr_p = model.predict(X_train, verbose=0)
    te_p = model.predict(X_test,  verbose=0)
    n_ch = X_train.shape[2]

    # Per-channel MSE (9 features)
    tr_ch_mse = np.mean((X_train - tr_p) ** 2, axis=1)   # (N, 9)
    te_ch_mse = np.mean((X_test  - te_p) ** 2, axis=1)   # (N, 9)

    # Global temporal diff
    tr_err  = np.mean((X_train - tr_p) ** 2, axis=2)
    te_err  = np.mean((X_test  - te_p) ** 2, axis=2)
    tr_diff = np.mean(np.abs(np.diff(tr_err, axis=1)), axis=1)
    te_diff = np.mean(np.abs(np.diff(te_err, axis=1)), axis=1)

    # Latent deviation
    latent_model = tf.keras.Model(inputs=model.input,
                                  outputs=model.get_layer("latent").output)
    tr_lat = latent_model.predict(X_train, verbose=0)
    te_lat = latent_model.predict(X_test,  verbose=0)
    center = np.mean(tr_lat, axis=0)
    tr_lat_score = np.linalg.norm(tr_lat - center, axis=1)
    te_lat_score = np.linalg.norm(te_lat - center, axis=1)

    # Global variance
    tr_var = np.var((X_train - tr_p), axis=(1, 2))
    te_var = np.var((X_test  - te_p), axis=(1, 2))

    # Stack: 9 per-channel + diff + latent + var = 12 features
    Xtr = np.hstack([tr_ch_mse, tr_diff.reshape(-1,1), tr_lat_score.reshape(-1,1), tr_var.reshape(-1,1)])
    Xte = np.hstack([te_ch_mse, te_diff.reshape(-1,1), te_lat_score.reshape(-1,1), te_var.reshape(-1,1)])
    sc  = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
    return Xtr, Xte


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  Post-Training Experiment Runner")
    print("=" * 65)

    # Load data
    print(f"\n[INFO] Loading {DATASET.upper()} data...")
    if DATASET == "skab":
        X_train, X_test, y_test = load_skab()
    elif DATASET == "nab":
        X_train, X_test, y_test = load_nab()
    elif DATASET == "smap":
        X_train, X_test, y_test = load_smap()
    elif DATASET == "smd":
        X_train, X_test, y_test = load_data("machine-1-1", dataset="smd")
    else:
        raise ValueError(f"Unknown dataset {DATASET}")
    
    print(f"  Train: {X_train.shape} | Test: {X_test.shape} | Anomaly ratio: {y_test.mean():.4f}")

    # Load model
    model_path = PROJECT_ROOT / "models" / f"federated_{DATASET}_model.keras"
    print(f"\n[INFO] Loading model: {model_path.name}")
    model = tf.keras.models.load_model(str(model_path))

    results = []
    run_ts  = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── Extract features once ─────────────────────────────────────
    print("\n[INFO] Extracting standard features (4-feature)...")
    Xtr_std, Xte_std = extract_standard(model, X_train, X_test)

    print("[INFO] Extracting per-channel features (12-feature)...")
    Xtr_ch, Xte_ch = extract_perchannel(model, X_train, X_test)

    # ═══════════════════════════════════════════════════════════════
    # EXP A: OCSVM nu grid search (standard 4-feature)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("  EXP A: OCSVM nu Grid Search (4-feature)")
    print("=" * 65)
    print(f"  {'nu':>6} | {'Adj Prec':>9} | {'Adj Rec':>7} | {'Adj F1':>7} | {'Raw F1':>7} | {'Predicted':>9}")
    print("  " + "-" * 56)

    best_a_f1 = 0
    for nu in [0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]:
        clf = OneClassSVM(nu=nu, kernel="rbf", gamma="scale")
        clf.fit(Xtr_std)
        sc  = -clf.decision_function(Xte_std)
        p_adj, r_adj, f_adj, n, p_raw, r_raw, f_raw = score(sc, y_test)
        marker = "  << BEST" if f_adj > best_a_f1 else ""
        best_a_f1 = max(best_a_f1, f_adj)
        print(f"  {nu:>6.3f} | {p_adj:>9.4f} | {r_adj:>7.4f} | {f_adj:>7.4f} | {f_raw:>7.4f} | {n:>9}{marker}")
        results.append({"Experiment": "A_OCSVM_4feat", "Param": nu, "Features": 4,
                         "Classifier": "OCSVM", "Precision": p_adj, "Recall": r_adj, "F1": f_adj,
                         "Raw_Precision": p_raw, "Raw_Recall": r_raw, "Raw_F1": f_raw,
                         "Predicted": n, "Timestamp": run_ts})

    # ═══════════════════════════════════════════════════════════════
    # EXP B: Per-channel features (12 features) with OCSVM
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("  EXP B: Per-Channel MSE Features (12-feature) + OCSVM")
    print("=" * 65)
    print(f"  {'nu':>6} | {'Adj Prec':>9} | {'Adj Rec':>7} | {'Adj F1':>7} | {'Raw F1':>7} | {'Predicted':>9}")
    print("  " + "-" * 56)

    best_b_f1 = 0
    for nu in [0.005, 0.01, 0.02, 0.05, 0.10, 0.15]:
        clf = OneClassSVM(nu=nu, kernel="rbf", gamma="scale")
        clf.fit(Xtr_ch)
        sc  = -clf.decision_function(Xte_ch)
        p_adj, r_adj, f_adj, n, p_raw, r_raw, f_raw = score(sc, y_test)
        marker = "  << BEST" if f_adj > best_b_f1 else ""
        best_b_f1 = max(best_b_f1, f_adj)
        print(f"  {nu:>6.3f} | {p_adj:>9.4f} | {r_adj:>7.4f} | {f_adj:>7.4f} | {f_raw:>7.4f} | {n:>9}{marker}")
        results.append({"Experiment": "B_OCSVM_12feat", "Param": nu, "Features": 12,
                         "Classifier": "OCSVM", "Precision": p_adj, "Recall": r_adj, "F1": f_adj,
                         "Raw_Precision": p_raw, "Raw_Recall": r_raw, "Raw_F1": f_raw,
                         "Predicted": n, "Timestamp": run_ts})

    # ═══════════════════════════════════════════════════════════════
    # EXP C: ECOD from pyod (no hyperparameters)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("  EXP C: ECOD Classifier (Extreme Value Theory)")
    print("=" * 65)
    try:
        from pyod.models.ecod import ECOD
        for feat_name, Xtr_f, Xte_f in [("4-feature", Xtr_std, Xte_std),
                                          ("12-feature", Xtr_ch,  Xte_ch)]:
            clf = ECOD()
            clf.fit(Xtr_f)
            sc  = clf.decision_function(Xte_f)
            p_adj, r_adj, f_adj, n, p_raw, r_raw, f_raw = score(sc, y_test)
            print(f"  ECOD ({feat_name}): Adj F1={f_adj:.4f} | Raw F1={f_raw:.4f} | Predicted={n}")
            results.append({"Experiment": f"C_ECOD_{feat_name.replace('-','')}", "Param": "N/A",
                             "Features": int(feat_name.split("-")[0]),
                             "Classifier": "ECOD", "Precision": p_adj, "Recall": r_adj, "F1": f_adj,
                             "Raw_Precision": p_raw, "Raw_Recall": r_raw, "Raw_F1": f_raw,
                             "Predicted": n, "Timestamp": run_ts})
    except ImportError:
        print("  [SKIP] pyod not installed. Run: pip install pyod")
        results.append({"Experiment": "C_ECOD", "Param": "N/A", "Features": "N/A",
                         "Classifier": "ECOD", "Precision": "N/A", "Recall": "N/A",
                         "F1": "SKIPPED - pip install pyod", "Raw_Precision": "N/A", 
                         "Raw_Recall": "N/A", "Raw_F1": "N/A",
                         "Predicted": "N/A", "Timestamp": run_ts})

    # ── Save all results ──────────────────────────────────────────
    df = pd.DataFrame(results)
    out = RESULTS_DIR / f"{DATASET}_experiment_results.csv"
    # Append if file already exists
    if out.exists():
        existing = pd.read_csv(out)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(out, index=False)
    print(f"\n[SAVED] Experiment results -> {out}")

    # Print summary
    numeric_df = df[df["F1"].apply(lambda x: isinstance(x, float))]
    best = numeric_df.loc[numeric_df["F1"].idxmax()]
    print("\n" + "=" * 65)
    print(f"  BEST OVERALL: {best['Experiment']} | param={best['Param']}")
    print(f"  Precision={best['Precision']:.4f} | Recall={best['Recall']:.4f} | F1={best['F1']:.4f}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
