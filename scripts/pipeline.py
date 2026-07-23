"""
Master Experiment Orchestrator
================================
Run after federated_skab training completes.
Chains ALL experiments in order:
  1. Post-training eval experiments (A, B, C) — ~15 min
  2. Retrain with 25 rounds — ~3.5 hrs
  3. Retrain with lower LR + 20 epochs — ~3.5 hrs
  4. Run post-training eval on each new model
  5. Print final leaderboard and update history
"""
import subprocess
import sys
import os
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR  = PROJECT_ROOT / "results"
HISTORY_PATH = Path(r"C:\Users\mehta\.gemini\antigravity-ide\brain\910d4eb8-27a5-464b-af7d-667bd7482d78\history.md")

DATASET = sys.argv[1] if len(sys.argv) > 1 else "skab"

def run(cmd, label):
    print(f"\n{'='*65}")
    print(f"  RUNNING: {label}")
    print(f"{'='*65}")
    result = subprocess.run(cmd, shell=True, cwd=str(PROJECT_ROOT),
                            capture_output=False, text=True)
    return result.returncode == 0


def update_history(run_id, description, arch, algo, classifier, param,
                   precision, recall, f1, notes=""):
    """Append a result row to history.md."""
    if not HISTORY_PATH.exists():
        return

    entry = f"""
---

## {run_id} — {description}
**Date:** {datetime.now().strftime('%Y-%m-%d')} | **Status:** Complete

**Config:** {arch} | {algo} | {classifier} (param={param})

| Precision | Recall | F1 |
|---|---|---|
| {precision:.4f} | {recall:.4f} | {f1:.4f} |

**Notes:** {notes}
"""
    with open(HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(entry)
    print(f"[HISTORY] Appended {run_id} to history.md")


def get_best_result(experiment_filter=None):
    """Read experiment_results.csv and return the best F1 row."""
    path = RESULTS_DIR / f"{DATASET}_experiment_results.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[pd.to_numeric(df["F1"], errors="coerce").notna()]
    df["F1"] = df["F1"].astype(float)
    if experiment_filter:
        df = df[df["Experiment"].str.startswith(experiment_filter)]
    if df.empty:
        return None
    return df.loc[df["F1"].idxmax()]


def swap_training_config(rounds=15, lr=1e-4, epochs=15):
    """Patch train.py hyperparameters in-place."""
    train_path = PROJECT_ROOT / "scripts" / "train.py"
    content = train_path.read_text(encoding="utf-8")

    import re
    content = re.sub(r"^EPOCHS\s*=\s*\d+", f"EPOCHS = {epochs}", content, flags=re.MULTILINE)
    content = re.sub(r"^ROUNDS\s*=\s*\d+", f"ROUNDS = {rounds}", content, flags=re.MULTILINE)
    content = re.sub(r"learning_rate=[\d.e\-]+, clipnorm",
                     f"learning_rate={lr}, clipnorm", content)
    train_path.write_text(content, encoding="utf-8")
    print(f"[CONFIG] Set ROUNDS={rounds}, EPOCHS={epochs}, lr={lr}")


def main():
    print("\n" + "=" * 65)
    print("  MASTER EXPERIMENT ORCHESTRATOR")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # ── PHASE 1: Eval experiments on current model ────────────────
    print(f"\n[PHASE 1] Running evaluation experiments (A, B, C) for {DATASET.upper()}...")
    run(f"python scripts/evaluate.py {DATASET}", f"Eval Experiments A+B+C ({DATASET})")

    best_eval = get_best_result()
    if best_eval is not None:
        update_history(
            run_id=f"{DATASET.upper()} - Run 005",
            description=f"Eval Experiments on CNN-LSTM + FedProx Model ({DATASET})",
            arch="CNN-LSTM (no Attention)",
            algo="FedProx (mu=0.01), 15 rounds",
            classifier=best_eval["Classifier"],
            param=best_eval["Param"],
            precision=float(best_eval["Precision"]),
            recall=float(best_eval["Recall"]),
            f1=float(best_eval["F1"]),
            notes=f"Best experiment: {best_eval['Experiment']} with {best_eval['Features']} features. "
                  f"Grid search over OCSVM nu values + per-channel MSE (12 features) + ECOD."
        )

    # ── PHASE 2: Retrain with 25 rounds ──────────────────────────
    print(f"\n[PHASE 2] Retraining {DATASET.upper()} with 25 federated rounds...")
    swap_training_config(rounds=25, lr=1e-4, epochs=15)
    success = run(f"python scripts/train.py {DATASET}", f"Retrain: 25 rounds, lr=1e-4, epochs=15 ({DATASET})")

    if success:
        print(f"[PHASE 2] Running eval on 25-round model for {DATASET}...")
        run(f"python scripts/evaluate.py {DATASET}", f"Eval after 25 rounds ({DATASET})")
        best_25r = get_best_result()
        if best_25r is not None:
            update_history(
                run_id=f"{DATASET.upper()} - Run 006",
                description=f"CNN-LSTM + FedProx, 25 Rounds ({DATASET})",
                arch="CNN-LSTM (no Attention)",
                algo="FedProx (mu=0.01), 25 rounds, epochs=15, lr=1e-4",
                classifier=best_25r["Classifier"],
                param=best_25r["Param"],
                precision=float(best_25r["Precision"]),
                recall=float(best_25r["Recall"]),
                f1=float(best_25r["F1"]),
                notes="Increased federated rounds from 15 to 25 to test if further convergence improves anomaly boundaries."
            )

    # ── PHASE 3: Retrain with lower LR + more local epochs ───────
    print(f"\n[PHASE 3] Retraining {DATASET.upper()} with lower LR + 20 local epochs...")
    swap_training_config(rounds=15, lr=5e-5, epochs=20)
    success = run(f"python scripts/train.py {DATASET}", f"Retrain: 15 rounds, lr=5e-5, epochs=20 ({DATASET})")

    if success:
        print(f"[PHASE 3] Running eval on low-LR model for {DATASET}...")
        run(f"python scripts/evaluate.py {DATASET}", f"Eval after low-LR retrain ({DATASET})")
        best_llr = get_best_result()
        if best_llr is not None:
            update_history(
                run_id=f"{DATASET.upper()} - Run 007",
                description=f"CNN-LSTM + FedProx, Lower LR + More Local Epochs ({DATASET})",
                arch="CNN-LSTM (no Attention)",
                algo="FedProx (mu=0.01), 15 rounds, epochs=20, lr=5e-5",
                classifier=best_llr["Classifier"],
                param=best_llr["Param"],
                precision=float(best_llr["Precision"]),
                recall=float(best_llr["Recall"]),
                f1=float(best_llr["F1"]),
                notes="Lower learning rate allows finer gradient steps. "
                      "More local epochs lets clients converge further before aggregation."
            )

    # ── Reset config back to defaults ────────────────────────────
    swap_training_config(rounds=15, lr=1e-4, epochs=15)

    # ── Final leaderboard ─────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"  FINAL LEADERBOARD for {DATASET.upper()}")
    print("=" * 65)
    df = pd.read_csv(RESULTS_DIR / f"{DATASET}_experiment_results.csv")
    df = df[pd.to_numeric(df["F1"], errors="coerce").notna()]
    df["F1"] = df["F1"].astype(float)
    top = df.sort_values("F1", ascending=False).head(10)
    print(top[["Experiment", "Param", "Features", "Classifier", "Precision", "Recall", "F1", "Timestamp"]].to_string(index=False))
    print("=" * 65)
    print(f"\n[DONE] All experiments complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
