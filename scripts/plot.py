"""
Generate publication-quality plots for the research paper.
Reads results CSVs from results/ and produces high-DPI figures.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.append(str(Path(__file__).resolve().parents[1]))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Publication style
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

COLORS = ['#2196F3', '#FF5722', '#4CAF50', '#FFC107', '#9C27B0', '#00BCD4']


def plot_baselines_comparison():
    """Bar chart comparing F1 scores across methods."""
    csv_path = RESULTS_DIR / "skab_baselines_summary.csv"
    if not csv_path.exists():
        print(f"[SKIP] {csv_path} not found")
        return
    
    df = pd.read_csv(csv_path)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    methods = df['Method'].values
    f1_scores = df['F1'].values
    precisions = df['Precision'].values
    recalls = df['Recall'].values
    
    x = np.arange(len(methods))
    width = 0.25
    
    bars1 = ax.bar(x - width, precisions, width, label='Precision', color=COLORS[0], alpha=0.85)
    bars2 = ax.bar(x, recalls, width, label='Recall', color=COLORS[1], alpha=0.85)
    bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', color=COLORS[2], alpha=0.85)
    
    ax.set_xlabel('Method')
    ax.set_ylabel('Score')
    ax.set_title('Anomaly Detection Performance Comparison (SKAB Dataset)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha='right', fontsize=9)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "baselines_comparison.png")
    plt.savefig(FIGURES_DIR / "baselines_comparison.pdf")
    print("[OK] Saved baselines_comparison.png/pdf")
    plt.close()


def plot_convergence():
    """Line plot showing F1 / Val Loss vs. federated round number."""
    csv_path = RESULTS_DIR / "skab_federated_history.csv"
    if not csv_path.exists():
        print(f"[SKIP] {csv_path} not found")
        return
    
    df = pd.read_csv(csv_path)
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    ax1.plot(df['Round'], df['Val Loss'], 'o-', color=COLORS[0], linewidth=2, markersize=5, label='Val Loss (MSE)')
    ax1.set_xlabel('Federated Round')
    ax1.set_ylabel('Validation Loss (MSE)', color=COLORS[0])
    ax1.tick_params(axis='y', labelcolor=COLORS[0])
    ax1.grid(alpha=0.3)
    
    # If F1 scores are available (from final round), annotate
    if 'F1 Score' in df.columns:
        f1_rows = df.dropna(subset=['F1 Score'])
        if len(f1_rows) > 0:
            ax2 = ax1.twinx()
            ax2.plot(f1_rows['Round'], f1_rows['F1 Score'], 's-', color=COLORS[2], linewidth=2, markersize=8, label='F1-Score')
            ax2.set_ylabel('F1-Score', color=COLORS[2])
            ax2.tick_params(axis='y', labelcolor=COLORS[2])
    
    ax1.set_title('Federated Training Convergence (SKAB Dataset)')
    fig.legend(loc='upper right', bbox_to_anchor=(0.88, 0.88))
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "convergence.png")
    plt.savefig(FIGURES_DIR / "convergence.pdf")
    print("[OK] Saved convergence.png/pdf")
    plt.close()


def plot_quantization_profile():
    """Scatter plot showing Latency vs. F1 vs. Model Size."""
    csv_path = RESULTS_DIR / "skab_quantization_profile.csv"
    if not csv_path.exists():
        print(f"[SKIP] {csv_path} not found")
        return
    
    df = pd.read_csv(csv_path)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Bar chart of model sizes
    models = df['Model Type'].values
    sizes = df['Size (KB)'].values
    latencies = df['Avg Latency (ms)'].values
    f1s = df['F1 Score'].values
    
    colors = [COLORS[i % len(COLORS)] for i in range(len(models))]
    
    bars = ax1.barh(models, sizes, color=colors, alpha=0.85)
    ax1.set_xlabel('Model Size (KB)')
    ax1.set_title('Model Size Comparison')
    ax1.grid(axis='x', alpha=0.3)
    for bar, size in zip(bars, sizes):
        ax1.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, 
                f'{size:.0f} KB', va='center', fontsize=9)
    
    # Right: Latency vs F1 scatter
    for i, (model, lat, f1) in enumerate(zip(models, latencies, f1s)):
        ax2.scatter(lat, f1, s=150, color=colors[i], zorder=5, edgecolors='black', linewidth=0.5)
        ax2.annotate(model.replace('TFLite ', '').replace('Keras ', ''), 
                    (lat, f1), textcoords="offset points",
                    xytext=(8, 5), fontsize=8)
    
    ax2.set_xlabel('Average Latency (ms)')
    ax2.set_ylabel('F1-Score')
    ax2.set_title('Latency vs. Accuracy Trade-off')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "quantization_profile.png")
    plt.savefig(FIGURES_DIR / "quantization_profile.pdf")
    print("[OK] Saved quantization_profile.png/pdf")
    plt.close()


def plot_ablation():
    """Bar chart showing ablation study results."""
    csv_path = RESULTS_DIR / "skab_baselines_summary.csv"
    if not csv_path.exists():
        print(f"[SKIP] {csv_path} not found")
        return
    
    df = pd.read_csv(csv_path)
    
    # Filter to ablation rows only
    ablation_df = df[df['Method'].str.contains('Ablation|Full Proposed', case=False)]
    
    if len(ablation_df) == 0:
        print("[SKIP] No ablation data found")
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    methods = ablation_df['Method'].values
    # Shorten labels
    short_labels = []
    for m in methods:
        m = m.replace('Ablation ', '').replace('(', '').replace(')', '').replace('+ OCSVM', '')
        m = m.replace('Full Proposed Hybrid Model', 'Full Model\n(All 4 Features)')
        short_labels.append(m.strip())
    
    f1_scores = ablation_df['F1'].values
    
    bars = ax.bar(range(len(short_labels)), f1_scores, color=COLORS[:len(short_labels)], alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(short_labels)))
    ax.set_xticklabels(short_labels, rotation=15, ha='right', fontsize=9)
    ax.set_ylabel('F1-Score')
    ax.set_title('Ablation Study: Feature Combination Impact')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, f1 in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
               f'{f1:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ablation_study.png")
    plt.savefig(FIGURES_DIR / "ablation_study.pdf")
    print("[OK] Saved ablation_study.png/pdf")
    plt.close()


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=== Generating Publication Plots ===\n")
    
    plot_baselines_comparison()
    plot_convergence()
    plot_quantization_profile()
    plot_ablation()
    
    print(f"\n[DONE] All plots saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
