"""Generate visualizations for a specific model.

This script creates plots for a single model:
- Metrics table (PR-AUC, F1 macro/micro, Hamming Loss)
- Per-label F1 bar chart
- PR curves
- Recall at 90% Precision
- Precision at Top 1000

Plots are saved to the results/ directory.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


def load_model_metrics(experiments_dir: Path, model_name: str) -> Dict:
    """Load summary metrics for a model across all folds."""
    model_dir = experiments_dir / model_name
    summary_path = model_dir / "summary_metrics.json"
    
    if not summary_path.exists():
        return {}
    
    with open(summary_path, "r") as f:
        summary = json.load(f)
    
    # Aggregate across folds
    metrics = {}
    for fold_name, fold_metrics in summary.items():
        for metric_name, value in fold_metrics.items():
            if metric_name not in metrics:
                metrics[metric_name] = []
            metrics[metric_name].append(value)
    
    # Average across folds
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    std_metrics = {k: np.std(v) for k, v in metrics.items()}
    
    return {"mean": avg_metrics, "std": std_metrics, "folds": summary}


def load_per_label_metrics(experiments_dir: Path, model_name: str) -> pd.DataFrame:
    """Load and aggregate per-label metrics across all folds."""
    model_dir = experiments_dir / model_name
    fold_dirs = sorted(model_dir.glob("fold*"))
    
    all_per_label = []
    for fold_dir in fold_dirs:
        per_label_path = fold_dir / "per_label_metrics.csv"
        if per_label_path.exists():
            df = pd.read_csv(per_label_path)
            df["fold"] = fold_dir.name
            df["model"] = model_name
            all_per_label.append(df)
    
    if not all_per_label:
        return pd.DataFrame()
    
    combined = pd.concat(all_per_label, ignore_index=True)
    # Average across folds
    avg_df = combined.groupby("label").agg({
        "precision": "mean",
        "recall": "mean",
        "f1": "mean",
        "support": "mean",
    }).reset_index()
    avg_df["model"] = model_name
    
    return avg_df


def load_test_predictions(experiments_dir: Path, model_name: str, fold: str = "fold1") -> Optional[pd.DataFrame]:
    """Load test predictions for a specific fold."""
    model_dir = experiments_dir / model_name
    fold_dirs = sorted(model_dir.glob(f"{fold}*"))
    
    if not fold_dirs:
        return None
    
    pred_path = fold_dirs[0] / "test_predictions.csv"
    if not pred_path.exists():
        return None
    
    return pd.read_csv(pred_path)


def create_metrics_table(model_data: Dict, model_name: str, output_dir: Path):
    """Create a metrics table for the model."""
    mean_metrics = model_data["mean"]
    std_metrics = model_data["std"]
    
    metrics = {
        "Metric": [
            "PR-AUC (Macro)",
            "F1 Score (Micro)",
            "F1 Score (Macro)",
            "Hamming Loss"
        ],
        "Mean": [
            mean_metrics.get("macro_pr_auc", mean_metrics.get("pr_auc", 0)),
            mean_metrics.get("micro_f1", 0),
            mean_metrics.get("macro_f1", 0),
            mean_metrics.get("hamming_loss", 0)
        ],
        "Std": [
            std_metrics.get("macro_pr_auc", std_metrics.get("pr_auc", 0)),
            std_metrics.get("micro_f1", 0),
            std_metrics.get("macro_f1", 0),
            std_metrics.get("hamming_loss", 0)
        ]
    }
    
    df = pd.DataFrame(metrics)
    df["Value"] = df.apply(lambda row: f"{row['Mean']:.4f} ± {row['Std']:.4f}", axis=1)
    df_display = df[["Metric", "Value"]]
    
    # Create table visualization
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("tight")
    ax.axis("off")
    
    table = ax.table(
        cellText=df_display.values,
        colLabels=df_display.columns,
        cellLoc="left",
        loc="center",
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    # Style header
    for i in range(len(df_display.columns)):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")
    
    plt.title(f"Metrics Summary: {model_name.upper()}", fontsize=14, fontweight="bold", pad=20)
    plt.savefig(output_dir / "metrics_table.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Also save as CSV
    df[["Metric", "Mean", "Std"]].to_csv(output_dir / "metrics_table.csv", index=False)
    print(f"✅ Saved: {output_dir / 'metrics_table.png'}")
    print(f"✅ Saved: {output_dir / 'metrics_table.csv'}")




def plot_per_label_f1(per_label_df: pd.DataFrame, model_name: str, output_dir: Path):
    """Plot per-label F1 bar chart for the model."""
    if per_label_df.empty:
        print("⚠️  No per-label data found")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(per_label_df["label"], per_label_df["f1"], alpha=0.8, color="steelblue")
    ax.set_xlabel("Label", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title(f"Per-Label F1 Score: {model_name.upper()}", fontsize=14, fontweight="bold")
    ax.set_xticklabels(per_label_df["label"], rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim([0, max(per_label_df["f1"]) * 1.2 if len(per_label_df) > 0 else 1.0])
    
    # Add value labels
    for i, (label, f1) in enumerate(zip(per_label_df["label"], per_label_df["f1"])):
        ax.text(i, f1 + 0.01, f"{f1:.3f}", ha="center", va="bottom", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / "per_label_f1.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {output_dir / 'per_label_f1.png'}")


def plot_pr_curves(model_name: str, experiments_dir: Path, 
                   output_dir: Path, fold: str = "fold1"):
    """Plot Precision-Recall curves for the model."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(label_cols)))
    
    pred_df = load_test_predictions(experiments_dir, model_name, fold)
    if pred_df is None:
        print(f"⚠️  No predictions found for {model_name}")
        return
    
    # Compute PR curves for each label
    all_precisions = []
    all_recalls = []
    label_pr_aucs = []
    
    for label, color in zip(label_cols, colors):
        true_col = label
        prob_col = f"{label}_prob"
        
        if true_col not in pred_df.columns or prob_col not in pred_df.columns:
            continue
        
        y_true = pred_df[true_col].values
        y_prob = pred_df[prob_col].values
        
        if len(np.unique(y_true)) < 2:
            continue
        
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        all_precisions.append(precision)
        all_recalls.append(recall)
        
        # Calculate PR-AUC for this label
        try:
            pr_auc = average_precision_score(y_true, y_prob)
            label_pr_aucs.append(pr_auc)
        except:
            pr_auc = 0.0
        
        # Plot individual label curve
        ax.plot(recall, precision, label=f"{label} (AUC={pr_auc:.3f})", 
                linewidth=2, color=color, alpha=0.7)
    
    # Compute macro-averaged curve
    if all_precisions:
        recall_interp = np.linspace(0, 1, 100)
        precision_interp = []
        
        for prec, rec in zip(all_precisions, all_recalls):
            prec = np.array(prec)[::-1]
            rec = np.array(rec)[::-1]
            prec_interp = np.interp(recall_interp, rec, prec, left=1.0, right=0.0)
            precision_interp.append(prec_interp)
        
        mean_precision = np.mean(precision_interp, axis=0)
        macro_pr_auc = np.mean(label_pr_aucs) if label_pr_aucs else 0.0
        
        ax.plot(recall_interp, mean_precision, label=f"Macro-Avg (AUC={macro_pr_auc:.3f})", 
                linewidth=3, color="black", linestyle="--")
    
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"Precision-Recall Curves: {model_name.upper()}", fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_dir / "pr_curves.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {output_dir / 'pr_curves.png'}")


def plot_recall_at_90_precision(model_name: str, experiments_dir: Path,
                                output_dir: Path, fold: str = "fold1"):
    """Plot Recall at 90% Precision for the model."""
    label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    
    pred_df = load_test_predictions(experiments_dir, model_name, fold)
    if pred_df is None:
        print(f"⚠️  No predictions found for {model_name}")
        return
    
    results = []
    for label in label_cols:
        true_col = label
        prob_col = f"{label}_prob"
        
        if true_col not in pred_df.columns or prob_col not in pred_df.columns:
            continue
        
        y_true = pred_df[true_col].values
        y_prob = pred_df[prob_col].values
        
        if len(np.unique(y_true)) < 2:
            continue
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        
        # Find recall at 90% precision
        target_precision = 0.90
        valid_idx = np.where(precision >= target_precision)[0]
        
        if len(valid_idx) > 0:
            recall_at_90 = recall[valid_idx].max()
        else:
            recall_at_90 = 0.0
        
        results.append({
            "label": label,
            "recall_at_90": recall_at_90
        })
    
    if not results:
        print("⚠️  No data for recall at 90% precision")
        return
    
    df = pd.DataFrame(results)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(df["label"], df["recall_at_90"], alpha=0.8, color="steelblue")
    ax.set_xlabel("Label", fontsize=12)
    ax.set_ylabel("Recall at 90% Precision", fontsize=12)
    ax.set_title(f"Recall at 90% Precision: {model_name.upper()}", fontsize=14, fontweight="bold")
    ax.set_xticklabels(df["label"], rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_ylim([0, max(df["recall_at_90"]) * 1.2 if len(df) > 0 else 1.0])
    
    # Add value labels
    for i, (label, recall) in enumerate(zip(df["label"], df["recall_at_90"])):
        ax.text(i, recall + 0.01, f"{recall:.3f}", ha="center", va="bottom", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / "recall_at_90_precision.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {output_dir / 'recall_at_90_precision.png'}")


def plot_precision_at_top1000(model_data: Dict, model_name: str, output_dir: Path):
    """Plot Precision at Top 1000 for the model."""
    mean_metrics = model_data["mean"]
    std_metrics = model_data["std"]
    
    prec_at_1000 = mean_metrics.get("precision_at_1000", 0)
    prec_at_1000_std = std_metrics.get("precision_at_1000", 0)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar([model_name], [prec_at_1000], yerr=[prec_at_1000_std], 
                   alpha=0.8, capsize=10, color="coral", width=0.5)
    ax.set_ylabel("Precision at Top 1000", fontsize=12)
    ax.set_title(f"Precision at Top 1000: {model_name.upper()}", fontsize=14, fontweight="bold")
    ax.set_xticklabels([model_name])
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim([0, max(prec_at_1000 * 1.2, 0.1) if prec_at_1000 > 0 else 1.0])
    
    # Add value label
    ax.text(0, prec_at_1000 + prec_at_1000_std + 0.01, f"{prec_at_1000:.3f} ± {prec_at_1000_std:.3f}", 
            ha="center", va="bottom", fontsize=11, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(output_dir / "precision_at_top1000.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {output_dir / 'precision_at_top1000.png'}")




def main():
    parser = argparse.ArgumentParser(description="Generate visualizations for a specific model")
    parser.add_argument("--model", type=str, required=True,
                       help="Model name (e.g., lstm, naive_bayes, logistic)")
    parser.add_argument("--experiments-dir", type=Path, default=Path("experiments"),
                       help="Directory containing experiment results")
    parser.add_argument("--output-dir", type=Path, default=Path("results"),
                       help="Output directory for plots")
    parser.add_argument("--fold", type=str, default="fold1",
                       help="Fold to use for PR curves and recall@90 (default: fold1)")
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = args.model
    model_dir = args.experiments_dir / model_name
    
    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_dir}")
        return
    
    if not (model_dir / "summary_metrics.json").exists():
        print(f"❌ summary_metrics.json not found for {model_name}")
        return
    
    print(f"📊 Loading metrics for {model_name}...")
    
    # Load metrics
    model_data = load_model_metrics(args.experiments_dir, model_name)
    per_label_df = load_per_label_metrics(args.experiments_dir, model_name)
    
    if not model_data:
        print(f"❌ No metrics found for {model_name}")
        return
    
    # Generate all plots
    print("\n📈 Generating visualizations...")
    create_metrics_table(model_data, model_name, args.output_dir)
    plot_per_label_f1(per_label_df, model_name, args.output_dir)
    plot_pr_curves(model_name, args.experiments_dir, args.output_dir, args.fold)
    plot_recall_at_90_precision(model_name, args.experiments_dir, args.output_dir, args.fold)
    plot_precision_at_top1000(model_data, model_name, args.output_dir)
    
    print(f"\n✅ All plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

