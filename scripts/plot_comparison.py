import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
from pathlib import Path
import pandas as pd
from typing import Dict

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def load_model_metrics(experiments_dir: Path, model_name: str) -> Dict:
    """Load summary metrics for a model across all folds."""
    model_dir = experiments_dir / model_name
    
    summary_path = model_dir / "summary_metrics.json"
    if summary_path.exists():
        # For TF-IDF models, summary is at model level
        with open(summary_path, "r") as f:
            summary = json.load(f)
    else:
        # For BERT, load from each fold's summary_metrics.json
        summary = {}
        for fold_dir in sorted(model_dir.iterdir()):
            if fold_dir.is_dir():
                fold_summary_path = fold_dir / "summary_metrics.json"
                if fold_summary_path.exists():
                    with open(fold_summary_path, "r") as f:
                        fold_summary = json.load(f)
                    # fold_summary is {fold_name: metrics}, but since it's per fold, take the value
                    fold_name = list(fold_summary.keys())[0]
                    summary[fold_name] = fold_summary[fold_name]
    
    if not summary:
        return {}
    
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

def create_comparison_table(models_data: Dict, output_dir: Path):
    """Create a comparison table for all models."""
    rows = []
    for model_name, data in models_data.items():
        mean_metrics = data["mean"]
        std_metrics = data["std"]
        row = {
            "Model": model_name.upper(),
            "Macro F1": f"{mean_metrics.get('macro_f1', 0):.4f} ± {std_metrics.get('macro_f1', 0):.4f}",
            "Micro F1": f"{mean_metrics.get('micro_f1', 0):.4f} ± {std_metrics.get('micro_f1', 0):.4f}",
            "Macro Precision": f"{mean_metrics.get('macro_precision', 0):.4f} ± {std_metrics.get('macro_precision', 0):.4f}",
            "Macro Recall": f"{mean_metrics.get('macro_recall', 0):.4f} ± {std_metrics.get('macro_recall', 0):.4f}",
            "Hamming Loss": f"{mean_metrics.get('hamming_loss', 0):.4f} ± {std_metrics.get('hamming_loss', 0):.4f}",
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Create table visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("tight")
    ax.axis("off")
    
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")
    
    plt.title("Model Comparison: Evaluation Metrics", fontsize=14, fontweight="bold", pad=20)
    plt.savefig(output_dir / "model_comparison_table.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Also save as CSV
    df.to_csv(output_dir / "model_comparison_table.csv", index=False)
    print(f"✅ Saved: {output_dir / 'model_comparison_table.png'}")
    print(f"✅ Saved: {output_dir / 'model_comparison_table.csv'}")

def plot_macro_f1_comparison(models_data: Dict, output_dir: Path):
    """Plot Macro F1 comparison across models."""
    models = list(models_data.keys())
    means = [models_data[m]["mean"].get("macro_f1", 0) for m in models]
    stds = [models_data[m]["std"].get("macro_f1", 0) for m in models]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, means, yerr=stds, capsize=5, alpha=0.8, color=['lightgreen', 'lightcoral', 'lightblue', 'skyblue'])
    ax.set_ylabel("Macro F1 Score", fontsize=12)
    ax.set_title("Macro F1 Score Comparison Across Models", fontsize=14, fontweight="bold")
    ax.set_xticklabels([m.upper() for m in models], rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim([0, max(means) * 1.2 if means else 1.0])
    
    # Add value labels
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 0.01, f"{mean:.3f} ± {std:.3f}", ha="center", va="bottom", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / "macro_f1_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {output_dir / 'macro_f1_comparison.png'}")

def plot_per_fold_comparison(models_data: Dict, output_dir: Path):
    """Plot per-fold Macro F1 for each model."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = ['lightgreen', 'lightcoral', 'lightblue', 'skyblue']
    
    for idx, (model_name, data) in enumerate(models_data.items()):
        ax = axes[idx]
        folds = list(data["folds"].keys())
        f1_scores = [data["folds"][fold].get("macro_f1", 0) for fold in folds]
        
        bars = ax.bar(range(len(folds)), f1_scores, alpha=0.8, color=colors[idx])
        ax.set_title(f"{model_name.upper()}: Macro F1 per Fold", fontsize=12, fontweight="bold")
        ax.set_ylabel("Macro F1 Score")
        ax.set_xticks(range(len(folds)))
        ax.set_xticklabels(folds, rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim([0, max(f1_scores) * 1.2 if f1_scores else 1.0])
        
        # Add value labels
        for i, score in enumerate(f1_scores):
            ax.text(i, score + 0.01, f"{score:.3f}", ha="center", va="bottom", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / "per_fold_macro_f1.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {output_dir / 'per_fold_macro_f1.png'}")

def main():
    experiments_dir = Path("experiments/train")
    models = ['tfidf_logistic', 'tfidf_svm', 'tfidf_random_forest', 'bert']
    output_dir = Path("reports/Plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("📊 Loading metrics for all models...")
    models_data = {}
    for model in models:
        print(f"   Loading {model}...")
        data = load_model_metrics(experiments_dir, model)
        if data:
            models_data[model] = data
        else:
            print(f"   ❌ No data for {model}")
    
    if not models_data:
        print("❌ No model data found")
        return
    
    print(f"\n📈 Generating comparison plots...")
    print(f"   Output directory: {output_dir}")
    
    create_comparison_table(models_data, output_dir)
    plot_macro_f1_comparison(models_data, output_dir)
    plot_per_fold_comparison(models_data, output_dir)
    
    print(f"\n✅ All comparison plots saved to: {output_dir}")

if __name__ == "__main__":
    main()