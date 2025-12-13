"""Analyze Naive Bayes performance across different alpha values.

Trains NB models with different alpha values, evaluates on test set,
and visualizes how performance changes, especially for rare labels.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score, f1_score

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import load_fold_frames
from src.data.normalization import DEFAULT_NORMALIZATION_CONFIG_PATH
from src.features.tfidf import create_tfidf_vectorizer
from src.models.tfidf_naive_bayes import train_multilabel_tfidf_naive_bayes
from src.pipeline.train import TrainConfig, _prepare_normalizer, _resolve_text_series
from src.utils.metrics import compute_multilabel_metrics, probs_to_preds

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

RARE_LABELS = ["threat", "identity_hate"]
ALL_LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def train_and_evaluate_nb(
    config: TrainConfig,
    alpha: float,
    fold_name: str,
) -> Dict:
    """Train NB with given alpha and evaluate on test set."""
    
    # Load data
    base_df, fold_frames, identity_cols, _ = load_fold_frames(
        seed=config.seed,
        data_path=config.data_path,
        splits_dir=config.splits_dir,
    )
    label_cols = config.resolve_label_cols(base_df)
    
    if fold_name not in fold_frames:
        raise ValueError(f"Fold '{fold_name}' not found. Available: {list(fold_frames.keys())}")
    
    fold_splits = fold_frames[fold_name]
    
    # Prepare data
    train_df = fold_splits["train"].reset_index(drop=True)
    test_df = fold_splits["test"].reset_index(drop=True)
    
    normalizer, _ = _prepare_normalizer(config)
    X_train = _resolve_text_series(train_df, config.text_col, normalizer, config, None)
    X_test = _resolve_text_series(test_df, config.text_col, normalizer, config, None)
    
    y_train = train_df[label_cols].values.astype(int)
    y_test = test_df[label_cols].values.astype(int)
    
    # Create TF-IDF
    tfidf = create_tfidf_vectorizer(**config.vectorizer_params)
    X_train_vec = tfidf.fit_transform(X_train.tolist())
    X_test_vec = tfidf.transform(X_test.tolist())
    
    # Train models
    print(f"Training NB with alpha={alpha}...")
    nb_params = {"alpha": alpha, "fit_prior": True}
    _, label_models = train_multilabel_tfidf_naive_bayes(
        X_train.tolist(),
        y_train,
        label_cols,
        vectorizer_params=config.vectorizer_params,
        nb_params=nb_params,
    )
    
    # Evaluate on test set
    test_probs = {
        label: model.predict_proba(X_test_vec)[:, 1]
        for label, model in label_models.items()
    }
    y_test_pred = probs_to_preds(test_probs, threshold=config.threshold)
    
    # Overall metrics
    overall_metrics, per_label_df = compute_multilabel_metrics(
        y_test, y_test_pred, label_cols, prob_dict=test_probs
    )
    
    # Compute PR-AUC
    y_test_probs_array = np.array([test_probs[label] for label in label_cols]).T
    try:
        macro_pr_auc = average_precision_score(y_test, y_test_probs_array, average="macro")
        micro_pr_auc = average_precision_score(y_test, y_test_probs_array, average="micro")
    except Exception:
        macro_pr_auc = 0.0
        micro_pr_auc = 0.0
    
    # Per-label metrics (extract from per_label_df)
    per_label_metrics = {}
    for _, row in per_label_df.iterrows():
        label = row["label"]
        per_label_metrics[label] = {
            "f1": float(row["f1"]),
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "pr_auc": float(row.get("pr_auc", 0.0)),
        }
    
    return {
        "alpha": alpha,
        "overall": {
            "macro_f1": overall_metrics["macro_f1"],
            "micro_f1": overall_metrics["micro_f1"],
            "macro_pr_auc": macro_pr_auc,
            "micro_pr_auc": micro_pr_auc,
        },
        "per_label": per_label_metrics,
        "per_label_df": per_label_df,
    }


def plot_alpha_analysis(results: List[Dict], output_dir: Path):
    """Create visualizations for alpha analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract data
    alphas = [r["alpha"] for r in results]
    macro_f1s = [r["overall"]["macro_f1"] for r in results]
    micro_f1s = [r["overall"]["micro_f1"] for r in results]
    macro_pr_aucs = [r["overall"]["macro_pr_auc"] for r in results]
    
    # Plot 1: Overall metrics vs alpha
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # F1 scores
    axes[0].plot(alphas, macro_f1s, marker="o", label="Macro F1", linewidth=2)
    axes[0].plot(alphas, micro_f1s, marker="s", label="Micro F1", linewidth=2)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Alpha (log scale)", fontsize=12)
    axes[0].set_ylabel("F1 Score", fontsize=12)
    axes[0].set_title("Overall F1 Scores vs Alpha", fontsize=14, fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # PR-AUC
    axes[1].plot(alphas, macro_pr_aucs, marker="o", label="Macro PR-AUC", linewidth=2, color="green")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Alpha (log scale)", fontsize=12)
    axes[1].set_ylabel("PR-AUC", fontsize=12)
    axes[1].set_title("Macro PR-AUC vs Alpha", fontsize=14, fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "nb_alpha_overall_metrics.png", dpi=200, bbox_inches="tight")
    print(f"Saved overall metrics plot to {output_dir / 'nb_alpha_overall_metrics.png'}")
    plt.close()
    
    # Plot 2: Rare labels performance
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for label in RARE_LABELS:
        label_f1s = [r["per_label"][label]["f1"] for r in results]
        label_pr_aucs = [r["per_label"][label]["pr_auc"] for r in results]
        
        axes[0].plot(alphas, label_f1s, marker="o", label=label.replace("_", " ").title(), linewidth=2)
        axes[1].plot(alphas, label_pr_aucs, marker="o", label=label.replace("_", " ").title(), linewidth=2)
    
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Alpha (log scale)", fontsize=12)
    axes[0].set_ylabel("F1 Score", fontsize=12)
    axes[0].set_title("Rare Labels: F1 Score vs Alpha", fontsize=14, fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Alpha (log scale)", fontsize=12)
    axes[1].set_ylabel("PR-AUC", fontsize=12)
    axes[1].set_title("Rare Labels: PR-AUC vs Alpha", fontsize=14, fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "nb_alpha_rare_labels.png", dpi=200, bbox_inches="tight")
    print(f"Saved rare labels plot to {output_dir / 'nb_alpha_rare_labels.png'}")
    plt.close()
    
    # Plot 3: All labels F1 comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for label in ALL_LABELS:
        label_f1s = [r["per_label"][label]["f1"] for r in results]
        ax.plot(alphas, label_f1s, marker="o", label=label.replace("_", " ").title(), linewidth=2)
    
    ax.set_xscale("log")
    ax.set_xlabel("Alpha (log scale)", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("All Labels: F1 Score vs Alpha", fontsize=14, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "nb_alpha_all_labels_f1.png", dpi=200, bbox_inches="tight")
    print(f"Saved all labels F1 plot to {output_dir / 'nb_alpha_all_labels_f1.png'}")
    plt.close()
    
    # Plot 4: Heatmap of F1 scores by label and alpha
    f1_matrix = []
    for label in ALL_LABELS:
        label_f1s = [r["per_label"][label]["f1"] for r in results]
        f1_matrix.append(label_f1s)
    
    f1_df = pd.DataFrame(
        f1_matrix,
        index=[l.replace("_", " ").title() for l in ALL_LABELS],
        columns=[f"α={a}" for a in alphas]
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(f1_df, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax, cbar_kws={"label": "F1 Score"})
    ax.set_title("F1 Score Heatmap: Labels vs Alpha", fontsize=14, fontweight="bold")
    ax.set_xlabel("Alpha Value", fontsize=12)
    ax.set_ylabel("Label", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / "nb_alpha_f1_heatmap.png", dpi=200, bbox_inches="tight")
    print(f"Saved F1 heatmap to {output_dir / 'nb_alpha_f1_heatmap.png'}")
    plt.close()


def create_summary_table(results: List[Dict], output_dir: Path):
    """Create summary table of results."""
    summary_data = []
    
    for r in results:
        row = {
            "alpha": r["alpha"],
            "macro_f1": r["overall"]["macro_f1"],
            "micro_f1": r["overall"]["micro_f1"],
            "macro_pr_auc": r["overall"]["macro_pr_auc"],
            "micro_pr_auc": r["overall"]["micro_pr_auc"],
        }
        
        # Add rare label metrics
        for label in RARE_LABELS:
            row[f"{label}_f1"] = r["per_label"][label]["f1"]
            row[f"{label}_pr_auc"] = r["per_label"][label]["pr_auc"]
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save CSV
    csv_path = output_dir / "nb_alpha_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved summary table to {csv_path}")
    
    # Print formatted table
    print("\n" + "="*80)
    print("NAIVE BAYES ALPHA ANALYSIS SUMMARY")
    print("="*80)
    print("\nOverall Metrics:")
    print(summary_df[["alpha", "macro_f1", "micro_f1", "macro_pr_auc", "micro_pr_auc"]].to_string(index=False))
    
    print("\nRare Labels (Threat & Identity Hate):")
    rare_cols = ["alpha"] + [f"{label}_{metric}" for label in RARE_LABELS for metric in ["f1", "pr_auc"]]
    print(summary_df[rare_cols].to_string(index=False))
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Naive Bayes performance across different alpha values"
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[0.01, 0.05, 0.1, 1.0, 5.0],
        help="Alpha values to test (default: 0.01 0.05 0.1 1.0 5.0)",
    )
    parser.add_argument(
        "--fold",
        type=str,
        default="fold1_seed42",
        help="Fold to use (default: fold1_seed42)",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/raw/train.csv"),
        help="Path to training data",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data/splits"),
        help="Directory containing fold JSON files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default="toy",
        choices=["raw", "toy", "rich", "config"],
        help="Text normalization strategy",
    )
    parser.add_argument(
        "--normalization-config",
        type=Path,
        default=DEFAULT_NORMALIZATION_CONFIG_PATH,
        help="Normalization config path",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for predictions",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/nb_alpha_analysis"),
        help="Output directory for results and plots",
    )
    
    args = parser.parse_args()
    
    # Create config
    config = TrainConfig(
        data_path=args.data_path,
        splits_dir=args.splits_dir,
        seed=args.seed,
        normalization=args.normalization,
        normalization_config=args.normalization_config,
        threshold=args.threshold,
    )
    
    print("="*80)
    print("NAIVE BAYES ALPHA ANALYSIS")
    print("="*80)
    print(f"Alpha values: {args.alphas}")
    print(f"Fold: {args.fold}")
    print(f"Normalization: {args.normalization}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Train and evaluate for each alpha
    results = []
    for alpha in args.alphas:
        result = train_and_evaluate_nb(config, alpha, args.fold)
        results.append(result)
        print(f"Alpha {alpha}: Macro F1={result['overall']['macro_f1']:.4f}, "
              f"Micro F1={result['overall']['micro_f1']:.4f}, "
              f"PR-AUC={result['overall']['macro_pr_auc']:.4f}")
    
    # Create visualizations
    plot_alpha_analysis(results, args.output_dir)
    
    # Create summary table
    create_summary_table(results, args.output_dir)
    
    # Save full results as JSON
    json_path = args.output_dir / "nb_alpha_results.json"
    with open(json_path, "w") as f:
        # Convert numpy types to native Python types
        json_results = []
        for r in results:
            json_r = {
                "alpha": float(r["alpha"]),
                "overall": {k: float(v) for k, v in r["overall"].items()},
                "per_label": {
                    label: {k: float(v) for k, v in metrics.items()}
                    for label, metrics in r["per_label"].items()
                },
            }
            json_results.append(json_r)
        json.dump(json_results, f, indent=2)
    
    print(f"\nFull results saved to {json_path}")
    print(f"Analysis complete! Check {args.output_dir} for all outputs.")


if __name__ == "__main__":
    main()

