#!/usr/bin/env python3
"""
Generate calibration plots for toxic comment classification models.

Calibration plots show how well predicted probabilities match actual outcomes.
A well-calibrated model should have predictions that align with the diagonal line.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

def load_model_predictions(model_name: str) -> pd.DataFrame:
    """Load predictions for a specific model."""
    exp_dir = Path("experiments/train") / model_name
    if not exp_dir.exists():
        return None

    if model_name == "bert":
        fold_dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir() and not d.name.startswith('.')],
                          key=lambda x: x.name, reverse=True)
        if fold_dirs:
            latest_fold = fold_dirs[0]
            timestamp_dirs = sorted([d for d in latest_fold.iterdir() if d.is_dir() and not d.name.startswith('.')],
                                   key=lambda x: x.name, reverse=True)
            if timestamp_dirs:
                pred_file = timestamp_dirs[0] / "test_predictions.csv"
            else:
                return None
    else:
        timestamp_dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir() and not d.name.startswith('.')],
                               key=lambda x: x.name, reverse=True)
        if timestamp_dirs:
            pred_file = timestamp_dirs[0] / "test_predictions.csv"
        else:
            return None

    if pred_file.exists():
        return pd.read_csv(pred_file)
    return None

def create_calibration_plots():
    """Create calibration plots for all models."""

    print("🔍 Generating calibration plots...")

    # Create output directory
    plots_dir = Path("reports/analysis")
    plots_dir.mkdir(exist_ok=True, parents=True)

    models = ["tfidf_logistic", "tfidf_svm", "tfidf_random_forest", "bert"]
    calibration_results = {}

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, label in enumerate(LABEL_COLS):
        ax = axes[idx]

        for model_name in models:
            pred_df = load_model_predictions(model_name)
            if pred_df is None:
                continue

            prob_col = f"{label}_prob"
            true_col = label

            if prob_col in pred_df.columns and true_col in pred_df.columns:
                y_true = pred_df[true_col].values
                y_prob = pred_df[prob_col].values

                # Compute calibration curve
                prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='quantile')

                # Plot calibration curve
                model_display_name = model_name.replace('tfidf_', '').replace('_', ' ').title()
                ax.plot(prob_pred, prob_true, marker='o', label=model_display_name, alpha=0.8)

                # Compute Brier score
                brier = brier_score_loss(y_true, y_prob)

                # Store results
                if model_name not in calibration_results:
                    calibration_results[model_name] = {}
                calibration_results[model_name][label] = {
                    'brier_score': float(brier),
                    'ece': float(np.mean(np.abs(prob_pred - prob_true)))  # Expected Calibration Error
                }

        # Plot perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', alpha=0.5)

        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title(f'Calibration Plot - {label.replace("_", " ").title()}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / "calibration_plots.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Create summary table plot
    fig, ax = plt.subplots(figsize=(12, 8))

    summary_data = []
    for model_name in models:
        if model_name in calibration_results:
            for label in LABEL_COLS:
                if label in calibration_results[model_name]:
                    summary_data.append({
                        'Model': model_name.replace('tfidf_', '').replace('_', ' ').title(),
                        'Label': label.replace('_', ' ').title(),
                        'Brier Score': calibration_results[model_name][label]['brier_score'],
                        'ECE': calibration_results[model_name][label]['ece']
                    })

    if summary_data:
        summary_df = pd.DataFrame(summary_data)

        # Plot Brier scores
        sns.barplot(data=summary_df, x='Label', y='Brier Score', hue='Model', ax=ax)
        ax.set_title('Calibration Quality: Brier Score by Model and Label\n(Lower is Better)')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plots_dir / "calibration_summary.png", dpi=300, bbox_inches='tight')
        plt.close()

    # Save calibration results
    with open(plots_dir / "calibration_metrics.json", 'w') as f:
        json.dump(calibration_results, f, indent=2)

    print("✅ Calibration plots generated!")
    print(f"📁 Saved to: {plots_dir}")
    print("\n📊 Generated files:")
    print("- calibration_plots.png: Individual calibration curves for each label")
    print("- calibration_summary.png: Brier score comparison across models")
    print("- calibration_metrics.json: Detailed calibration metrics")

    # Print summary
    print("\n📈 Calibration Summary:")
    for model_name in models:
        if model_name in calibration_results:
            avg_brier = np.mean([calibration_results[model_name][label]['brier_score']
                               for label in calibration_results[model_name]])
            avg_ece = np.mean([calibration_results[model_name][label]['ece']
                             for label in calibration_results[model_name]])
            print(f"  {model_name}: Avg Brier={avg_brier:.4f}, Avg ECE={avg_ece:.4f}")
if __name__ == "__main__":
    create_calibration_plots()