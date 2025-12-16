#!/usr/bin/env python3
"""
Generate comprehensive plots for extended evaluation analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

def create_extended_evaluation_plots():
    """Create plots for the extended evaluation folder."""

    extended_dir = Path("reports/extended_evaluation")
    plots_dir = extended_dir
    plots_dir.mkdir(exist_ok=True)

    # 1. Out-of-domain performance comparison
    print("📊 Creating out-of-domain evaluation plot...")

    with open(extended_dir / "extended_evaluation_report.json", 'r') as f:
        extended_data = json.load(f)

    ood_results = extended_data["sections"]["out_of_domain_evaluation"]["results"]

    ood_data = []
    for label, metrics in ood_results.items():
        if not np.isnan(metrics.get("auc", np.nan)):  # Skip NaN values
            ood_data.append({
                'Label': label.replace('_', ' ').title(),
                'AUC': metrics.get("auc", 0),
                'Precision': metrics.get("precision", 0),
                'Recall': metrics.get("recall", 0),
                'F1': metrics.get("f1", 0),
                'Support': metrics.get("support", 0)
            })

    if ood_data:
        ood_df = pd.DataFrame(ood_data)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # AUC scores
        sns.barplot(data=ood_df, x='Label', y='AUC', ax=axes[0,0])
        axes[0,0].set_title('AUC Scores on Out-of-Domain Data')
        axes[0,0].set_xticklabels(axes[0,0].get_xticklabels(), rotation=45)
        axes[0,0].set_ylabel('AUC Score')
        axes[0,0].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random')
        axes[0,0].legend()

        # F1 scores
        sns.barplot(data=ood_df, x='Label', y='F1', ax=axes[0,1])
        axes[0,1].set_title('F1 Scores on Out-of-Domain Data')
        axes[0,1].set_xticklabels(axes[0,1].get_xticklabels(), rotation=45)
        axes[0,1].set_ylabel('F1 Score')

        # Precision vs Recall
        sns.scatterplot(data=ood_df, x='Precision', y='Recall', hue='Label', s=100, ax=axes[1,0])
        axes[1,0].set_title('Precision vs Recall on Out-of-Domain Data')
        axes[1,0].set_xlabel('Precision')
        axes[1,0].set_ylabel('Recall')
        axes[1,0].legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Support (sample sizes)
        sns.barplot(data=ood_df, x='Label', y='Support', ax=axes[1,1])
        axes[1,1].set_title('Sample Sizes for Each Label (Out-of-Domain)')
        axes[1,1].set_xticklabels(axes[1,1].get_xticklabels(), rotation=45)
        axes[1,1].set_ylabel('Number of Samples')

        plt.tight_layout()
        plt.savefig(plots_dir / "out_of_domain_performance.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 2. Synthetic predictions analysis
    print("📊 Creating synthetic predictions analysis plot...")

    synthetic_file = extended_dir / "synthetic_predictions.csv"
    if synthetic_file.exists():
        synthetic_df = pd.read_csv(synthetic_file)

        # Analyze prediction distributions
        prob_cols = [col for col in synthetic_df.columns if '_prob' in col]
        pred_cols = [col for col in synthetic_df.columns if '_pred' in col]

        if prob_cols:
            # Probability distributions
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            axes = axes.flatten()

            for idx, label in enumerate(LABEL_COLS):
                prob_col = f"{label}_prob"
                if prob_col in synthetic_df.columns:
                    ax = axes[idx]
                    sns.histplot(data=synthetic_df, x=prob_col, bins=20, ax=ax, alpha=0.7)
                    ax.set_title(f'{label.replace("_", " ").title()} Prediction Distribution')
                    ax.set_xlabel('Predicted Probability')
                    ax.set_ylabel('Count')
                    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Threshold')
                    ax.legend()

            plt.tight_layout()
            plt.savefig(plots_dir / "synthetic_probability_distributions.png", dpi=300, bbox_inches='tight')
            plt.close()

            # Prediction summary
            pred_summary = []
            for label in LABEL_COLS:
                prob_col = f"{label}_prob"
                pred_col = f"{label}_pred"
                true_col = label

                if all(col in synthetic_df.columns for col in [prob_col, pred_col, true_col]):
                    probs = synthetic_df[prob_col].values
                    preds = synthetic_df[pred_col].values
                    true_vals = synthetic_df[true_col].values

                    pred_summary.append({
                        'Label': label.replace('_', ' ').title(),
                        'Mean Probability': np.mean(probs),
                        'Std Probability': np.std(probs),
                        'Positive Predictions': np.sum(preds),
                        'Total Samples': len(preds),
                        'Positive Rate': np.mean(preds)
                    })

            if pred_summary:
                summary_df = pd.DataFrame(pred_summary)

                fig, axes = plt.subplots(1, 2, figsize=(16, 6))

                # Mean probabilities
                sns.barplot(data=summary_df, x='Label', y='Mean Probability', ax=axes[0])
                axes[0].set_title('Mean Predicted Probabilities on Synthetic Data')
                axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)
                axes[0].set_ylabel('Mean Probability')
                axes[0].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Decision Threshold')
                axes[0].legend()

                # Positive prediction rates
                sns.barplot(data=summary_df, x='Label', y='Positive Rate', ax=axes[1])
                axes[1].set_title('Positive Prediction Rates on Synthetic Data')
                axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
                axes[1].set_ylabel('Positive Prediction Rate')

                plt.tight_layout()
                plt.savefig(plots_dir / "synthetic_prediction_summary.png", dpi=300, bbox_inches='tight')
                plt.close()

    # 3. Performance degradation analysis
    print("📊 Creating performance degradation analysis...")

    # Compare in-domain vs out-of-domain performance
    # We need to load the original metrics for comparison

    try:
        with open(Path("reports/analysis/overall_metrics.json"), 'r') as f:
            in_domain_metrics = json.load(f)
    except FileNotFoundError:
        print("⚠️  In-domain metrics not found, skipping degradation analysis")
        in_domain_metrics = {}

    if in_domain_metrics and ood_data:
        degradation_data = []

        for ood_item in ood_data:
            label = ood_item['Label'].lower().replace(' ', '_')
            model_metrics = in_domain_metrics.get('bert', {})  # Using BERT as example

            in_domain_auc = model_metrics.get(f"{label}_auc", 0)
            ood_auc = ood_item['AUC']

            if in_domain_auc > 0:
                degradation = in_domain_auc - ood_auc
                degradation_pct = (degradation / in_domain_auc) * 100

                degradation_data.append({
                    'Label': ood_item['Label'],
                    'In-Domain AUC': in_domain_auc,
                    'Out-of-Domain AUC': ood_auc,
                    'AUC Degradation': degradation,
                    'Degradation %': degradation_pct
                })

        if degradation_data:
            deg_df = pd.DataFrame(degradation_data)

            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # AUC comparison
            deg_melted = deg_df.melt(id_vars=['Label'],
                                   value_vars=['In-Domain AUC', 'Out-of-Domain AUC'],
                                   var_name='Domain', value_name='AUC')
            sns.barplot(data=deg_melted, x='Label', y='AUC', hue='Domain', ax=axes[0])
            axes[0].set_title('AUC Performance: In-Domain vs Out-of-Domain')
            axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)
            axes[0].legend(title='Domain')

            # Degradation percentage
            sns.barplot(data=deg_df, x='Label', y='Degradation %', ax=axes[1])
            axes[1].set_title('Performance Degradation on Out-of-Domain Data')
            axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
            axes[1].set_ylabel('AUC Degradation (%)')
            axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.5)

            plt.tight_layout()
            plt.savefig(plots_dir / "performance_degradation_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()

    print("✅ Extended evaluation plots generated!")
    print(f"📁 Saved to: {plots_dir}")

if __name__ == "__main__":
    create_extended_evaluation_plots()