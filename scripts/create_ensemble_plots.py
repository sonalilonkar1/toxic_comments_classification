#!/usr/bin/env python3
"""
Generate comprehensive plots for ensemble analysis.
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

def create_ensemble_plots():
    """Create plots for the ensemble folder."""

    ensemble_dir = Path("reports/ensemble")
    plots_dir = ensemble_dir
    plots_dir.mkdir(exist_ok=True)

    # 1. Ensemble vs Individual model comparison
    print("📊 Creating ensemble comparison plot...")

    with open(ensemble_dir / "ensemble_comparison.json", 'r') as f:
        ensemble_data = json.load(f)

    # Extract macro F1 scores
    comparison_data = []

    # Individual models
    for model_name, labels in ensemble_data.get("individual_models", {}).items():
        macro_f1 = np.mean([label_data.get("f1", 0) for label_data in labels.values()])
        comparison_data.append({
            'Method': model_name.replace('tfidf_', '').replace('_', ' ').title(),
            'Type': 'Individual',
            'Macro F1': macro_f1
        })

    # Ensemble methods
    for method_name, labels in ensemble_data.get("ensemble_methods", {}).items():
        if isinstance(labels, dict):
            macro_f1 = np.mean([label_data.get("f1", 0) for label_data in labels.values()])
            method_display = method_name.replace('_', ' ').title()
            comparison_data.append({
                'Method': method_display,
                'Type': 'Ensemble',
                'Macro F1': macro_f1
            })

    if comparison_data:
        comp_df = pd.DataFrame(comparison_data)

        plt.figure(figsize=(14, 8))
        sns.barplot(data=comp_df, x='Method', y='Macro F1', hue='Type')
        plt.title('Individual Models vs Ensemble Methods: Macro F1 Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Macro-averaged F1 Score')
        plt.legend(title='Method Type')
        plt.tight_layout()
        plt.savefig(plots_dir / "ensemble_vs_individual_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 2. Ensemble method performance by label
    print("📊 Creating per-label ensemble performance plot...")

    label_performance = []
    ensemble_methods = ensemble_data.get("ensemble_methods", {})

    for method_name, labels in ensemble_methods.items():
        if isinstance(labels, dict):
            for label, metrics in labels.items():
                label_performance.append({
                    'Ensemble Method': method_name.replace('_', ' ').title(),
                    'Label': label.replace('_', ' ').title(),
                    'F1 Score': metrics.get('f1', 0),
                    'Precision': metrics.get('precision', 0),
                    'Recall': metrics.get('recall', 0)
                })

    if label_performance:
        perf_df = pd.DataFrame(label_performance)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # F1 Score
        sns.barplot(data=perf_df, x='Label', y='F1 Score', hue='Ensemble Method', ax=axes[0])
        axes[0].set_title('F1 Score by Ensemble Method and Label')
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)
        axes[0].legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Precision
        sns.barplot(data=perf_df, x='Label', y='Precision', hue='Ensemble Method', ax=axes[1])
        axes[1].set_title('Precision by Ensemble Method and Label')
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
        axes[1].legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Recall
        sns.barplot(data=perf_df, x='Label', y='Recall', hue='Ensemble Method', ax=axes[2])
        axes[2].set_title('Recall by Ensemble Method and Label')
        axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45)
        axes[2].legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig(plots_dir / "ensemble_per_label_performance.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 3. Ensemble summary statistics
    print("📊 Creating ensemble summary plot...")

    with open(ensemble_dir / "ensemble_summary.json", 'r') as f:
        summary_data = json.load(f)

    summary_stats = []
    for method, stats in summary_data.items():
        if isinstance(stats, dict):
            summary_stats.append({
                'Method': method.replace('_', ' ').title(),
                'Mean F1': stats.get('mean_f1', 0),
                'Std F1': stats.get('std_f1', 0),
                'Improvement': stats.get('improvement_over_best', 0)
            })

    if summary_stats:
        summary_df = pd.DataFrame(summary_stats)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Mean F1 with error bars
        ax1.bar(summary_df['Method'], summary_df['Mean F1'], yerr=summary_df['Std F1'], capsize=5)
        ax1.set_title('Ensemble Methods: Mean F1 Score with Standard Deviation')
        ax1.set_ylabel('Macro F1 Score')
        ax1.set_xticklabels(summary_df['Method'], rotation=45, ha='right')

        # Improvement over best individual
        ax2.bar(summary_df['Method'], summary_df['Improvement'])
        ax2.set_title('Improvement Over Best Individual Model')
        ax2.set_ylabel('F1 Score Improvement')
        ax2.set_xticklabels(summary_df['Method'], rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(plots_dir / "ensemble_summary_statistics.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 4. Prediction diversity analysis (if CSV files exist)
    print("📊 Creating prediction diversity plot...")

    csv_files = [
        "probability_mean_predictions.csv",
        "probability_median_predictions.csv",
        "voting_majority_predictions.csv",
        "voting_unanimous_predictions.csv",
        "weighted_ensemble_predictions.csv"
    ]

    diversity_data = []
    for csv_file in csv_files:
        csv_path = ensemble_dir / csv_file
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            method_name = csv_file.replace('_predictions.csv', '').replace('_', ' ').title()

            # Calculate prediction diversity (variance in predictions)
            prob_cols = [col for col in df.columns if '_prob' in col]
            if prob_cols:
                pred_variance = df[prob_cols].var().mean()
                diversity_data.append({
                    'Method': method_name,
                    'Prediction Variance': pred_variance
                })

    if diversity_data:
        diversity_df = pd.DataFrame(diversity_data)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=diversity_df, x='Method', y='Prediction Variance')
        plt.title('Prediction Diversity Across Ensemble Methods\n(Higher variance = more diverse predictions)')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Average Prediction Variance')
        plt.tight_layout()
        plt.savefig(plots_dir / "ensemble_prediction_diversity.png", dpi=300, bbox_inches='tight')
        plt.close()

    print("✅ Ensemble plots generated!")
    print(f"📁 Saved to: {plots_dir}")

if __name__ == "__main__":
    create_ensemble_plots()