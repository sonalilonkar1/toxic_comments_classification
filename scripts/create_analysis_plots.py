#!/usr/bin/env python3
"""
Generate comprehensive plots for analysis folder data.
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

def create_analysis_plots():
    """Create plots for the analysis folder."""

    analysis_dir = Path("reports/analysis")
    plots_dir = analysis_dir
    plots_dir.mkdir(exist_ok=True)

    # 1. Best models by AUC ranking
    print("📊 Creating AUC ranking plot...")

    with open(analysis_dir / "analysis_summary.json", 'r') as f:
        analysis_data = json.load(f)

    auc_data = []
    for item in analysis_data["best_models_by_auc"]:
        auc_data.append({
            'Model': item['Model'],
            'Label': item['Label'].replace('_', ' ').title(),
            'AUC': item['AUC']
        })

    if auc_data:
        auc_df = pd.DataFrame(auc_data)

        plt.figure(figsize=(12, 8))
        sns.barplot(data=auc_df, x='AUC', y='Label', hue='Model', orient='h')
        plt.title('Best Performing Models by AUC for Each Toxicity Label')
        plt.xlabel('AUC Score')
        plt.ylabel('Toxicity Label')
        plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(plots_dir / "auc_ranking_plot.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 2. Calibration metrics comparison
    print("📊 Creating calibration comparison plot...")

    with open(analysis_dir / "calibration_metrics.json", 'r') as f:
        calib_data = json.load(f)

    calib_summary = []
    for model, labels in calib_data.items():
        for label, metrics in labels.items():
            calib_summary.append({
                'Model': model.replace('tfidf_', '').replace('_', ' ').title(),
                'Label': label.replace('_', ' ').title(),
                'Brier Score': metrics['brier_score'],
                'ECE': metrics['ece']
            })

    if calib_summary:
        calib_df = pd.DataFrame(calib_summary)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

        # Brier Score
        sns.barplot(data=calib_df, x='Label', y='Brier Score', hue='Model', ax=ax1)
        ax1.set_title('Brier Score by Model and Label\n(Lower is Better)')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
        ax1.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')

        # ECE
        sns.barplot(data=calib_df, x='Label', y='ECE', hue='Model', ax=ax2)
        ax2.set_title('Expected Calibration Error by Model and Label\n(Lower is Better)')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
        ax2.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig(plots_dir / "calibration_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 3. Overall metrics heatmap
    print("📊 Creating overall metrics heatmap...")

    with open(analysis_dir / "overall_metrics.json", 'r') as f:
        overall_data = json.load(f)

    # Extract macro metrics
    macro_data = []
    for model, metrics in overall_data.items():
        if 'macro_pr_auc' in metrics and 'macro_roc_auc' in metrics:
            macro_data.append({
                'Model': model.replace('tfidf_', '').replace('_', ' ').title(),
                'Macro PR-AUC': metrics['macro_pr_auc'],
                'Macro ROC-AUC': metrics['macro_roc_auc']
            })

    if macro_data:
        macro_df = pd.DataFrame(macro_data)
        macro_df = macro_df.set_index('Model')

        plt.figure(figsize=(10, 6))
        sns.heatmap(macro_df, annot=True, cmap='YlGnBu', fmt='.3f', cbar_kws={'label': 'Score'})
        plt.title('Macro-Averaged AUC Scores Across All Models')
        plt.tight_layout()
        plt.savefig(plots_dir / "macro_auc_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 4. Prediction correlations heatmap
    print("📊 Creating prediction correlations heatmap...")

    with open(analysis_dir / "prediction_correlations.json", 'r') as f:
        corr_data = json.load(f)

    # Extract correlations for toxic label (as example)
    toxic_corr = corr_data.get('toxic', {})
    if toxic_corr:
        # Create correlation matrix
        models = []
        corr_matrix = {}

        for key, value in toxic_corr.items():
            if isinstance(value, dict):
                models.append(key)
                corr_matrix[key] = value

        if models:
            # Create DataFrame
            corr_df = pd.DataFrame(corr_matrix)
            corr_df = corr_df[models]  # Ensure square matrix

            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr_df, dtype=bool))  # Upper triangle mask
            sns.heatmap(corr_df, mask=mask, annot=True, cmap='coolwarm', fmt='.3f',
                       center=0, cbar_kws={'label': 'Correlation'})
            plt.title('Prediction Correlations Between Models (Toxic Label)')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(plots_dir / "prediction_correlations_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()

    print("✅ Analysis plots generated!")
    print(f"📁 Saved to: {plots_dir}")

if __name__ == "__main__":
    create_analysis_plots()