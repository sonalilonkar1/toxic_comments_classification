#!/usr/bin/env python3
"""
Generate comprehensive plots for error analysis.
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

def create_error_analysis_plots():
    """Create plots for the error analysis folder."""

    error_dir = Path("reports/error_analysis")
    plots_dir = error_dir
    plots_dir.mkdir(exist_ok=True)

    # 1. Error distribution by label
    print("📊 Creating error distribution plot...")

    with open(error_dir / "error_statistics.json", 'r') as f:
        error_stats = json.load(f)

    error_data = []
    for label, stats in error_stats.items():
        total_errors = stats["false_positives"] + stats["false_negatives"]
        error_rate = total_errors / stats["total_samples"]

        error_data.append({
            'Label': label.replace('_', ' ').title(),
            'False Positives': stats["false_positives"],
            'False Negatives': stats["false_negatives"],
            'Total Errors': total_errors,
            'Error Rate': error_rate,
            'Precision': stats["precision"],
            'Recall': stats["recall"],
            'F1 Score': stats["f1_score"]
        })

    if error_data:
        error_df = pd.DataFrame(error_data)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Error counts
        error_melted = error_df.melt(id_vars=['Label'],
                                   value_vars=['False Positives', 'False Negatives'],
                                   var_name='Error Type', value_name='Count')
        sns.barplot(data=error_melted, x='Label', y='Count', hue='Error Type', ax=axes[0,0])
        axes[0,0].set_title('Error Distribution by Type and Label')
        axes[0,0].set_xticklabels(axes[0,0].get_xticklabels(), rotation=45)
        axes[0,0].legend(title='Error Type')

        # Error rates
        sns.barplot(data=error_df, x='Label', y='Error Rate', ax=axes[0,1])
        axes[0,1].set_title('Error Rate by Label')
        axes[0,1].set_xticklabels(axes[0,1].get_xticklabels(), rotation=45)

        # Precision vs Recall
        sns.scatterplot(data=error_df, x='Precision', y='Recall', hue='Label', s=100, ax=axes[1,0])
        axes[1,0].set_title('Precision vs Recall Trade-off by Label')
        axes[1,0].set_xlabel('Precision')
        axes[1,0].set_ylabel('Recall')
        axes[1,0].legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left')

        # F1 Scores
        sns.barplot(data=error_df, x='Label', y='F1 Score', ax=axes[1,1])
        axes[1,1].set_title('F1 Score by Label')
        axes[1,1].set_xticklabels(axes[1,1].get_xticklabels(), rotation=45)

        plt.tight_layout()
        plt.savefig(plots_dir / "error_distribution_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 2. Confidence distribution analysis
    print("📊 Creating confidence distribution plot...")

    with open(error_dir / "confidence_distribution.json", 'r') as f:
        confidence_data = json.load(f)

    conf_data = []
    for label, distributions in confidence_data.items():
        # Positive class
        pos = distributions["positive_class"]
        conf_data.append({
            'Label': label.replace('_', ' ').title(),
            'Class': 'Positive (Toxic)',
            'Mean': pos["mean"],
            'Median': pos["median"],
            'Std': pos["std"],
            'Count': pos["count"]
        })

        # Negative class
        neg = distributions["negative_class"]
        conf_data.append({
            'Label': label.replace('_', ' ').title(),
            'Class': 'Negative (Benign)',
            'Mean': neg["mean"],
            'Median': neg["median"],
            'Std': neg["std"],
            'Count': neg["count"]
        })

    if conf_data:
        conf_df = pd.DataFrame(conf_data)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Mean confidence by class
        sns.barplot(data=conf_df, x='Label', y='Mean', hue='Class', ax=axes[0])
        axes[0].set_title('Mean Prediction Confidence by Label and True Class')
        axes[0].set_ylabel('Mean Confidence')
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)
        axes[0].legend(title='True Class')

        # Confidence variability (std)
        sns.barplot(data=conf_df, x='Label', y='Std', hue='Class', ax=axes[1])
        axes[1].set_title('Confidence Variability by Label and True Class')
        axes[1].set_ylabel('Confidence Standard Deviation')
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
        axes[1].legend(title='True Class')

        plt.tight_layout()
        plt.savefig(plots_dir / "confidence_distribution_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 3. False positives analysis
    print("📊 Creating false positives analysis plot...")

    with open(error_dir / "false_positives.json", 'r') as f:
        fp_data = json.load(f)

    fp_summary = []
    for label, analysis in fp_data.items():
        fp_summary.append({
            'Label': label.replace('_', ' ').title(),
            'False Positives': analysis["count"],
            'Avg Confidence': analysis["avg_confidence"],
            'Max Confidence': analysis["max_confidence"],
            'Total Samples': error_stats[label]["total_samples"]
        })

    if fp_summary:
        fp_df = pd.DataFrame(fp_summary)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # False positive counts
        sns.barplot(data=fp_df, x='Label', y='False Positives', ax=axes[0])
        axes[0].set_title('False Positive Counts by Label')
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)

        # Average confidence of false positives
        sns.barplot(data=fp_df, x='Label', y='Avg Confidence', ax=axes[1])
        axes[1].set_title('Average Confidence of False Positives')
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
        axes[1].set_ylabel('Average Confidence')

        # Max confidence of false positives
        sns.barplot(data=fp_df, x='Label', y='Max Confidence', ax=axes[2])
        axes[2].set_title('Maximum Confidence of False Positives')
        axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45)
        axes[2].set_ylabel('Maximum Confidence')

        plt.tight_layout()
        plt.savefig(plots_dir / "false_positives_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 4. False negatives analysis
    print("📊 Creating false negatives analysis plot...")

    with open(error_dir / "false_negatives.json", 'r') as f:
        fn_data = json.load(f)

    fn_summary = []
    for label, analysis in fn_data.items():
        fn_summary.append({
            'Label': label.replace('_', ' ').title(),
            'False Negatives': analysis["count"],
            'Avg Confidence': analysis["avg_confidence"],
            'Max Confidence': analysis["max_confidence"],
            'Total Samples': error_stats[label]["total_samples"]
        })

    if fn_summary:
        fn_df = pd.DataFrame(fn_summary)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # False negative counts
        sns.barplot(data=fn_df, x='Label', y='False Negatives', ax=axes[0])
        axes[0].set_title('False Negative Counts by Label')
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)

        # Average confidence of false negatives
        sns.barplot(data=fn_df, x='Label', y='Avg Confidence', ax=axes[1])
        axes[1].set_title('Average Confidence of False Negatives')
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
        axes[1].set_ylabel('Average Confidence')

        # Max confidence of false negatives
        sns.barplot(data=fn_df, x='Label', y='Max Confidence', ax=axes[2])
        axes[2].set_title('Maximum Confidence of False Negatives')
        axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45)
        axes[2].set_ylabel('Maximum Confidence')

        plt.tight_layout()
        plt.savefig(plots_dir / "false_negatives_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 5. Multi-label analysis
    print("📊 Creating multi-label analysis plot...")

    with open(error_dir / "multilabel_analysis.json", 'r') as f:
        multilabel_data = json.load(f)

    # Label distribution
    label_dist = {
        'Single Label': [multilabel_data["true_labels"]["single_label"],
                        multilabel_data["predicted_labels"]["single_label"]],
        'Multi Label': [multilabel_data["true_labels"]["multi_label"],
                       multilabel_data["predicted_labels"]["multi_label"]],
        'No Labels': [multilabel_data["true_labels"]["no_labels"],
                     multilabel_data["predicted_labels"]["no_labels"]]
    }

    dist_df = pd.DataFrame(label_dist, index=['True Labels', 'Predicted Labels']).T

    plt.figure(figsize=(10, 6))
    dist_df.plot(kind='bar', width=0.8)
    plt.title('Multi-label Distribution: True vs Predicted')
    plt.ylabel('Number of Comments')
    plt.xticks(rotation=0)
    plt.legend(title='Type')
    plt.tight_layout()
    plt.savefig(plots_dir / "multilabel_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Label co-occurrence heatmap
    cooccurrence = multilabel_data["cooccurrence"]
    if cooccurrence:
        labels = LABEL_COLS
        cooccur_matrix = np.zeros((len(labels), len(labels)))

        for i, label1 in enumerate(labels):
            for j, label2 in enumerate(labels):
                if i < j:  # Upper triangle only
                    key = f"{label1}_{label2}"
                    cooccur_matrix[i, j] = cooccurrence.get(key, 0)
                    cooccur_matrix[j, i] = cooccur_matrix[i, j]  # Symmetric

        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(cooccur_matrix, dtype=bool))
        sns.heatmap(cooccur_matrix, mask=mask, annot=True, fmt='.0f',
                   xticklabels=[l.replace('_', ' ').title() for l in labels],
                   yticklabels=[l.replace('_', ' ').title() for l in labels],
                   cmap='Blues')
        plt.title('Label Co-occurrence Matrix')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(plots_dir / "label_cooccurrence_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()

    print("✅ Error analysis plots generated!")
    print(f"📁 Saved to: {plots_dir}")

if __name__ == "__main__":
    create_error_analysis_plots()