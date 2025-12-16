"""Error analysis script for toxic comment classification."""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from collections import Counter
import re
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

def load_predictions(model_name: str) -> pd.DataFrame:
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

def analyze_errors():
    """Comprehensive error analysis."""
    print("🔍 Starting error analysis...")
    
    # Load BERT predictions (best performing model)
    bert_df = load_predictions("bert")
    if bert_df is None:
        print("❌ Could not load BERT predictions")
        return
    
    print(f"📊 Analyzing {len(bert_df)} predictions from BERT model")
    
    # Create analysis directory
    analysis_dir = Path("reports/error_analysis")
    analysis_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. Overall error statistics
    print("📈 Computing overall error statistics...")
    error_stats = {}
    
    for label in LABEL_COLS:
        pred_col = f"{label}_pred"
        prob_col = f"{label}_prob"
        true_col = label
        
        if all(col in bert_df.columns for col in [pred_col, prob_col, true_col]):
            y_true = bert_df[true_col].values
            y_pred = bert_df[pred_col].values
            y_prob = bert_df[prob_col].values
            
            # Error types
            fp_mask = (y_pred == 1) & (y_true == 0)  # False positives
            fn_mask = (y_pred == 0) & (y_true == 1)  # False negatives
            tp_mask = (y_pred == 1) & (y_true == 1)  # True positives
            tn_mask = (y_pred == 0) & (y_true == 0)  # True negatives
            
            error_stats[label] = {
                "total_samples": len(y_true),
                "positive_samples": int(y_true.sum()),
                "negative_samples": int((1 - y_true).sum()),
                "true_positives": int(tp_mask.sum()),
                "true_negatives": int(tn_mask.sum()),
                "false_positives": int(fp_mask.sum()),
                "false_negatives": int(fn_mask.sum()),
                "precision": int(tp_mask.sum()) / (int(tp_mask.sum()) + int(fp_mask.sum())) if (tp_mask.sum() + fp_mask.sum()) > 0 else 0,
                "recall": int(tp_mask.sum()) / (int(tp_mask.sum()) + int(fn_mask.sum())) if (tp_mask.sum() + fn_mask.sum()) > 0 else 0,
                "f1_score": 2 * (int(tp_mask.sum()) / (int(tp_mask.sum()) + int(fp_mask.sum())) if (tp_mask.sum() + fp_mask.sum()) > 0 else 0) * (int(tp_mask.sum()) / (int(tp_mask.sum()) + int(fn_mask.sum())) if (tp_mask.sum() + fn_mask.sum()) > 0 else 0) / ((int(tp_mask.sum()) / (int(tp_mask.sum()) + int(fp_mask.sum())) if (tp_mask.sum() + fp_mask.sum()) > 0 else 0) + (int(tp_mask.sum()) / (int(tp_mask.sum()) + int(fn_mask.sum())) if (tp_mask.sum() + fn_mask.sum()) > 0 else 0)) if ((int(tp_mask.sum()) / (int(tp_mask.sum()) + int(fp_mask.sum())) if (tp_mask.sum() + fp_mask.sum()) > 0 else 0) + (int(tp_mask.sum()) / (int(tp_mask.sum()) + int(fn_mask.sum())) if (tp_mask.sum() + fn_mask.sum()) > 0 else 0)) > 0 else 0
            }
    
    # Save error statistics
    with open(analysis_dir / "error_statistics.json", 'w') as f:
        json.dump(error_stats, f, indent=2)
    
    # 2. False positive analysis
    print("🔍 Analyzing false positives...")
    fp_analysis = {}
    
    for label in LABEL_COLS:
        pred_col = f"{label}_pred"
        prob_col = f"{label}_prob"
        true_col = label
        
        if all(col in bert_df.columns for col in [pred_col, prob_col, true_col]):
            fp_mask = (bert_df[pred_col] == 1) & (bert_df[true_col] == 0)
            fp_comments = bert_df[fp_mask]['comment_text'].tolist()
            fp_probs = bert_df[fp_mask][prob_col].tolist()
            
            # Analyze common patterns in false positives
            fp_words = []
            for comment in fp_comments:
                words = re.findall(r'\b\w+\b', str(comment).lower())
                fp_words.extend(words)
            
            word_freq = Counter(fp_words).most_common(20)
            
            fp_analysis[label] = {
                "count": len(fp_comments),
                "avg_confidence": float(np.mean(fp_probs)) if fp_probs else 0,
                "max_confidence": float(np.max(fp_probs)) if fp_probs else 0,
                "common_words": dict(word_freq),
                "sample_comments": fp_comments[:5]  # First 5 examples
            }
    
    with open(analysis_dir / "false_positives.json", 'w') as f:
        json.dump(fp_analysis, f, indent=2)
    
    # 3. False negative analysis
    print("🔍 Analyzing false negatives...")
    fn_analysis = {}
    
    for label in LABEL_COLS:
        pred_col = f"{label}_pred"
        prob_col = f"{label}_prob"
        true_col = label
        
        if all(col in bert_df.columns for col in [pred_col, prob_col, true_col]):
            fn_mask = (bert_df[pred_col] == 0) & (bert_df[true_col] == 1)
            fn_comments = bert_df[fn_mask]['comment_text'].tolist()
            fn_probs = bert_df[fn_mask][prob_col].tolist()
            
            # Analyze common patterns in false negatives
            fn_words = []
            for comment in fn_comments:
                words = re.findall(r'\b\w+\b', str(comment).lower())
                fn_words.extend(words)
            
            word_freq = Counter(fn_words).most_common(20)
            
            fn_analysis[label] = {
                "count": len(fn_comments),
                "avg_confidence": float(np.mean(fn_probs)) if fn_probs else 0,
                "max_confidence": float(np.max(fn_probs)) if fn_probs else 0,
                "common_words": dict(word_freq),
                "sample_comments": fn_comments[:5]  # First 5 examples
            }
    
    with open(analysis_dir / "false_negatives.json", 'w') as f:
        json.dump(fn_analysis, f, indent=2)
    
    # 4. Confidence distribution analysis
    print("📊 Analyzing confidence distributions...")
    confidence_analysis = {}
    
    for label in LABEL_COLS:
        prob_col = f"{label}_prob"
        true_col = label
        
        if all(col in bert_df.columns for col in [prob_col, true_col]):
            probs = bert_df[prob_col].values
            true_labels = bert_df[true_col].values
            
            # Confidence for different groups
            pos_probs = probs[true_labels == 1]
            neg_probs = probs[true_labels == 0]
            
            confidence_analysis[label] = {
                "positive_class": {
                    "count": len(pos_probs),
                    "mean": float(np.mean(pos_probs)),
                    "median": float(np.median(pos_probs)),
                    "std": float(np.std(pos_probs)),
                    "min": float(np.min(pos_probs)),
                    "max": float(np.max(pos_probs))
                },
                "negative_class": {
                    "count": len(neg_probs),
                    "mean": float(np.mean(neg_probs)),
                    "median": float(np.median(neg_probs)),
                    "std": float(np.std(neg_probs)),
                    "min": float(np.min(neg_probs)),
                    "max": float(np.max(neg_probs))
                }
            }
    
    with open(analysis_dir / "confidence_distribution.json", 'w') as f:
        json.dump(confidence_analysis, f, indent=2)
    
    # 5. Multi-label analysis
    print("🏷️ Analyzing multi-label patterns...")
    multilabel_analysis = {}
    
    # Count comments with multiple labels
    true_label_sums = bert_df[LABEL_COLS].sum(axis=1)
    pred_label_sums = bert_df[[f"{label}_pred" for label in LABEL_COLS]].sum(axis=1)
    
    multilabel_analysis["true_labels"] = {
        "single_label": int((true_label_sums == 1).sum()),
        "multi_label": int((true_label_sums > 1).sum()),
        "no_labels": int((true_label_sums == 0).sum())
    }
    
    multilabel_analysis["predicted_labels"] = {
        "single_label": int((pred_label_sums == 1).sum()),
        "multi_label": int((pred_label_sums > 1).sum()),
        "no_labels": int((pred_label_sums == 0).sum())
    }
    
    # Label co-occurrence
    label_cooccurrence = {}
    for i, label1 in enumerate(LABEL_COLS):
        for label2 in LABEL_COLS[i+1:]:
            cooccur = int(((bert_df[label1] == 1) & (bert_df[label2] == 1)).sum())
            label_cooccurrence[f"{label1}_{label2}"] = cooccur
    
    multilabel_analysis["cooccurrence"] = label_cooccurrence
    
    with open(analysis_dir / "multilabel_analysis.json", 'w') as f:
        json.dump(multilabel_analysis, f, indent=2)
    
    # 6. Generate insights report
    print("💡 Generating insights report...")
    insights = {
        "analysis_timestamp": pd.Timestamp.now().isoformat(),
        "model": "bert",
        "total_samples": len(bert_df),
        "key_insights": []
    }
    
    # Most problematic labels (highest error rates)
    error_rates = {}
    for label, stats in error_stats.items():
        total_errors = stats["false_positives"] + stats["false_negatives"]
        error_rate = total_errors / stats["total_samples"]
        error_rates[label] = error_rate
    
    worst_label = max(error_rates, key=error_rates.get)
    insights["key_insights"].append(f"Worst performing label: {worst_label} with {error_rates[worst_label]:.3f} error rate")
    
    # Most confident false positives
    high_conf_fp = {}
    for label, analysis in fp_analysis.items():
        if analysis["count"] > 0 and analysis["max_confidence"] > 0.8:
            high_conf_fp[label] = analysis["max_confidence"]
    
    if high_conf_fp:
        most_conf_fp_label = max(high_conf_fp, key=high_conf_fp.get)
        insights["key_insights"].append(f"Highest confidence false positive in {most_conf_fp_label}: {high_conf_fp[most_conf_fp_label]:.3f}")
    
    # Multi-label insights
    multi_label_pct = multilabel_analysis["true_labels"]["multi_label"] / sum(multilabel_analysis["true_labels"].values())
    insights["key_insights"].append(f"{multi_label_pct:.1%} of comments have multiple toxicity labels")
    
    with open(analysis_dir / "error_insights.json", 'w') as f:
        json.dump(insights, f, indent=2)
    
    print("✅ Error analysis complete!")
    print(f"📁 Results saved to: {analysis_dir}")
    print("\n📊 Generated files:")
    print("- error_statistics.json: Overall error statistics")
    print("- false_positives.json: False positive analysis")
    print("- false_negatives.json: False negative analysis")
    print("- confidence_distribution.json: Confidence distribution analysis")
    print("- multilabel_analysis.json: Multi-label pattern analysis")
    print("- error_insights.json: Key insights and recommendations")

if __name__ == "__main__":
    analyze_errors()