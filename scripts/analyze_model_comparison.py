"""Model comparison analysis script."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

def load_test_data():
    """Load test data with true labels."""
    # Instead of loading external test data, we'll use the predictions files
    # which contain both predictions and true labels from cross-validation
    return None

def load_model_predictions(model_name):
    """Load predictions for a specific model."""
    # Find the latest experiment directory
    exp_dir = Path("experiments/train") / model_name
    if not exp_dir.exists():
        print(f"Experiment directory not found: {exp_dir}")
        return None
    
    # For BERT, find the latest fold directory first
    if model_name == "bert":
        fold_dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir() and not d.name.startswith('.')], 
                          key=lambda x: x.name, reverse=True)
        if not fold_dirs:
            print(f"No fold directories found for {model_name}")
            return None
        latest_fold = fold_dirs[0]
        # Find timestamped subdirectories
        timestamp_dirs = sorted([d for d in latest_fold.iterdir() if d.is_dir() and not d.name.startswith('.')], 
                               key=lambda x: x.name, reverse=True)
        if timestamp_dirs:
            pred_file = timestamp_dirs[0] / "test_predictions.csv"
        else:
            return None
    else:
        # For TF-IDF models, find all timestamped directories and get the latest one
        timestamp_dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir() and not d.name.startswith('.')], 
                               key=lambda x: x.name, reverse=True)
        if timestamp_dirs:
            pred_file = timestamp_dirs[0] / "test_predictions.csv"
        else:
            return None
    
    if pred_file.exists():
        return pd.read_csv(pred_file)
    else:
        print(f"Predictions file not found: {pred_file}")
        return None
    """Load predictions for a specific model."""
    # Find the latest experiment directory
    exp_dir = Path("experiments/train") / model_name
    if not exp_dir.exists():
        print(f"Experiment directory not found: {exp_dir}")
        return None
    
    # For BERT, find the latest fold directory first
    if model_name == "bert":
        fold_dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir() and not d.name.startswith('.')], 
                          key=lambda x: x.name, reverse=True)
        if not fold_dirs:
            print(f"No fold directories found for {model_name}")
            return None
        latest_fold = fold_dirs[0]
        # Find timestamped subdirectories
        timestamp_dirs = sorted([d for d in latest_fold.iterdir() if d.is_dir() and not d.name.startswith('.')], 
                               key=lambda x: x.name, reverse=True)
        if timestamp_dirs:
            pred_file = timestamp_dirs[0] / "test_predictions.csv"
        else:
            return None
    else:
        # For TF-IDF models, find all timestamped directories and get the latest one
        timestamp_dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir() and not d.name.startswith('.')], 
                               key=lambda x: x.name, reverse=True)
        if timestamp_dirs:
            pred_file = timestamp_dirs[0] / "test_predictions.csv"
        else:
            return None
    
    if pred_file.exists():
        return pd.read_csv(pred_file)
    else:
        print(f"Predictions file not found: {pred_file}")
        return None

def analyze_model_comparison():
    """Comprehensive model comparison analysis."""
    print("🔍 Loading model predictions...")
    
    # Load predictions for all models
    models = ["tfidf_logistic", "tfidf_svm", "tfidf_random_forest", "bert"]
    predictions = {}
    
    for model in models:
        print(f"📊 Loading predictions for {model}...")
        pred_df = load_model_predictions(model)
        if pred_df is not None:
            predictions[model] = pred_df
            print(f"✅ Loaded {model} predictions: {len(pred_df)} samples")
        else:
            print(f"❌ Failed to load {model} predictions")
    
    if not predictions:
        print("No predictions loaded. Exiting.")
        return
    
    # Create analysis directory
    analysis_dir = Path("reports/analysis")
    analysis_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. Overall Performance Comparison
    print("\n📈 Computing overall performance metrics...")
    overall_metrics = {}
    
    for model_name, pred_df in predictions.items():
        metrics = {}
        
        # Collect per-label probabilities and true labels for macro calculations
        all_probabilities = []
        all_true_labels = []
        
        for label in LABEL_COLS:
            prob_col = f"{label}_prob"
            pred_col = f"{label}_pred"
            true_col = label
            
            if prob_col in pred_df.columns and true_col in pred_df.columns:
                # Use probabilities for AUC
                y_true = pred_df[true_col].values
                y_prob = pred_df[prob_col].values
                auc = roc_auc_score(y_true, y_prob)
                metrics[f"{label}_auc"] = auc
                
                # Collect for macro calculations
                all_probabilities.append(y_prob)
                all_true_labels.append(y_true)
            
            if pred_col in pred_df.columns and true_col in pred_df.columns:
                # Use predictions for precision/recall/f1
                y_true = pred_df[true_col].values
                y_pred = pred_df[pred_col].values
                
                report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                metrics[f"{label}_precision"] = report['1']['precision']
                metrics[f"{label}_recall"] = report['1']['recall']
                metrics[f"{label}_f1"] = report['1']['f1-score']
        
        # Calculate macro-averaged metrics
        if all_probabilities and all_true_labels:
            # Macro PR-AUC
            macro_pr_auc = np.mean([average_precision_score(y_true, y_prob) 
                                   for y_true, y_prob in zip(all_true_labels, all_probabilities)])
            metrics["macro_pr_auc"] = macro_pr_auc
            
            # Macro ROC-AUC  
            macro_roc_auc = np.mean([roc_auc_score(y_true, y_prob) 
                                    for y_true, y_prob in zip(all_true_labels, all_probabilities)])
            metrics["macro_roc_auc"] = macro_roc_auc
        
        overall_metrics[model_name] = metrics
    
    # Save overall metrics
    with open(analysis_dir / "overall_metrics.json", 'w') as f:
        json.dump(overall_metrics, f, indent=2)
    
    # 2. Create comparison plots
    print("📊 Creating comparison plots...")
    
    # AUC Comparison
    auc_data = []
    for model_name, metrics in overall_metrics.items():
        for label in LABEL_COLS:
            auc_key = f"{label}_auc"
            if auc_key in metrics:
                auc_data.append({
                    'Model': model_name.replace('tfidf_', '').replace('_', ' ').title(),
                    'Label': label.replace('_', ' ').title(),
                    'AUC': metrics[auc_key]
                })
    
    if auc_data:
        auc_df = pd.DataFrame(auc_data)
        plt.figure(figsize=(12, 8))
        sns.barplot(data=auc_df, x='Label', y='AUC', hue='Model')
        plt.title('AUC Comparison Across Models and Labels')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(analysis_dir / "auc_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # F1 Score Comparison
    f1_data = []
    for model_name, metrics in overall_metrics.items():
        for label in LABEL_COLS:
            f1_key = f"{label}_f1"
            if f1_key in metrics:
                f1_data.append({
                    'Model': model_name.replace('tfidf_', '').replace('_', ' ').title(),
                    'Label': label.replace('_', ' ').title(),
                    'F1': metrics[f1_key]
                })
    
    if f1_data:
        f1_df = pd.DataFrame(f1_data)
        plt.figure(figsize=(12, 8))
        sns.barplot(data=f1_df, x='Label', y='F1', hue='Model')
        plt.title('F1 Score Comparison Across Models and Labels')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(analysis_dir / "f1_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Prediction Confidence Analysis
    print("🎯 Analyzing prediction confidence...")
    
    confidence_data = []
    for model_name, pred_df in predictions.items():
        for label in LABEL_COLS:
            prob_col = f"{label}_prob"
            pred_col = f"{label}_pred"
            
            if prob_col in pred_df.columns and pred_col in pred_df.columns:
                probs = pred_df[prob_col].values
                preds = pred_df[pred_col].values
                
                # Confidence for positive predictions
                pos_probs = probs[preds == 1]
                if len(pos_probs) > 0:
                    confidence_data.append({
                        'Model': model_name.replace('tfidf_', '').replace('_', ' ').title(),
                        'Label': label.replace('_', ' ').title(),
                        'Type': 'Positive Predictions',
                        'Mean_Confidence': np.mean(pos_probs),
                        'Median_Confidence': np.median(pos_probs)
                    })
                
                # Confidence for negative predictions
                neg_probs = probs[preds == 0]
                if len(neg_probs) > 0:
                    confidence_data.append({
                        'Model': model_name.replace('tfidf_', '').replace('_', ' ').title(),
                        'Label': label.replace('_', ' ').title(),
                        'Type': 'Negative Predictions',
                        'Mean_Confidence': np.mean(1 - neg_probs),
                        'Median_Confidence': np.median(1 - neg_probs)
                    })
    
    if confidence_data:
        conf_df = pd.DataFrame(confidence_data)
        plt.figure(figsize=(15, 10))
        sns.boxplot(data=conf_df, x='Label', y='Mean_Confidence', hue='Model')
        plt.title('Prediction Confidence Distribution by Model and Label')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(analysis_dir / "confidence_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Error Analysis
    print("🔍 Performing error analysis...")
    
    error_analysis = {}
    for model_name, pred_df in predictions.items():
        model_errors = {}
        
        for label in LABEL_COLS:
            pred_col = f"{label}_pred"
            true_col = label
            if pred_col in pred_df.columns and true_col in pred_df.columns:
                y_true = pred_df[true_col].values
                y_pred = pred_df[pred_col].values
                
                # False positives and false negatives
                fp_mask = (y_pred == 1) & (y_true == 0)
                fn_mask = (y_pred == 0) & (y_true == 1)
                
                model_errors[label] = {
                    'false_positives': int(fp_mask.sum()),
                    'false_negatives': int(fn_mask.sum()),
                    'total_positives': int(y_true.sum()),
                    'total_predictions': len(y_true)
                }
        
        error_analysis[model_name] = model_errors
    
    # Save error analysis
    with open(analysis_dir / "error_analysis.json", 'w') as f:
        json.dump(error_analysis, f, indent=2)
    
    # 5. Correlation Analysis
    print("📊 Analyzing prediction correlations...")
    
    # Create a dataframe with all predictions
    correlation_data = predictions[list(predictions.keys())[0]][['comment_text']].copy()
    
    for model_name, pred_df in predictions.items():
        for label in LABEL_COLS:
            prob_col = f"{label}_prob"
            if prob_col in pred_df.columns:
                correlation_data[f"{model_name}_{label}_prob"] = pred_df[prob_col]
    
    # Compute correlations between model predictions for each label
    correlations = {}
    for label in LABEL_COLS:
        label_cols = [col for col in correlation_data.columns if col.endswith(f"{label}_prob")]
        if len(label_cols) > 1:
            corr_matrix = correlation_data[label_cols].corr()
            correlations[label] = corr_matrix.to_dict()
    
    with open(analysis_dir / "prediction_correlations.json", 'w') as f:
        json.dump(correlations, f, indent=2)
    
    # 6. Generate summary report
    print("📝 Generating summary report...")
    
    # Get sample size from first predictions dataframe
    sample_size = len(list(predictions.values())[0]) if predictions else 0
    
    summary = {
        "analysis_timestamp": pd.Timestamp.now().isoformat(),
        "validation_data_size": sample_size,
        "models_analyzed": list(predictions.keys()),
        "labels_analyzed": LABEL_COLS,
        "data_source": "cross_validation_predictions",
        "key_findings": []
    }
    
    # Find best performing model per label
    if auc_data:
        auc_df = pd.DataFrame(auc_data)
        best_models = auc_df.loc[auc_df.groupby('Label')['AUC'].idxmax()]
        summary["best_models_by_auc"] = best_models.to_dict('records')
    
    # Overall statistics
    total_predictions = sum(len(pred_df) for pred_df in predictions.values())
    summary["total_predictions_analyzed"] = total_predictions
    
    with open(analysis_dir / "analysis_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("✅ Model comparison analysis complete!")
    print(f"📁 Results saved to: {analysis_dir}")
    print("\n📊 Generated files:")
    print("- overall_metrics.json: Detailed performance metrics")
    print("- auc_comparison.png: AUC comparison plot")
    print("- f1_comparison.png: F1 score comparison plot")
    print("- confidence_analysis.png: Prediction confidence analysis")
    print("- error_analysis.json: False positive/negative analysis")
    print("- prediction_correlations.json: Model prediction correlations")
    print("- analysis_summary.json: Summary report")

if __name__ == "__main__":
    analyze_model_comparison()