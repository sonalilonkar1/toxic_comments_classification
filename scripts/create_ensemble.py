"""Ensemble methods for toxic comment classification."""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, roc_auc_score
import json
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

def load_all_predictions() -> Dict[str, pd.DataFrame]:
    """Load predictions from all models."""
    models = ["tfidf_logistic", "tfidf_svm", "tfidf_random_forest", "bert"]
    predictions = {}
    
    for model_name in models:
        exp_dir = Path("experiments/train") / model_name
        if not exp_dir.exists():
            continue
        
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
                    continue
        else:
            timestamp_dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir() and not d.name.startswith('.')], 
                                   key=lambda x: x.name, reverse=True)
            if timestamp_dirs:
                pred_file = timestamp_dirs[0] / "test_predictions.csv"
            else:
                continue
        
        if pred_file.exists():
            predictions[model_name] = pd.read_csv(pred_file)
            print(f"✅ Loaded {model_name} predictions: {len(predictions[model_name])} samples")
        else:
            print(f"❌ Failed to load {model_name} predictions")
    
    return predictions

def create_probability_ensemble(predictions: Dict[str, pd.DataFrame], method: str = "mean") -> pd.DataFrame:
    """Create ensemble predictions using probability averaging."""
    # Use BERT predictions as base (they have the true labels)
    base_df = predictions["bert"].copy()
    
    ensemble_probs = {}
    
    for label in LABEL_COLS:
        prob_cols = []
        
        # Collect probability columns from all models
        for model_name, pred_df in predictions.items():
            prob_col = f"{label}_prob"
            if prob_col in pred_df.columns:
                prob_cols.append(pred_df[prob_col].values)
        
        if prob_cols:
            if method == "mean":
                ensemble_prob = np.mean(prob_cols, axis=0)
            elif method == "median":
                ensemble_prob = np.median(prob_cols, axis=0)
            elif method == "max":
                ensemble_prob = np.max(prob_cols, axis=0)
            else:
                ensemble_prob = np.mean(prob_cols, axis=0)
            
            ensemble_probs[f"{label}_prob"] = ensemble_prob
            ensemble_probs[f"{label}_pred"] = (ensemble_prob > 0.5).astype(int)
    
    # Update base dataframe with ensemble predictions
    for col, values in ensemble_probs.items():
        base_df[col] = values
    
    return base_df

def create_voting_ensemble(predictions: Dict[str, pd.DataFrame], method: str = "majority") -> pd.DataFrame:
    """Create ensemble predictions using voting."""
    base_df = predictions["bert"].copy()
    
    for label in LABEL_COLS:
        pred_cols = []
        
        # Collect prediction columns from all models
        for model_name, pred_df in predictions.items():
            pred_col = f"{label}_pred"
            if pred_col in pred_df.columns:
                pred_cols.append(pred_df[pred_col].values)
        
        if pred_cols:
            pred_array = np.array(pred_cols)
            
            if method == "majority":
                # Majority voting
                ensemble_pred = (np.sum(pred_array, axis=0) > len(pred_cols) / 2).astype(int)
            elif method == "unanimous":
                # All models must agree
                ensemble_pred = np.all(pred_array, axis=0).astype(int)
            elif method == "any":
                # Any model can trigger positive
                ensemble_pred = np.any(pred_array, axis=0).astype(int)
            
            base_df[f"{label}_pred"] = ensemble_pred
            # For probability, use the average of individual model probabilities
            prob_cols = [predictions[model][f"{label}_prob"].values for model in predictions.keys() 
                        if f"{label}_prob" in predictions[model].columns]
            if prob_cols:
                base_df[f"{label}_prob"] = np.mean(prob_cols, axis=0)
    
    return base_df

def create_weighted_ensemble(predictions: Dict[str, pd.DataFrame], weights: Dict[str, float] = None) -> pd.DataFrame:
    """Create weighted ensemble predictions."""
    base_df = predictions["bert"].copy()
    
    # Default weights (BERT gets higher weight since it's best performing)
    if weights is None:
        weights = {
            "tfidf_logistic": 0.15,
            "tfidf_svm": 0.15,
            "tfidf_random_forest": 0.15,
            "bert": 0.55
        }
    
    for label in LABEL_COLS:
        weighted_probs = []
        total_weight = 0
        
        for model_name, pred_df in predictions.items():
            prob_col = f"{label}_prob"
            if prob_col in pred_df.columns and model_name in weights:
                weighted_prob = pred_df[prob_col].values * weights[model_name]
                weighted_probs.append(weighted_prob)
                total_weight += weights[model_name]
        
        if weighted_probs and total_weight > 0:
            ensemble_prob = np.sum(weighted_probs, axis=0) / total_weight
            base_df[f"{label}_prob"] = ensemble_prob
            base_df[f"{label}_pred"] = (ensemble_prob > 0.5).astype(int)
    
    return base_df

def evaluate_ensemble(ensemble_df: pd.DataFrame, ensemble_name: str) -> Dict:
    """Evaluate ensemble performance."""
    results = {}
    
    for label in LABEL_COLS:
        prob_col = f"{label}_prob"
        pred_col = f"{label}_pred"
        true_col = label
        
        if all(col in ensemble_df.columns for col in [prob_col, pred_col, true_col]):
            y_true = ensemble_df[true_col].values
            y_pred = ensemble_df[pred_col].values
            y_prob = ensemble_df[prob_col].values
            
            # Calculate metrics
            auc = roc_auc_score(y_true, y_prob)
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            
            results[label] = {
                "auc": auc,
                "precision": report["1"]["precision"],
                "recall": report["1"]["recall"],
                "f1": report["1"]["f1-score"]
            }
    
    return results

def create_ensemble_comparison():
    """Create comprehensive ensemble comparison."""
    print("🔄 Starting ensemble analysis...")
    
    # Load all predictions
    predictions = load_all_predictions()
    if not predictions:
        print("❌ No predictions loaded")
        return
    
    # Create analysis directory
    analysis_dir = Path("reports/ensemble")
    analysis_dir.mkdir(exist_ok=True, parents=True)
    
    # Define ensemble methods
    ensemble_methods = {
        "probability_mean": lambda: create_probability_ensemble(predictions, "mean"),
        "probability_median": lambda: create_probability_ensemble(predictions, "median"),
        "voting_majority": lambda: create_voting_ensemble(predictions, "majority"),
        "voting_unanimous": lambda: create_voting_ensemble(predictions, "unanimous"),
        "weighted_ensemble": lambda: create_weighted_ensemble(predictions)
    }
    
    # Evaluate all ensembles
    ensemble_results = {}
    
    for method_name, create_func in ensemble_methods.items():
        print(f"🏗️ Creating {method_name} ensemble...")
        try:
            ensemble_df = create_func()
            results = evaluate_ensemble(ensemble_df, method_name)
            ensemble_results[method_name] = results
            
            # Save ensemble predictions
            ensemble_df.to_csv(analysis_dir / f"{method_name}_predictions.csv", index=False)
            
            print(f"✅ {method_name} completed")
        except Exception as e:
            print(f"❌ {method_name} failed: {e}")
    
    # Compare with individual models
    print("📊 Comparing with individual models...")
    individual_results = {}
    
    for model_name, pred_df in predictions.items():
        results = evaluate_ensemble(pred_df, model_name)
        individual_results[model_name] = results
    
    # Create comparison summary
    comparison = {
        "analysis_timestamp": pd.Timestamp.now().isoformat(),
        "individual_models": individual_results,
        "ensemble_methods": ensemble_results,
        "best_performing": {}
    }
    
    # Find best method for each label
    for label in LABEL_COLS:
        best_auc = 0
        best_method = None
        
        # Check individual models
        for model_name, results in individual_results.items():
            if label in results and results[label]["auc"] > best_auc:
                best_auc = results[label]["auc"]
                best_method = f"individual_{model_name}"
        
        # Check ensemble methods
        for method_name, results in ensemble_results.items():
            if label in results and results[label]["auc"] > best_auc:
                best_auc = results[label]["auc"]
                best_method = f"ensemble_{method_name}"
        
        comparison["best_performing"][label] = {
            "method": best_method,
            "auc": best_auc
        }
    
    # Save comparison
    with open(analysis_dir / "ensemble_comparison.json", 'w') as f:
        json.dump(comparison, f, indent=2)
    
    # Create summary report
    summary = {
        "total_models": len(predictions),
        "total_ensembles": len(ensemble_results),
        "improvement_stats": {}
    }
    
    # Calculate average improvements
    for method_name, results in ensemble_results.items():
        improvements = []
        for label in LABEL_COLS:
            if label in results:
                ensemble_auc = results[label]["auc"]
                # Compare with best individual model
                best_individual_auc = max(individual_results[model][label]["auc"] 
                                        for model in individual_results 
                                        if label in individual_results[model])
                improvement = ensemble_auc - best_individual_auc
                improvements.append(improvement)
        
        if improvements:
            summary["improvement_stats"][method_name] = {
                "avg_improvement": float(np.mean(improvements)),
                "max_improvement": float(np.max(improvements)),
                "min_improvement": float(np.min(improvements))
            }
    
    with open(analysis_dir / "ensemble_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("✅ Ensemble analysis complete!")
    print(f"📁 Results saved to: {analysis_dir}")
    print("\n📊 Generated files:")
    print("- *_predictions.csv: Ensemble prediction files")
    print("- ensemble_comparison.json: Detailed comparison metrics")
    print("- ensemble_summary.json: Summary statistics")

if __name__ == "__main__":
    create_ensemble_comparison()