"""SHAP analysis for Bag-of-Words models."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from src.features.tfidf import create_tfidf_vectorizer

def load_model_artifacts(fold_dir: Path) -> Dict[str, Any]:
    """Load trained model and vectorizer from a fold directory."""
    model_dir = fold_dir / "models"
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
    artifacts = {}
    
    # Load Vectorizer
    tfidf_path = model_dir / "tfidf.joblib"
    if not tfidf_path.exists():
        raise FileNotFoundError("TF-IDF vectorizer not found.")
    artifacts["tfidf"] = joblib.load(tfidf_path)
    
    # Load Models (one per label)
    label_models = {}
    for model_path in model_dir.glob("*.joblib"):
        if model_path.name == "tfidf.joblib":
            continue
        label = model_path.stem
        label_models[label] = joblib.load(model_path)
    
    artifacts["models"] = label_models
    return artifacts


def run_shap_analysis(
    fold_dir: Path,
    X_sample: List[str],
    output_dir: Optional[Path] = None,
    top_k_words: int = 20
):
    """Generate SHAP summary plots for BoW models.
    
    Note: SHAP KernelExplainer is slow. We use a small background sample.
    """
    if output_dir is None:
        output_dir = fold_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading artifacts from {fold_dir}...")
    artifacts = load_model_artifacts(fold_dir)
    tfidf = artifacts["tfidf"]
    label_models = artifacts["models"]
    
    # Transform sample to feature space
    X_vec = tfidf.transform(X_sample)
    feature_names = get_feature_names(tfidf)
    
    # Convert sparse to dense for SHAP (warning: memory intensive)
    # Ideally use LinearExplainer for linear models which handles sparse, 
    # but KernelExplainer is generic. For Logistic/LinearSVM, we can check coefs directly too.
    
    # For linear models, we can use the coefficients directly for global importance
    # independent of SHAP for speed, but SHAP is requested.
    
    print("Generating SHAP plots...")
    
    for label, model in label_models.items():
        print(f"Analyzing label: {label}")
        
        # Check if model is supported by TreeExplainer (RF/XGB) or LinearExplainer (LogReg/SVM)
        explainer = None
        
        try:
            # Try Linear Explainer first (fastest for LogReg/SVM)
            explainer = shap.LinearExplainer(model, X_vec, feature_perturbation="interventional")
            shap_values = explainer.shap_values(X_vec)
        except Exception:
            try:
                # Try Tree Explainer (RF/XGB)
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_vec)
                
                # TreeExplainer for binary class might return list [neg, pos] or just pos
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
            except Exception as e:
                print(f"Skipping SHAP for {label}: {e} (KernelExplainer too slow for full run)")
                continue

        if explainer:
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_vec, feature_names=feature_names, show=False, max_display=top_k_words)
            plt.title(f"SHAP Summary: {label}")
            plt.tight_layout()
            plt.savefig(output_dir / f"shap_summary_{label}.png")
            plt.close()


def get_feature_names(vectorizer: Any) -> List[str]:
    """Extract feature names from Vectorizer or FeatureUnion."""
    if isinstance(vectorizer, FeatureUnion):
        names = []
        for name, trans in vectorizer.transformer_list:
            if hasattr(trans, "get_feature_names_out"):
                names.extend(trans.get_feature_names_out())
            else:
                names.extend(trans.get_feature_names())
        return names
    
    if hasattr(vectorizer, "get_feature_names_out"):
        return vectorizer.get_feature_names_out()
    return vectorizer.get_feature_names()

