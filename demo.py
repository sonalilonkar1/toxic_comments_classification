"""Interactive Demo for Toxic Comment Classification Presentation.

Features:
- Paste a comment and see per-label probabilities.
- Threshold slider to see flagged labels.
- Top-K mode to show flagged comments from test set.
- Model switch (LR, SVM, RF, BERT) with PR-AUC display.
- Explainability: Custom perturbation-based explanations for flagged comments using TF-IDF models only.
"""

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import torch
from sklearn.metrics import average_precision_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Add project root to path for imports
import sys
sys.path.append(str(Path(__file__).parent))
from src.data.preprocess import toy_normalize

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
MODELS_DIR = EXPERIMENTS_DIR / "train"

# Label columns
LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# Load models and data
@st.cache_resource
def load_model_data(model_name: str, fold: str = "fold1_seed42"):
    """Load trained model, vectorizer/tokenizer, and test data for a given model and fold."""
    if model_name == "bert":
        return load_bert_model_data(fold)
    else:
        return load_tfidf_model_data(model_name, fold)

@st.cache_resource
def load_tfidf_model_data(model_name: str, fold: str = "fold1_seed42"):
    """Load TF-IDF trained model, vectorizer, and test data for a given model and fold."""
    model_base_dir = MODELS_DIR / f"tfidf_{model_name}"
    
    # Find the latest timestamped directory for this fold
    if model_name == "xgboost":
        # XGBoost has different naming pattern: fold1_seed42-seed42-normtoy-modelxgboost-
        fold_prefix = f"{fold}-seed42-normtoy-model{model_name}-"
    else:
        # Standard pattern: fold1_seed42-normtoy-modellogistic-
        fold_prefix = f"{fold}-normtoy-model{model_name}-"
    
    fold_dirs = [d for d in model_base_dir.iterdir() if d.is_dir() and d.name.startswith(fold_prefix)]
    
    if not fold_dirs:
        raise FileNotFoundError(f"No directories found for {model_name} {fold}")
    
    # Get the latest (most recent) directory
    model_dir = sorted(fold_dirs, key=lambda x: x.name, reverse=True)[0]

    # Load vectorizer
    vectorizer_path = model_dir / "models" / "tfidf.joblib"
    vectorizer = joblib.load(vectorizer_path)

    # Load models (dict of label -> classifier)
    models = {}
    for label in LABEL_COLS:
        model_path = model_dir / "models" / f"{label}.joblib"
        models[label] = joblib.load(model_path)

    # Load test data
    split_path = DATA_DIR / "splits" / f"{fold}.json"
    with open(split_path, "r") as f:
        splits = json.load(f)
    test_indices = splits["test"]

    data_path = DATA_DIR / "raw" / "train.csv"
    df = pd.read_csv(data_path)
    test_df = df.iloc[test_indices].copy()
    test_texts = test_df["comment_text"].tolist()
    test_labels = test_df[LABEL_COLS].values

    # Load overall metrics for PR-AUC
    metrics_path = model_dir / "overall_metrics.json"
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    pr_auc = metrics.get("macro_pr_auc", 0.0)

    return vectorizer, models, test_texts, test_labels, pr_auc

@st.cache_resource
def load_bert_model_data(fold: str = "fold1_seed42"):
    """Load BERT model, tokenizer, and test data for a given fold."""
    model_base_dir = MODELS_DIR / "bert"
    
    # Find the latest fold directory
    fold_dirs = [d for d in model_base_dir.iterdir() if d.is_dir() and d.name == fold]
    
    if not fold_dirs:
        raise FileNotFoundError(f"No BERT directory found for {fold}")
    
    fold_dir = fold_dirs[0]
    
    # Find the model subdirectory (contains model/, tokenizer/, etc.)
    subdirs = [d for d in fold_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    if not subdirs:
        raise FileNotFoundError(f"No model subdirectory found in {fold_dir}")
    
    model_dir = sorted(subdirs, key=lambda x: x.name, reverse=True)[0]
    models_dir = model_dir / "model"
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(str(models_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(models_dir))
    model.eval()
    
    # Load test data
    split_path = DATA_DIR / "splits" / f"{fold}.json"
    with open(split_path, "r") as f:
        splits = json.load(f)
    test_indices = splits["test"]

    data_path = DATA_DIR / "raw" / "train.csv"
    df = pd.read_csv(data_path)
    test_df = df.iloc[test_indices].copy()
    test_texts = test_df["comment_text"].tolist()
    test_labels = test_df[LABEL_COLS].values

    # Load overall metrics for PR-AUC
    metrics_path = model_dir / "overall_metrics.json"
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    pr_auc = metrics.get("macro_pr_auc", 0.0)

    return tokenizer, model, test_texts, test_labels, pr_auc

@st.cache_resource
def load_explainer(model_name: str, fold: str = "fold1_seed42"):
    """Load explainer for the model - using custom implementation instead of SHAP."""
    if model_name == "bert":
        # BERT explanations not implemented
        return None, None
    
    model_base_dir = MODELS_DIR / f"tfidf_{model_name}"
    
    # Find the latest timestamped directory for this fold
    fold_prefix = f"{fold}-normtoy-model{model_name}-"
    fold_dirs = [d for d in model_base_dir.iterdir() if d.is_dir() and d.name.startswith(fold_prefix)]
    
    if not fold_dirs:
        raise FileNotFoundError(f"No directories found for {model_name} {fold}")
    
    # Get the latest (most recent) directory
    model_dir = sorted(fold_dirs, key=lambda x: x.name, reverse=True)[0]
    
    vectorizer, models, _, _, _ = load_tfidf_model_data(model_name, fold)

    # For simplicity, use the 'toxic' model as representative
    model = models["toxic"]
    
    # Return model and vectorizer for custom explanation
    return model, vectorizer
    
    return explainer, vectorizer

def predict_single_comment(text: str, _vectorizer_or_tokenizer, models_or_model, model_type: str):
    """Predict probabilities for a single comment."""
    if model_type == "bert":
        tokenizer, model = _vectorizer_or_tokenizer, models_or_model
        processed_text = toy_normalize(text)
        inputs = tokenizer([processed_text], padding=True, truncation=True, max_length=256, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs_tensor = torch.sigmoid(logits)
            probs_array = probs_tensor[0].cpu().numpy()
        
        probs = {label: float(prob) for label, prob in zip(LABEL_COLS, probs_array)}
    else:
        # TF-IDF models
        vectorizer, models = _vectorizer_or_tokenizer, models_or_model
        X = vectorizer.transform([text])
        probs = {}
        for label in LABEL_COLS:
            prob = models[label].predict_proba(X)[0, 1]  # Prob of positive class
            probs[label] = prob
    
    return probs

@st.cache_data
def get_top_k_flagged(_test_texts, _test_labels, _vectorizer_or_tokenizer, _models_or_model, _model_type: str, _k: int, _max_samples: int = 1000):
    """Get top-K comments most likely to be flagged (by max prob)."""
    # Limit analysis to first max_samples for performance
    n_samples = min(len(_test_texts), _max_samples)
    
    max_probs = []
    for i in range(n_samples):
        probs = predict_single_comment(_test_texts[i], _vectorizer_or_tokenizer, _models_or_model, _model_type)
        max_prob = max(probs.values())
        max_probs.append((i, max_prob))

    # Sort by max prob descending
    max_probs.sort(key=lambda x: x[1], reverse=True)
    top_k = max_probs[:_k]

    flagged_comments = []
    for idx, prob in top_k:
        comment = _test_texts[idx]
        true_labels = [LABEL_COLS[j] for j in range(len(LABEL_COLS)) if _test_labels[idx, j] == 1]
        flagged_comments.append({
            "comment": comment,
            "max_prob": prob,
            "true_labels": true_labels
        })

    return flagged_comments

def explain_comment(text: str, model_name: str, model, vectorizer):
    """Get custom explanation for a comment using perturbation analysis."""
    # Tokenize the text (simple split for now)
    words = text.lower().split()
    
    # Get baseline prediction
    baseline_X = vectorizer.transform([text])
    baseline_prob = model.predict_proba(baseline_X)[0, 1]
    
    # Calculate impact of removing each word
    word_impacts = []
    for i, word in enumerate(words):
        # Create text without this word
        words_without = words[:i] + words[i+1:]
        text_without = ' '.join(words_without)
        
        # Get prediction without this word
        X_without = vectorizer.transform([text_without])
        prob_without = model.predict_proba(X_without)[0, 1]
        
        # Calculate impact (how much the probability changed)
        impact = baseline_prob - prob_without
        
        word_impacts.append((word, impact))
    
    # Sort by absolute impact
    sorted_impacts = sorted(word_impacts, key=lambda x: abs(x[1]), reverse=True)
    
    # Separate positive and negative impacts
    top_positive = [(word, impact) for word, impact in sorted_impacts if impact > 0][:10]
    top_negative = [(word, impact) for word, impact in sorted_impacts if impact < 0][:10]
    
    return top_positive, top_negative

# Streamlit App
st.title("Toxic Comment Classification Demo")
st.markdown("Interactive demo for the presentation: Predict, threshold, top-K, model comparison, explainability, and experiments results.")

# Model selection
model_options = ["logistic", "svm", "random_forest", "xgboost", "bert"]
selected_model = st.selectbox("Select Model", model_options, index=0)

# Load data for selected model
try:
    if selected_model == "bert":
        tokenizer, model, test_texts, test_labels, pr_auc = load_model_data(selected_model)
        model_type = "bert"
        st.success(f"Loaded BERT model. Macro PR-AUC: {pr_auc:.3f}")
    else:
        vectorizer, models, test_texts, test_labels, pr_auc = load_model_data(selected_model)
        model_type = "tfidf"
        st.success(f"Loaded {selected_model} model. Macro PR-AUC: {pr_auc:.3f}")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Comment input
st.subheader("1. Single Comment Prediction")
comment = st.text_area("Paste a comment here:", "This is a great comment!")

if st.button("Predict"):
    probs = predict_single_comment(comment, vectorizer if model_type == "tfidf" else tokenizer, 
                                   models if model_type == "tfidf" else model, model_type)
    st.session_state.last_prediction = probs
    st.session_state.last_comment = comment

# Show prediction results if available
if 'last_prediction' in st.session_state:
    probs = st.session_state.last_prediction
    st.write("Per-Label Probabilities:")
    for label, prob in probs.items():
        st.write(f"- {label}: {prob:.3f}")

    # Threshold slider (always visible when there's a prediction)
    threshold = st.slider("Threshold for Flagging", 0.0, 1.0, 0.5, 0.01, key="threshold_slider")
    flagged = [label for label, prob in probs.items() if prob >= threshold]
    st.write(f"Flagged Labels (≥ {threshold}): {flagged}")
    
    # Show the comment that was predicted
    st.write(f"**Predicted comment:** {st.session_state.last_comment}")

# Top-K Mode
st.subheader("2. Top-K Flagged Comments")
k = st.number_input("Top-K (e.g., 10 for top 10)", min_value=1, max_value=50, value=10)
max_samples = st.slider("Analyze first N samples", min_value=100, max_value=2000, value=500, step=100)

if st.button("Show Top-K"):
    with st.spinner(f"Analyzing {max_samples} comments... This may take a moment."):
        try:
            top_k_comments = get_top_k_flagged(test_texts, test_labels, 
                                                vectorizer if model_type == "tfidf" else tokenizer,
                                                models if model_type == "tfidf" else model, 
                                                model_type, k, max_samples)
            
            if top_k_comments:
                st.write(f"Top {k} Most Likely Flagged Comments (from {max_samples} analyzed):")
                for i, item in enumerate(top_k_comments):
                    st.write(f"**{i+1}.** Max Prob: {item['max_prob']:.3f}")
                    st.write(f"Comment: {item['comment'][:200]}...")
                    if item['true_labels']:
                        st.write(f"True Labels: {item['true_labels']}")
                    else:
                        st.write("True Labels: None")
                    st.write("---")
            else:
                st.error("No comments found. Try increasing the sample size or using a different model.")
                
        except Exception as e:
            st.error(f"Error analyzing comments: {e}")
            st.info("Try using a TF-IDF model (Logistic/SVM) instead of BERT for faster analysis.")

# Explainability
st.subheader("3. Explainability for a Flagged Comment")
if model_type == "bert":
    st.info("🔍 Custom explainability shows how removing words affects toxicity predictions. This feature is only available for TF-IDF based models (Logistic, SVM, Random Forest, XGBoost).")
    explain_available = False
else:
    explain_available = True

if explain_available:
    explain_comment_input = st.text_area("Paste a flagged comment for explanation:", comment)

    if st.button("Explain"):
        try:
            model, vec = load_explainer(selected_model)
            top_pos, top_neg = explain_comment(explain_comment_input, selected_model, model, vec)
            st.write("**Words that INCREASE toxicity probability when removed:**")
            for word, impact in top_pos:
                st.write(f"- '{word}': +{impact:.4f} (removing it decreases toxicity)")
            st.write("**Words that DECREASE toxicity probability when removed:**")
            for word, impact in top_neg:
                st.write(f"- '{word}': {impact:.4f} (removing it increases toxicity)")
        except Exception as e:
            st.error(f"Explanation failed: {e}")
else:
    st.write("💡 **BERT models** use advanced contextual understanding, making traditional word-level explanations complex. Consider using the model's attention weights or other interpretability methods for BERT.")

st.markdown("---")
st.markdown("**Error Taxonomy Link**: If a comment is flagged due to profanity-only (e.g., 'damn' high score), it's a false positive under context-aware rules. Adjust thresholds or add context features.")
st.markdown("**Model Notes**: BERT provides better contextual understanding than TF-IDF models but lacks SHAP explainability. Use TF-IDF models for detailed word-level explanations.")