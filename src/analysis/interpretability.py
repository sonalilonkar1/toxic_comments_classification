"""SHAP and Deep Learning Interpretability analysis."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from src.features.tfidf import create_tfidf_vectorizer

# Deep Learning Imports
try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from src.models.deep.lstm import MultiLabelLSTM
    from src.features.lstm_preprocessing import LSTMPreprocessor
    from src.models.deep.load_lstm import load_lstm_artifacts
    DEEP_AVAILABLE = True
except ImportError:
    DEEP_AVAILABLE = False


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


# -------------------------------------------------------------------------
# Deep Learning Interpretability
# -------------------------------------------------------------------------

def run_deep_interpretability(
    fold_dir: Path,
    model_type: str,
    X_sample: List[str],
    output_dir: Optional[Path] = None,
    target_label: Optional[str] = None,
    label_cols: Optional[List[str]] = None,
):
    """Generate interpretability plots (Attention/Saliency) for Deep models."""
    if not DEEP_AVAILABLE:
        print("Deep learning libraries (torch, transformers) not found. Skipping.")
        return

    if output_dir is None:
        output_dir = fold_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running Deep Learning Interpretability for {model_type}...")
    
    if model_type == "bert":
        run_bert_interpretability(fold_dir, X_sample, output_dir, label_cols=label_cols)
    elif model_type == "lstm":
        run_lstm_interpretability(fold_dir, X_sample, output_dir, label_cols=label_cols)
    else:
        print(f"Unknown deep model type: {model_type}")


def run_bert_interpretability(
    fold_dir: Path,
    X_sample: List[str],
    output_dir: Path,
    label_cols: Optional[List[str]] = None,
):
    """Generate Attention plots for BERT."""
    model_dir = fold_dir / "final_model"
    if not model_dir.exists():
        # Fallback to checkpoints if final model not found
        ckpt_dir = fold_dir / "checkpoints"
        if ckpt_dir.exists():
            # Find latest checkpoint
            ckpts = sorted([d for d in ckpt_dir.glob("checkpoint-*") if d.is_dir()], key=lambda x: int(x.name.split("-")[1]))
            if ckpts:
                model_dir = ckpts[-1]
            else:
                print(f"No model found in {fold_dir}")
                return
        else:
            print(f"No model found in {fold_dir}")
            return

    print(f"Loading BERT from {model_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), output_attentions=True)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Failed to load BERT: {e}")
        return

    # Process samples
    print("Generating Attention plots...")
    
    # We visualize the attention of the [CLS] token (index 0) to all other tokens
    # in the last layer, averaged across heads.
    
    for i, text in enumerate(X_sample):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_ids = inputs["input_ids"]
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        
        with torch.no_grad():
            outputs = model(**inputs)
            # attentions: tuple of (batch, num_heads, seq_len, seq_len) per layer
            attentions = outputs.attentions
            
        # Get last layer attention
        last_layer_attn = attentions[-1] # (1, num_heads, seq_len, seq_len)
        
        # Average across heads
        avg_attn = last_layer_attn.mean(dim=1).squeeze(0) # (seq_len, seq_len)
        
        # Focus on [CLS] token attention (index 0)
        cls_attn = avg_attn[0, :].cpu().numpy() # (seq_len,)
        
        # Plot
        plot_token_importance(
            tokens, 
            cls_attn, 
            f"BERT Attention (Last Layer) - Sample {i+1}", 
            output_dir / f"bert_attention_sample_{i+1}.png"
        )


def run_lstm_interpretability(
    fold_dir: Path,
    X_sample: List[str],
    output_dir: Path,
    label_cols: Optional[List[str]] = None,
):
    """Generate Saliency plots for LSTM (Input Gradients)."""
    model_dir = fold_dir / "models"
    if not model_dir.exists():
        print(f"No model dir found: {model_dir}")
        return

    print(f"Loading LSTM from {model_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        model, preprocessor, config = load_lstm_artifacts(model_dir, device=device)
        # Ensure model is in eval mode but gradients enabled for saliency
        model.eval() 
    except Exception as e:
        print(f"Failed to load LSTM: {e}")
        return

    print("Generating Saliency plots...")
    
    for i, text in enumerate(X_sample):
        # Preprocess single text
        seq = preprocessor.transform([text])
        input_ids = torch.LongTensor(seq).to(device)
        
        # Get tokens for visualization
        # Note: tokenizer.sequences_to_texts returns string, we want list of words
        # Reconstruct from input_ids (handling OOV/padding)
        # We can use the preprocessor's index_word
        idx2word = preprocessor.get_index_word()
        tokens = []
        for idx in seq[0]:
            if idx == 0:
                tokens.append("<PAD>")
            else:
                tokens.append(idx2word.get(idx, "<OOV>"))
        
        # Forward pass with gradient tracking
        # We need to manually perform the forward pass to hook embeddings
        
        # 1. Embeddings
        embeds = model.embedding(input_ids)
        embeds.retain_grad()
        
        # 2. LSTM
        lstm_out, (h, c) = model.lstm(embeds)
        
        if model.bidirectional:
            hidden = torch.cat((h[-2], h[-1]), dim=1)
        else:
            hidden = h[-1]
            
        output = model.dropout(hidden)
        logits = model.fc(output)
        
        # 3. Predict top class
        probs = torch.sigmoid(logits)
        top_label_idx = torch.argmax(probs[0]).item()
        top_label = label_cols[top_label_idx] if label_cols else str(top_label_idx)
        top_prob = probs[0, top_label_idx].item()
        
        # 4. Backward for Saliency
        # We want gradient of the top class score w.r.t input embeddings
        model.zero_grad()
        logits[0, top_label_idx].backward()
        
        # 5. Get gradients
        grads = embeds.grad[0] # (seq_len, embed_dim)
        # Magnitude of gradient
        saliency = torch.norm(grads, dim=1).cpu().numpy()
        
        # Plot
        plot_token_importance(
            tokens, 
            saliency, 
            f"LSTM Saliency (Label: {top_label}, Prob: {top_prob:.2f}) - Sample {i+1}", 
            output_dir / f"lstm_saliency_sample_{i+1}.png"
        )


def plot_token_importance(
    tokens: List[str], 
    scores: np.ndarray, 
    title: str, 
    output_path: Path
):
    """Plot token importance scores as a bar chart."""
    # Filter out padding for cleaner plot
    valid_indices = [i for i, t in enumerate(tokens) if t not in ["<PAD>", "[PAD]"]]
    if not valid_indices:
        return
        
    filtered_tokens = [tokens[i] for i in valid_indices]
    filtered_scores = scores[valid_indices]
    
    # Normalize scores for visualization
    if filtered_scores.max() > 0:
        filtered_scores = filtered_scores / filtered_scores.max()
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(filtered_tokens)), filtered_scores)
    
    # Color bars by intensity
    for bar, score in zip(bars, filtered_scores):
        bar.set_color(plt.cm.viridis(score))
        
    plt.xticks(range(len(filtered_tokens)), filtered_tokens, rotation=45, ha="right")
    plt.title(title)
    plt.ylabel("Importance Score")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
