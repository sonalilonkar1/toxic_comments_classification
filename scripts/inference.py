"""Inference script for toxic comment classification models."""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocess import toy_normalize
from src.utils.metrics import probs_to_preds

LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

def load_tfidf_model(model_name: str):
    """Load TF-IDF model components."""
    model_dir = Path("experiments/train") / model_name
    
    # Find the latest fold directory (assuming timestamped names)
    fold_dirs = sorted([d for d in model_dir.iterdir() if d.is_dir()], key=lambda x: x.name, reverse=True)
    if not fold_dirs:
        raise FileNotFoundError(f"No fold directories found for {model_name}")
    
    latest_fold = fold_dirs[0]
    models_dir = latest_fold / "models"
    
    tfidf = joblib.load(models_dir / "tfidf.joblib")
    classifiers = {}
    for label in LABEL_COLS:
        classifiers[label] = joblib.load(models_dir / f"{label}.joblib")
    
    return tfidf, classifiers

def load_bert_model():
    """Load BERT model components."""
    model_dir = Path("experiments/train/bert")
    
    # Find the latest fold directory
    fold_dirs = sorted([d for d in model_dir.iterdir() if d.is_dir()], key=lambda x: x.name, reverse=True)
    if not fold_dirs:
        raise FileNotFoundError("No BERT fold directories found")
    
    latest_fold = fold_dirs[0]
    models_dir = latest_fold / "models" / "bert"
    
    tokenizer = AutoTokenizer.from_pretrained(models_dir)
    model = AutoModelForSequenceClassification.from_pretrained(models_dir)
    model.eval()
    
    return tokenizer, model

def predict_tfidf(texts: List[str], tfidf, classifiers):
    """Predict with TF-IDF model."""
    # Preprocess
    processed_texts = [toy_normalize(text) for text in texts]
    
    # Vectorize
    X = tfidf.transform(processed_texts)
    
    # Predict probabilities
    probs = {}
    for label in LABEL_COLS:
        clf = classifiers[label]
        probs[label] = clf.predict_proba(X)[:, 1]
    
    # Convert to DataFrame
    prob_df = pd.DataFrame(probs)
    pred_df = probs_to_preds(probs).astype(int)
    pred_df = pd.DataFrame(pred_df, columns=[f"{label}_pred" for label in LABEL_COLS])
    
    return prob_df, pred_df

def predict_bert(texts: List[str], tokenizer, model, device="cpu"):
    """Predict with BERT model."""
    model.to(device)
    
    # Preprocess
    processed_texts = [toy_normalize(text) for text in texts]
    
    # Tokenize
    inputs = tokenizer(processed_texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits).cpu().numpy()
    
    # Convert to DataFrame
    prob_df = pd.DataFrame(probs, columns=[f"{label}_prob" for label in LABEL_COLS])
    pred_df = (probs > 0.5).astype(int)
    pred_df = pd.DataFrame(pred_df, columns=[f"{label}_pred" for label in LABEL_COLS])
    
    return prob_df, pred_df

def main():
    parser = argparse.ArgumentParser(description="Run inference on new data")
    parser.add_argument("--model", type=str, required=True, 
                       choices=["tfidf_logistic", "tfidf_svm", "tfidf_random_forest", "bert"],
                       help="Model to use for prediction")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file with 'comment_text' column")
    parser.add_argument("--output", type=str, required=True, help="Output CSV file for predictions")
    parser.add_argument("--text-col", type=str, default="comment_text", help="Column name for text")
    
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.input)
    if args.text_col not in df.columns:
        raise ValueError(f"Column '{args.text_col}' not found in input CSV")
    
    texts = df[args.text_col].fillna("").tolist()
    
    print(f"Loaded {len(texts)} texts from {args.input}")
    
    # Load model
    if args.model.startswith("tfidf"):
        print(f"Loading TF-IDF {args.model} model...")
        tfidf, classifiers = load_tfidf_model(args.model)
        prob_df, pred_df = predict_tfidf(texts, tfidf, classifiers)
    elif args.model == "bert":
        print("Loading BERT model...")
        tokenizer, model = load_bert_model()
        prob_df, pred_df = predict_bert(texts, tokenizer, model)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    # Combine results
    result_df = pd.concat([df, prob_df, pred_df], axis=1)
    
    # Save
    result_df.to_csv(args.output, index=False)
    print(f"Saved predictions to {args.output}")
    print(f"Sample predictions:")
    print(result_df.head())

if __name__ == "__main__":
    main()