"""Flask API for toxic comment classification."""

from flask import Flask, request, jsonify
import pandas as pd
from pathlib import Path
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocess import toy_normalize
from scripts.monitoring import api_monitor, monitor_request, get_monitoring_stats, get_monitoring_report

LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

app = Flask(__name__)

# Global model variables
tfidf_models = {}
bert_model = None
bert_tokenizer = None

def load_models():
    """Load all models on startup."""
    global tfidf_models, bert_model, bert_tokenizer
    
    # Load TF-IDF models
    for model_name in ["tfidf_logistic", "tfidf_svm", "tfidf_random_forest"]:
        try:
            model_dir = Path("experiments/train") / model_name
            fold_dirs = sorted([d for d in model_dir.iterdir() if d.is_dir()], key=lambda x: x.name, reverse=True)
            if fold_dirs:
                latest_fold = fold_dirs[0]
                models_dir = latest_fold / "models"
                tfidf = joblib.load(models_dir / "tfidf.joblib")
                classifiers = {}
                for label in LABEL_COLS:
                    classifiers[label] = joblib.load(models_dir / f"{label}.joblib")
                tfidf_models[model_name] = {"tfidf": tfidf, "classifiers": classifiers}
                print(f"Loaded {model_name}")
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
    
    # Load BERT model
    try:
        model_dir = Path("experiments/train/bert")
        fold_dirs = sorted([d for d in model_dir.iterdir() if d.is_dir()], key=lambda x: x.name, reverse=True)
        if fold_dirs:
            latest_fold = fold_dirs[0]
            # Find the model subdirectory (contains model/, tokenizer/, etc.)
            subdirs = [d for d in latest_fold.iterdir() if d.is_dir() and not d.name.startswith('.')]
            if subdirs:
                model_subdir = sorted(subdirs, key=lambda x: x.name, reverse=True)[0]
                models_dir = model_subdir / "model"
                bert_tokenizer = AutoTokenizer.from_pretrained(str(models_dir))
                bert_model = AutoModelForSequenceClassification.from_pretrained(str(models_dir))
                bert_model.eval()
                print("Loaded BERT model")
    except Exception as e:
        print(f"Failed to load BERT model: {e}")

def predict_tfidf(text: str, tfidf, classifiers):
    """Predict with TF-IDF model."""
    processed_text = toy_normalize(text)
    X = tfidf.transform([processed_text])
    
    probs = {}
    preds = {}
    for label in LABEL_COLS:
        clf = classifiers[label]
        prob = clf.predict_proba(X)[0, 1]
        pred = int(prob > 0.5)
        probs[label] = float(prob)
        preds[label] = pred
    
    return probs, preds

def predict_bert(text: str, tokenizer, model):
    """Predict with BERT model."""
    processed_text = toy_normalize(text)
    
    inputs = tokenizer([processed_text], padding=True, truncation=True, max_length=256, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs_tensor = torch.sigmoid(logits)
        probs = probs_tensor[0].cpu().numpy()
    
    preds = (probs > 0.5).astype(int)
    
    result_probs = {label: float(prob) for label, prob in zip(LABEL_COLS, probs)}
    result_preds = {label: int(pred) for label, pred in zip(LABEL_COLS, preds)}
    
    return result_probs, result_preds

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify(get_monitoring_stats())

@app.route('/monitoring', methods=['GET'])
def monitoring():
    """Detailed monitoring report endpoint."""
    return jsonify(get_monitoring_report())

@app.route('/predict', methods=['POST'])
@monitor_request('predict', 'dynamic')  # Model will be determined from request
def predict():
    """Prediction endpoint."""
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' field"}), 400
    
    text = data['text']
    model_name = data.get('model', 'tfidf_logistic')
    
    if model_name not in tfidf_models and model_name != 'bert':
        return jsonify({"error": f"Model '{model_name}' not available"}), 400
    
    try:
        if model_name == 'bert':
            if not bert_model:
                return jsonify({"error": "BERT model not loaded"}), 500
            probs, preds = predict_bert(text, bert_tokenizer, bert_model)
        else:
            model_components = tfidf_models[model_name]
            probs, preds = predict_tfidf(text, model_components['tfidf'], model_components['classifiers'])
        
        # Update the monitor with the actual model used
        api_monitor.model_usage[model_name] += 1
        api_monitor.model_usage['dynamic'] -= 1  # Remove the placeholder
        
        return jsonify({
            "text": text,
            "model": model_name,
            "probabilities": probs,
            "predictions": preds
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
@monitor_request('batch_predict', 'dynamic')
def batch_predict():
    """Batch prediction endpoint."""
    data = request.get_json()
    
    if not data or 'texts' not in data:
        return jsonify({"error": "Missing 'texts' field"}), 400
    
    texts = data['texts']
    model_name = data.get('model', 'tfidf_logistic')
    
    if model_name not in tfidf_models and model_name != 'bert':
        return jsonify({"error": f"Model '{model_name}' not available"}), 400
    
    if not isinstance(texts, list):
        return jsonify({"error": "'texts' must be a list"}), 400
    
    try:
        results = []
        for text in texts:
            if model_name == 'bert':
                if not bert_model:
                    return jsonify({"error": "BERT model not loaded"}), 500
                probs, preds = predict_bert(text, bert_tokenizer, bert_model)
            else:
                model_components = tfidf_models[model_name]
                probs, preds = predict_tfidf(text, model_components['tfidf'], model_components['classifiers'])
            
            results.append({
                "text": text,
                "probabilities": probs,
                "predictions": preds
            })
        
        # Update the monitor with the actual model used
        api_monitor.model_usage[model_name] += 1
        api_monitor.model_usage['dynamic'] -= 1  # Remove the placeholder
        
        return jsonify({
            "model": model_name,
            "results": results
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    load_models()
    app.run(host='0.0.0.0', port=5001, debug=True)