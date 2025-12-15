"""Model interpretability analysis using SHAP for toxic comment classification."""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP not available. Install with: pip install shap")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
    plt.style.use('default')
    sns.set_palette("husl")
except ImportError:
    PLOTTING_AVAILABLE = False

LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

def load_model_data(model_name: str):
    """Load model and data for interpretability analysis."""
    exp_dir = Path("experiments/train") / model_name
    if not exp_dir.exists():
        return None, None

    # Load predictions
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
                return None, None
    else:
        timestamp_dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir() and not d.name.startswith('.')], 
                               key=lambda x: x.name, reverse=True)
        if timestamp_dirs:
            pred_file = timestamp_dirs[0] / "test_predictions.csv"
        else:
            return None, None

    if pred_file.exists():
        pred_df = pd.read_csv(pred_file)
        return pred_df, model_name
    return None, None

def analyze_tfidf_interpretability():
    """Analyze TF-IDF model interpretability."""
    print("🔍 Analyzing TF-IDF model interpretability...")

    if not SHAP_AVAILABLE:
        return {"error": "SHAP not available"}

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        import joblib

        # Load TF-IDF logistic model components
        model_name = "tfidf_logistic"
        exp_dir = Path("experiments/train") / model_name
        timestamp_dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir() and not d.name.startswith('.')], 
                               key=lambda x: x.name, reverse=True)

        if not timestamp_dirs:
            return {"error": "No TF-IDF model found"}

        model_dir = timestamp_dirs[0] / "models"
        tfidf_vectorizer = joblib.load(model_dir / "tfidf.joblib")

        # Load sample data for analysis
        pred_df, _ = load_model_data(model_name)
        if pred_df is None:
            return {"error": "Could not load predictions"}

        # Sample some toxic and non-toxic comments
        toxic_comments = pred_df[pred_df["toxic"] == 1]["comment_text"].head(50).tolist()
        clean_comments = pred_df[pred_df["toxic"] == 0]["comment_text"].head(50).tolist()

        sample_texts = toxic_comments + clean_comments

        # Transform texts
        X_tfidf = tfidf_vectorizer.transform(sample_texts)

        # Get feature names
        feature_names = tfidf_vectorizer.get_feature_names_out()

        # Analyze top features for toxic vs clean
        toxic_tfidf = tfidf_vectorizer.transform(toxic_comments).toarray()
        clean_tfidf = tfidf_vectorizer.transform(clean_comments).toarray()

        # Calculate mean TF-IDF scores
        toxic_means = np.mean(toxic_tfidf, axis=0)
        clean_means = np.mean(clean_tfidf, axis=0)

        # Find most distinctive features
        toxic_unique = toxic_means - clean_means
        clean_unique = clean_means - toxic_means

        # Get top features
        top_toxic_features = [(feature_names[i], toxic_unique[i]) for i in toxic_unique.argsort()[-20:][::-1]]
        top_clean_features = [(feature_names[i], clean_unique[i]) for i in clean_unique.argsort()[-20:][::-1]]

        return {
            "top_toxic_features": top_toxic_features,
            "top_clean_features": top_clean_features,
            "sample_size": len(sample_texts)
        }

    except Exception as e:
        return {"error": str(e)}

def analyze_bert_interpretability():
    """Analyze BERT model interpretability."""
    print("🧠 Analyzing BERT model interpretability...")

    if not SHAP_AVAILABLE:
        return {"error": "SHAP not available"}

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        # Load BERT model
        model_dir = Path("experiments/train/bert")
        fold_dirs = sorted([d for d in model_dir.iterdir() if d.is_dir() and not d.name.startswith('.')], 
                          key=lambda x: x.name, reverse=True)

        if not fold_dirs:
            return {"error": "No BERT model found"}

        latest_fold = fold_dirs[0]
        timestamp_dirs = sorted([d for d in latest_fold.iterdir() if d.is_dir() and not d.name.startswith('.')], 
                               key=lambda x: x.name, reverse=True)

        if not timestamp_dirs:
            return {"error": "No BERT model timestamp directory found"}

        models_dir = timestamp_dirs[0] / "model"
        tokenizer = AutoTokenizer.from_pretrained(str(models_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(models_dir))
        model.eval()

        # Load sample data
        pred_df, _ = load_model_data("bert")
        if pred_df is None:
            return {"error": "Could not load BERT predictions"}

        # Sample some comments for analysis
        sample_comments = pred_df["comment_text"].head(10).tolist()

        # Create a prediction function for SHAP
        def predict_proba(texts):
            inputs = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1).numpy()
            return probs

        # Use SHAP for text analysis
        explainer = shap.Explainer(predict_proba, tokenizer)
        shap_values = explainer(sample_comments)

        # Extract insights
        bert_insights = {
            "sample_size": len(sample_comments),
            "shap_available": True,
            "model_type": "BERT",
            "sample_explanations": []
        }

        # Analyze a few examples
        for i, text in enumerate(sample_comments[:3]):
            # Get prediction probabilities
            probs = predict_proba([text])[0]
            predicted_class = np.argmax(probs)

            bert_insights["sample_explanations"].append({
                "text": text[:100] + "..." if len(text) > 100 else text,
                "predicted_class": predicted_class,
                "probabilities": probs.tolist()
            })

        return bert_insights

    except Exception as e:
        return {"error": str(e)}

def create_feature_importance_analysis():
    """Create comprehensive feature importance analysis."""
    print("📊 Creating feature importance analysis...")

    analysis_dir = Path("reports/interpretability")
    analysis_dir.mkdir(exist_ok=True, parents=True)

    interpretability_report = {
        "generated_at": datetime.now().isoformat(),
        "shap_available": SHAP_AVAILABLE,
        "plotting_available": PLOTTING_AVAILABLE,
        "analyses": {}
    }

    # 1. TF-IDF Interpretability
    tfidf_analysis = analyze_tfidf_interpretability()
    interpretability_report["analyses"]["tfidf_interpretability"] = tfidf_analysis

    # 2. BERT Interpretability
    bert_analysis = analyze_bert_interpretability()
    interpretability_report["analyses"]["bert_interpretability"] = bert_analysis

    # 3. Comparative Insights
    comparative_insights = {
        "model_comparison": {
            "tfidf_strengths": "Identifies specific toxic words and phrases",
            "bert_strengths": "Understands context and nuanced language patterns",
            "complementary": "TF-IDF good for explicit toxic words, BERT good for implicit toxicity"
        },
        "practical_applications": [
            "Use TF-IDF for fast, interpretable filtering of obvious toxic content",
            "Use BERT for nuanced content moderation requiring context understanding",
            "Combine both for robust multi-stage toxicity detection"
        ]
    }

    interpretability_report["analyses"]["comparative_insights"] = comparative_insights

    # Save report
    with open(analysis_dir / "interpretability_report.json", 'w') as f:
        json.dump(interpretability_report, f, indent=2)

    print("✅ Feature importance analysis complete!")
    print(f"📁 Results saved to: {analysis_dir}")
    print("\n📊 Generated files:")
    print("- interpretability_report.json: Comprehensive interpretability analysis")

    return interpretability_report

def create_interpretability_plots():
    """Create visualization plots for interpretability."""
    if not PLOTTING_AVAILABLE:
        print("⚠️ Plotting not available, skipping visualization")
        return

    analysis_dir = Path("reports/interpretability")

    # Load TF-IDF analysis results
    try:
        with open(analysis_dir / "interpretability_report.json", 'r') as f:
            report = json.load(f)

        tfidf_data = report["analyses"]["tfidf_interpretability"]

        if "top_toxic_features" in tfidf_data:
            # Create feature importance plot
            features, scores = zip(*tfidf_data["top_toxic_features"][:15])

            plt.figure(figsize=(12, 8))
            plt.barh(range(len(features)), scores)
            plt.yticks(range(len(features)), features)
            plt.xlabel('TF-IDF Score Difference (Toxic - Clean)')
            plt.title('Top Features Indicative of Toxic Content (TF-IDF)')
            plt.tight_layout()
            plt.savefig(analysis_dir / "tfidf_toxic_features.png", dpi=300, bbox_inches='tight')
            plt.close()

            print("📊 Created TF-IDF feature importance plot")

    except Exception as e:
        print(f"⚠️ Could not create plots: {e}")

if __name__ == "__main__":
    create_feature_importance_analysis()
    create_interpretability_plots()