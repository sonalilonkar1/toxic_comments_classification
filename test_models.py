"""Test script to check model loading."""

from pathlib import Path
import joblib
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

def test_tfidf_loading():
    """Test loading TF-IDF models."""
    model_name = "tfidf_logistic"
    model_dir = Path("experiments/train") / model_name
    
    fold_dirs = sorted([d for d in model_dir.iterdir() if d.is_dir()], key=lambda x: x.name, reverse=True)
    if not fold_dirs:
        print(f"No fold directories found for {model_name}")
        return False
    
    latest_fold = fold_dirs[0]
    models_dir = latest_fold / "models"
    
    try:
        tfidf = joblib.load(models_dir / "tfidf.joblib")
        classifiers = {}
        for label in LABEL_COLS:
            classifiers[label] = joblib.load(models_dir / f"{label}.joblib")
        print(f"Successfully loaded {model_name}")
        return True
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        return False

def test_bert_loading():
    """Test loading BERT model."""
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        model_dir = Path("experiments/train/bert")
        fold_dirs = sorted([d for d in model_dir.iterdir() if d.is_dir()], key=lambda x: x.name, reverse=True)
        if not fold_dirs:
            print("No BERT fold directories found")
            return False
        
        latest_fold = fold_dirs[0]
        # Find the model subdirectory
        subdirs = [d for d in latest_fold.iterdir() if d.is_dir() and not d.name.startswith('.')]
        if not subdirs:
            print("No model subdirectories found")
            return False
        
        model_subdir = sorted(subdirs, key=lambda x: x.name, reverse=True)[0]
        models_dir = model_subdir / "model"
        
        tokenizer = AutoTokenizer.from_pretrained(str(models_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(models_dir))
        print("Successfully loaded BERT model")
        return True
    except Exception as e:
        print(f"Failed to load BERT model: {e}")
        return False

if __name__ == "__main__":
    print("Testing model loading...")
    tfidf_ok = test_tfidf_loading()
    bert_ok = test_bert_loading()
    print(f"TF-IDF: {'OK' if tfidf_ok else 'FAILED'}")
    print(f"BERT: {'OK' if bert_ok else 'FAILED'}")