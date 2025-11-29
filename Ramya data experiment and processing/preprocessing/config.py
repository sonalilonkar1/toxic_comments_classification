"""Configuration parameters for preprocessing."""

from pathlib import Path
from typing import Dict, Any

# Project root (assuming this file is in Ramya data experiment and processing/preprocessing/)
ROOT = Path(__file__).resolve().parents[2]

# Data paths
DATA_DIR = ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
SPLITS_DIR = DATA_DIR / "splits"
ARTIFACTS_DIR = ROOT / "artifacts"
PREPROCESSED_DIR = ARTIFACTS_DIR / "preprocessed"

# Label columns
LABEL_COLUMNS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

# Text cleaning configuration
TEXT_CLEANING_CONFIG = {
    'lowercase': True,
    'remove_urls': True,
    'remove_emails': True,
    'expand_contractions': True,
    'remove_special_chars': True,
    'keep_punctuation': True,
    'remove_stopwords': False,  # Set to True for traditional ML
    'lemmatize': False,  # Set to True for traditional ML
}

# Traditional ML vectorization configuration
VECTORIZER_CONFIG = {
    'tfidf': {
        'max_features': 50000,
        'ngram_range': (1, 2),  # Unigrams and bigrams
        'min_df': 2,
        'max_df': 0.95,
        'lowercase': True,
        'stop_words': 'english',
    },
    'count': {
        'max_features': 50000,
        'ngram_range': (1, 2),
        'min_df': 2,
        'max_df': 0.95,
        'lowercase': True,
        'stop_words': 'english',
        'binary': False,
    },
}

# LSTM preprocessing configuration
LSTM_CONFIG = {
    'vocab_size': 10000,
    'max_length': None,  # Will be determined from data
    'oov_token': '<OOV>',
    'padding': 'post',
    'truncating': 'post',
}

# BERT/DistilBERT preprocessing configuration
BERT_CONFIG = {
    'bert-base-uncased': {
        'model_name': 'bert-base-uncased',
        'max_length': 128,
        'padding': True,
        'truncation': True,
    },
    'distilbert-base-uncased': {
        'model_name': 'distilbert-base-uncased',
        'max_length': 128,
        'padding': True,
        'truncation': True,
    },
}

# Default model configurations
DEFAULT_MODEL_CONFIGS = {
    'naive_bayes': {
        'vectorizer_type': 'tfidf',
        'vectorizer_config': VECTORIZER_CONFIG['tfidf'],
        'text_cleaning': {**TEXT_CLEANING_CONFIG, 'remove_stopwords': True},
    },
    'logistic_regression': {
        'vectorizer_type': 'tfidf',
        'vectorizer_config': VECTORIZER_CONFIG['tfidf'],
        'text_cleaning': {**TEXT_CLEANING_CONFIG, 'remove_stopwords': True},
    },
    'svm': {
        'vectorizer_type': 'tfidf',
        'vectorizer_config': VECTORIZER_CONFIG['tfidf'],
        'text_cleaning': {**TEXT_CLEANING_CONFIG, 'remove_stopwords': True},
    },
    'random_forest': {
        'vectorizer_type': 'tfidf',
        'vectorizer_config': VECTORIZER_CONFIG['tfidf'],
        'text_cleaning': {**TEXT_CLEANING_CONFIG, 'remove_stopwords': True},
    },
    'xgboost': {
        'vectorizer_type': 'tfidf',
        'vectorizer_config': VECTORIZER_CONFIG['tfidf'],
        'text_cleaning': {**TEXT_CLEANING_CONFIG, 'remove_stopwords': True},
    },
    'lstm': {
        'text_cleaning': {**TEXT_CLEANING_CONFIG, 'remove_stopwords': False},
        'lstm_config': LSTM_CONFIG,
    },
    'bert': {
        'text_cleaning': {'minimal': True},  # Minimal cleaning for BERT
        'bert_config': BERT_CONFIG['bert-base-uncased'],
    },
    'distilbert': {
        'text_cleaning': {'minimal': True},  # Minimal cleaning for DistilBERT
        'bert_config': BERT_CONFIG['distilbert-base-uncased'],
    },
}

# File naming conventions
def get_preprocessed_filename(model_type: str, fold: int = 1, split: str = 'train') -> str:
    """Generate filename for preprocessed data."""
    return f"{model_type}_fold{fold}_{split}.pkl"

def get_vectorizer_filename(model_type: str, vectorizer_type: str, fold: int = 1) -> str:
    """Generate filename for saved vectorizer."""
    return f"{model_type}_{vectorizer_type}_fold{fold}.pkl"

def get_tokenizer_filename(model_type: str, fold: int = 1) -> str:
    """Generate filename for saved tokenizer."""
    return f"{model_type}_tokenizer_fold{fold}.pkl"

# Create directories if they don't exist
PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)

