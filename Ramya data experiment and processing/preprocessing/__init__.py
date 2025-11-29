"""Feature preprocessing and vectorization utilities.

This module provides preprocessing pipelines for different model types:
- Traditional ML (Naive Bayes, Logistic Regression, SVM, Random Forest, XGBoost)
- LSTM
- BERT/DistilBERT
"""

from .preprocessing import PreprocessingPipeline, preprocess_all_splits
from .text_cleaning import clean_text, clean_text_minimal
from .vectorizers import (
    TextVectorizer,
    TFIDFVectorizerWrapper,
    CountVectorizerWrapper,
    get_vectorizer,
)
from .lstm_preprocessing import (
    LSTMPreprocessor,
    load_glove_embeddings,
    load_word2vec_embeddings,
)
from .bert_preprocessing import BERTPreprocessor, get_bert_preprocessor
from .config import (
    LABEL_COLUMNS,
    DEFAULT_MODEL_CONFIGS,
    PREPROCESSED_DIR,
)

__all__ = [
    'PreprocessingPipeline',
    'preprocess_all_splits',
    'clean_text',
    'clean_text_minimal',
    'TextVectorizer',
    'TFIDFVectorizerWrapper',
    'CountVectorizerWrapper',
    'get_vectorizer',
    'LSTMPreprocessor',
    'load_glove_embeddings',
    'load_word2vec_embeddings',
    'BERTPreprocessor',
    'get_bert_preprocessor',
    'LABEL_COLUMNS',
    'DEFAULT_MODEL_CONFIGS',
    'PREPROCESSED_DIR',
]

