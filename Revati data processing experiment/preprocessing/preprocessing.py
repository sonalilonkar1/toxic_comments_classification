"""Unified preprocessing interface for all model types."""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .config import (
    ROOT,
    RAW_DATA_DIR,
    SPLITS_DIR,
    PREPROCESSED_DIR,
    LABEL_COLUMNS,
    DEFAULT_MODEL_CONFIGS,
    get_preprocessed_filename,
    get_vectorizer_filename,
    get_tokenizer_filename,
)
from .text_cleaning import clean_text, clean_text_minimal
from .vectorizers import get_vectorizer, TextVectorizer
from .lstm_preprocessing import LSTMPreprocessor
from .bert_preprocessing import BERTPreprocessor, get_bert_preprocessor


class PreprocessingPipeline:
    """Unified preprocessing pipeline for all model types."""
    
    def __init__(self, model_type: str, fold: int = 1, config: Optional[Dict] = None):
        """
        Initialize preprocessing pipeline.
        
        Args:
            model_type: Type of model ('naive_bayes', 'logistic_regression', 'svm',
                       'random_forest', 'xgboost', 'lstm', 'bert', 'distilbert')
            fold: Fold number (default: 1)
            config: Optional custom configuration dict
        """
        self.model_type = model_type
        self.fold = fold
        
        # Get configuration
        if config is None:
            if model_type not in DEFAULT_MODEL_CONFIGS:
                raise ValueError(
                    f"Unknown model type: {model_type}. "
                    f"Available: {list(DEFAULT_MODEL_CONFIGS.keys())}"
                )
            self.config = DEFAULT_MODEL_CONFIGS[model_type].copy()
        else:
            self.config = config
        
        # Initialize preprocessor based on model type
        self.preprocessor = None
        self.vectorizer = None
        self._initialize_preprocessor()
    
    def _initialize_preprocessor(self):
        """Initialize the appropriate preprocessor based on model type."""
        if self.model_type in ['naive_bayes', 'logistic_regression', 'svm', 'random_forest', 'xgboost']:
            # Traditional ML models use vectorizers
            vectorizer_type = self.config.get('vectorizer_type', 'tfidf')
            vectorizer_config = self.config.get('vectorizer_config', {})
            self.vectorizer = get_vectorizer(
                vectorizer_type=vectorizer_type,
                **vectorizer_config
            )
        elif self.model_type == 'lstm':
            # LSTM uses tokenizer
            lstm_config = self.config.get('lstm_config', {})
            self.preprocessor = LSTMPreprocessor(**lstm_config)
        elif self.model_type in ['bert', 'distilbert']:
            # BERT/DistilBERT uses HuggingFace tokenizer
            bert_config = self.config.get('bert_config', {})
            self.preprocessor = get_bert_preprocessor(**bert_config)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def load_data(self, split: str = 'train') -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Load data for a specific split.
        
        Args:
            split: Data split ('train', 'dev', 'test')
        
        Returns:
            Tuple of (dataframe with texts, labels array)
        """
        # Load raw data
        raw_data_path = RAW_DATA_DIR / "train.csv"
        if not raw_data_path.exists():
            raise FileNotFoundError(
                f"Raw data not found at {raw_data_path}. "
                "Run scripts/02_download_kaggle.sh first."
            )
        
        df = pd.read_csv(raw_data_path)
        
        # Load split indices
        split_file = SPLITS_DIR / f"fold{self.fold}.json"
        if not split_file.exists():
            raise FileNotFoundError(
                f"Split file not found at {split_file}. "
                "Run src.cli.make_splits first."
            )
        
        with open(split_file, 'r') as f:
            splits = json.load(f)
        
        if split not in splits:
            raise ValueError(f"Split '{split}' not found in {split_file}")
        
        indices = splits[split]
        df_split = df.iloc[indices].copy()
        
        # Extract texts and labels
        texts = df_split['comment_text'].values
        labels = df_split[LABEL_COLUMNS].values.astype(np.float32)
        
        return df_split, labels
    
    def clean_texts(self, texts: Union[List[str], np.ndarray]) -> List[str]:
        """
        Clean texts based on configuration.
        
        Args:
            texts: List or array of text strings
        
        Returns:
            List of cleaned text strings
        """
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        
        cleaning_config = self.config.get('text_cleaning', {})
        
        # Check if minimal cleaning (for BERT/DistilBERT)
        if cleaning_config.get('minimal', False):
            return [clean_text_minimal(text) for text in texts]
        else:
            return [clean_text(text, **cleaning_config) for text in texts]
    
    def fit(self, texts: Optional[Union[List[str], np.ndarray]] = None):
        """
        Fit preprocessor on training data.
        
        Args:
            texts: Optional texts to fit on (if None, loads train split)
        """
        if texts is None:
            df_train, _ = self.load_data('train')
            texts = df_train['comment_text'].values
        
        # Clean texts
        cleaned_texts = self.clean_texts(texts)
        
        # Fit preprocessor
        if self.vectorizer is not None:
            self.vectorizer.fit(cleaned_texts)
        elif self.preprocessor is not None:
            self.preprocessor.fit(cleaned_texts)
        else:
            raise ValueError("No preprocessor initialized")
    
    def transform(
        self,
        texts: Optional[Union[List[str], np.ndarray]] = None,
        split: Optional[str] = None,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Transform texts to features.
        
        Args:
            texts: Optional texts to transform (if None, loads from split)
            split: Data split to load ('train', 'dev', 'test')
        
        Returns:
            Transformed features (numpy array for traditional ML/LSTM, dict for BERT)
        """
        if texts is None:
            if split is None:
                raise ValueError("Either texts or split must be provided")
            df_split, _ = self.load_data(split)
            texts = df_split['comment_text'].values
        elif isinstance(texts, np.ndarray):
            texts = texts.tolist()
        
        # Clean texts
        cleaned_texts = self.clean_texts(texts)
        
        # Transform
        if self.vectorizer is not None:
            return self.vectorizer.transform(cleaned_texts)
        elif self.preprocessor is not None:
            if isinstance(self.preprocessor, LSTMPreprocessor):
                return self.preprocessor.transform(cleaned_texts)
            else:  # BERT/DistilBERT
                return self.preprocessor.transform(cleaned_texts)
        else:
            raise ValueError("No preprocessor initialized")
    
    def fit_transform(
        self,
        texts: Optional[Union[List[str], np.ndarray]] = None,
        split: Optional[str] = None,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Fit and transform texts.
        
        Args:
            texts: Optional texts to fit/transform (if None, loads train split)
            split: Data split to load ('train', 'dev', 'test')
        
        Returns:
            Transformed features
        """
        if texts is None:
            if split is None:
                split = 'train'
            df_split, _ = self.load_data(split)
            texts = df_split['comment_text'].values
        
        # Clean texts
        cleaned_texts = self.clean_texts(texts)
        
        # Fit and transform
        if self.vectorizer is not None:
            return self.vectorizer.fit_transform(cleaned_texts)
        elif self.preprocessor is not None:
            return self.preprocessor.fit_transform(cleaned_texts)
        else:
            raise ValueError("No preprocessor initialized")
    
    def preprocess_split(
        self,
        split: str,
        save: bool = True,
    ) -> Tuple[Union[np.ndarray, Dict[str, np.ndarray]], np.ndarray]:
        """
        Preprocess a data split and optionally save it.
        
        Args:
            split: Data split ('train', 'dev', 'test')
            save: Whether to save preprocessed data
        
        Returns:
            Tuple of (features, labels)
        """
        # Load data
        df_split, labels = self.load_data(split)
        
        # Fit on train split only
        if split == 'train':
            self.fit()
        
        # Transform
        features = self.transform(split=split)
        
        # Save if requested
        if save:
            self.save_preprocessed(split, features, labels)
        
        return features, labels
    
    def save_preprocessed(
        self,
        split: str,
        features: Union[np.ndarray, Dict[str, np.ndarray]],
        labels: np.ndarray,
    ):
        """Save preprocessed data to disk."""
        filename = get_preprocessed_filename(self.model_type, self.fold, split)
        filepath = PREPROCESSED_DIR / filename
        
        data = {
            'features': features,
            'labels': labels,
            'model_type': self.model_type,
            'fold': self.fold,
            'split': split,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Saved preprocessed data to {filepath}")
    
    def save_preprocessor(self):
        """Save preprocessor/vectorizer to disk."""
        if self.vectorizer is not None:
            vectorizer_type = self.config.get('vectorizer_type', 'tfidf')
            filename = get_vectorizer_filename(self.model_type, vectorizer_type, self.fold)
            filepath = PREPROCESSED_DIR / filename
            self.vectorizer.save(filepath)
            print(f"Saved vectorizer to {filepath}")
        elif self.preprocessor is not None:
            filename = get_tokenizer_filename(self.model_type, self.fold)
            filepath = PREPROCESSED_DIR / filename
            self.preprocessor.save(filepath)
            print(f"Saved tokenizer to {filepath}")
    
    def load_preprocessed(self, split: str) -> Tuple[Union[np.ndarray, Dict[str, np.ndarray]], np.ndarray]:
        """
        Load preprocessed data from disk.
        
        Args:
            split: Data split ('train', 'dev', 'test')
        
        Returns:
            Tuple of (features, labels)
        """
        filename = get_preprocessed_filename(self.model_type, self.fold, split)
        filepath = PREPROCESSED_DIR / filename
        
        if not filepath.exists():
            raise FileNotFoundError(
                f"Preprocessed data not found at {filepath}. "
                "Run preprocess_split() first."
            )
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        return data['features'], data['labels']
    
    @classmethod
    def load_preprocessor(cls, model_type: str, fold: int = 1):
        """
        Load a saved preprocessor.
        
        Args:
            model_type: Type of model
            fold: Fold number
        
        Returns:
            PreprocessingPipeline instance with loaded preprocessor
        """
        instance = cls(model_type, fold=fold)
        
        if instance.vectorizer is not None:
            vectorizer_type = instance.config.get('vectorizer_type', 'tfidf')
            filename = get_vectorizer_filename(model_type, vectorizer_type, fold)
            filepath = PREPROCESSED_DIR / filename
            if filepath.exists():
                instance.vectorizer = TextVectorizer.load(filepath)
        elif instance.preprocessor is not None:
            filename = get_tokenizer_filename(model_type, fold)
            filepath = PREPROCESSED_DIR / filename
            if filepath.exists():
                if isinstance(instance.preprocessor, LSTMPreprocessor):
                    instance.preprocessor = LSTMPreprocessor.load(filepath)
                else:  # BERT/DistilBERT
                    instance.preprocessor = BERTPreprocessor.load(filepath)
        
        return instance


def preprocess_all_splits(
    model_type: str,
    fold: int = 1,
    save: bool = True,
) -> Dict[str, Tuple[Union[np.ndarray, Dict[str, np.ndarray]], np.ndarray]]:
    """
    Preprocess all splits (train, dev, test) for a model type.
    
    Args:
        model_type: Type of model
        fold: Fold number
        save: Whether to save preprocessed data
    
    Returns:
        Dictionary mapping split names to (features, labels) tuples
    """
    pipeline = PreprocessingPipeline(model_type, fold=fold)
    
    results = {}
    for split in ['train', 'dev', 'test']:
        print(f"Preprocessing {split} split for {model_type} (fold {fold})...")
        features, labels = pipeline.preprocess_split(split, save=save)
        results[split] = (features, labels)
    
    # Save preprocessor
    if save:
        pipeline.save_preprocessor()
    
    return results

