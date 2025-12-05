"""Vectorization utilities for traditional ML models."""

import pickle
from pathlib import Path
from typing import Optional, Union

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


class TextVectorizer:
    """Base class for text vectorizers with save/load functionality."""
    
    def __init__(self, vectorizer, name: str):
        self.vectorizer = vectorizer
        self.name = name
        self.is_fitted = False
    
    def fit(self, texts):
        """Fit the vectorizer on training texts."""
        self.vectorizer.fit(texts)
        self.is_fitted = True
        return self
    
    def transform(self, texts):
        """Transform texts to vectors."""
        if not self.is_fitted:
            raise ValueError(f"{self.name} vectorizer must be fitted before transform")
        return self.vectorizer.transform(texts)
    
    def fit_transform(self, texts):
        """Fit and transform texts."""
        result = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        return result
    
    def save(self, filepath: Union[str, Path]):
        """Save vectorizer to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'name': self.name,
                'is_fitted': self.is_fitted,
            }, f)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]):
        """Load vectorizer from disk."""
        filepath = Path(filepath)
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        instance = cls(data['vectorizer'], data['name'])
        instance.is_fitted = data['is_fitted']
        return instance


class TFIDFVectorizerWrapper(TextVectorizer):
    """TF-IDF vectorizer wrapper."""
    
    def __init__(
        self,
        max_features: Optional[int] = None,
        ngram_range: tuple = (1, 1),
        min_df: Union[int, float] = 1,
        max_df: Union[int, float] = 1.0,
        lowercase: bool = True,
        stop_words: Optional[Union[str, list]] = None,
        analyzer: str = 'word',
        **kwargs
    ):
        """
        Initialize TF-IDF vectorizer.
        
        Args:
            max_features: Maximum number of features to use
            ngram_range: Range of n-grams to use (e.g., (1, 2) for unigrams and bigrams)
            min_df: Minimum document frequency (int or float for proportion)
            max_df: Maximum document frequency (int or float for proportion)
            lowercase: Convert to lowercase
            stop_words: Stopwords to remove ('english' or list)
            analyzer: Type of analyzer ('word', 'char', 'char_wb')
            **kwargs: Additional arguments for TfidfVectorizer
        """
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            lowercase=lowercase,
            stop_words=stop_words,
            analyzer=analyzer,
            **kwargs
        )
        super().__init__(vectorizer, 'TF-IDF')


class CountVectorizerWrapper(TextVectorizer):
    """Count vectorizer wrapper."""
    
    def __init__(
        self,
        max_features: Optional[int] = None,
        ngram_range: tuple = (1, 1),
        min_df: Union[int, float] = 1,
        max_df: Union[int, float] = 1.0,
        lowercase: bool = True,
        stop_words: Optional[Union[str, list]] = None,
        analyzer: str = 'word',
        binary: bool = False,
        **kwargs
    ):
        """
        Initialize Count vectorizer.
        
        Args:
            max_features: Maximum number of features to use
            ngram_range: Range of n-grams to use (e.g., (1, 2) for unigrams and bigrams)
            min_df: Minimum document frequency (int or float for proportion)
            max_df: Maximum document frequency (int or float for proportion)
            lowercase: Convert to lowercase
            stop_words: Stopwords to remove ('english' or list)
            analyzer: Type of analyzer ('word', 'char', 'char_wb')
            binary: If True, use binary features (presence/absence)
            **kwargs: Additional arguments for CountVectorizer
        """
        vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            lowercase=lowercase,
            stop_words=stop_words,
            analyzer=analyzer,
            binary=binary,
            **kwargs
        )
        super().__init__(vectorizer, 'Count')


def get_vectorizer(
    vectorizer_type: str = 'tfidf',
    max_features: Optional[int] = None,
    ngram_range: tuple = (1, 1),
    min_df: Union[int, float] = 1,
    max_df: Union[int, float] = 1.0,
    lowercase: bool = True,
    stop_words: Optional[Union[str, list]] = None,
    **kwargs
) -> TextVectorizer:
    """
    Factory function to create vectorizers.
    
    Args:
        vectorizer_type: Type of vectorizer ('tfidf' or 'count')
        max_features: Maximum number of features
        ngram_range: Range of n-grams
        min_df: Minimum document frequency
        max_df: Maximum document frequency
        lowercase: Convert to lowercase
        stop_words: Stopwords to remove
        **kwargs: Additional arguments for vectorizer
    
    Returns:
        Vectorizer instance
    """
    if vectorizer_type.lower() == 'tfidf':
        return TFIDFVectorizerWrapper(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            lowercase=lowercase,
            stop_words=stop_words,
            **kwargs
        )
    elif vectorizer_type.lower() == 'count':
        return CountVectorizerWrapper(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            lowercase=lowercase,
            stop_words=stop_words,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown vectorizer type: {vectorizer_type}. Use 'tfidf' or 'count'")

