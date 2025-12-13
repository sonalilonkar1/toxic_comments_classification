"""LSTM-specific preprocessing: tokenization, sequence padding, and embedding loading."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TF_AVAILABLE = True
except ImportError:
    try:
        # Try tf_keras (for compatibility with transformers)
        from tf_keras.preprocessing.text import Tokenizer
        from tf_keras.preprocessing.sequence import pad_sequences
        TF_AVAILABLE = True
    except ImportError:
        try:
            from keras.preprocessing.text import Tokenizer
            from keras.preprocessing.sequence import pad_sequences
            TF_AVAILABLE = True
        except ImportError:
            TF_AVAILABLE = False
            Tokenizer = None
            pad_sequences = None


class LSTMPreprocessor:
    """Preprocessor for LSTM models with tokenization and sequence padding.
    
    This class handles:
    - Building vocabulary from training text
    - Converting text to integer sequences
    - Padding/truncating sequences to fixed length
    - Saving/loading preprocessor state
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        max_length: Optional[int] = None,
        oov_token: str = '<OOV>',
        padding: str = 'post',
        truncating: str = 'post',
        filters: str = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
        lower: bool = True,
    ):
        """
        Initialize LSTM preprocessor.
        
        Args:
            vocab_size: Maximum vocabulary size (most frequent words kept)
            max_length: Maximum sequence length (None to auto-detect from data)
            oov_token: Token for out-of-vocabulary words
            padding: Padding type ('pre' or 'post')
            truncating: Truncation type ('pre' or 'post')
            filters: Characters to filter out during tokenization
            lower: Whether to lowercase text before tokenization
        """
        if not TF_AVAILABLE:
            raise ImportError(
                "TensorFlow or Keras is required for LSTM preprocessing. "
                "Install with: pip install tensorflow"
            )
        
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.oov_token = oov_token
        self.padding = padding
        self.truncating = truncating
        self.filters = filters
        self.lower = lower
        
        self.tokenizer = Tokenizer(
            num_words=vocab_size,
            oov_token=oov_token,
            filters=filters,
            lower=lower,
            split=' ',
        )
        self.is_fitted = False
        self.actual_max_length: Optional[int] = None
    
    def fit(self, texts: List[str]) -> LSTMPreprocessor:
        """
        Fit tokenizer on training texts.
        
        Args:
            texts: List of text strings to fit on
            
        Returns:
            Self for method chaining
        """
        if not texts:
            raise ValueError("Cannot fit on empty text list")
        
        self.tokenizer.fit_on_texts(texts)
        self.is_fitted = True
        
        # Determine max_length if not specified
        if self.max_length is None:
            sequences = self.tokenizer.texts_to_sequences(texts)
            self.actual_max_length = max(len(seq) for seq in sequences) if sequences else 0
        else:
            self.actual_max_length = self.max_length
        
        return self
    
    def transform(self, texts: List[str], max_length: Optional[int] = None) -> np.ndarray:
        """
        Transform texts to padded sequences.
        
        Args:
            texts: List of text strings to transform
            max_length: Override max_length for this transform (uses fitted value if None)
        
        Returns:
            Padded sequences as numpy array of shape (n_samples, sequence_length)
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform. Call fit() first.")
        
        sequences = self.tokenizer.texts_to_sequences(texts)
        length = max_length if max_length is not None else self.actual_max_length
        
        if length is None:
            raise ValueError("max_length must be set (either in __init__ or transform)")
        
        padded = pad_sequences(
            sequences,
            maxlen=length,
            padding=self.padding,
            truncating=self.truncating,
        )
        return padded
    
    def fit_transform(self, texts: List[str], max_length: Optional[int] = None) -> np.ndarray:
        """
        Fit and transform texts in one step.
        
        Args:
            texts: List of text strings
            max_length: Override max_length for this transform
        
        Returns:
            Padded sequences as numpy array
        """
        self.fit(texts)
        return self.transform(texts, max_length=max_length)
    
    def get_word_index(self) -> Dict[str, int]:
        """Get word to index mapping."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before getting word index")
        return self.tokenizer.word_index
    
    def get_index_word(self) -> Dict[int, str]:
        """Get index to word mapping."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before getting index word")
        return self.tokenizer.index_word
    
    def get_vocab_size(self) -> int:
        """
        Get actual vocabulary size.
        
        Returns:
            Actual vocab size (may be less than vocab_size if fewer unique words)
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before getting vocab size")
        # +1 for OOV token, +1 for padding (index 0)
        return min(self.vocab_size, len(self.tokenizer.word_index) + 2)
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save preprocessor to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'tokenizer': self.tokenizer,
            'vocab_size': self.vocab_size,
            'max_length': self.max_length,
            'actual_max_length': self.actual_max_length,
            'oov_token': self.oov_token,
            'padding': self.padding,
            'truncating': self.truncating,
            'filters': self.filters,
            'lower': self.lower,
            'is_fitted': self.is_fitted,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> LSTMPreprocessor:
        """Load preprocessor from disk."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Preprocessor file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls(
            vocab_size=data['vocab_size'],
            max_length=data['max_length'],
            oov_token=data['oov_token'],
            padding=data['padding'],
            truncating=data['truncating'],
            filters=data.get('filters', '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'),
            lower=data.get('lower', True),
        )
        instance.tokenizer = data['tokenizer']
        instance.actual_max_length = data['actual_max_length']
        instance.is_fitted = data['is_fitted']
        
        return instance


def load_glove_embeddings(
    filepath: Union[str, Path],
    word_index: Dict[str, int],
    embedding_dim: int = 100,
    vocab_size: Optional[int] = None,
) -> np.ndarray:
    """
    Load pre-trained GloVe embeddings and create embedding matrix.
    
    Args:
        filepath: Path to GloVe embeddings file (e.g., glove.6B.100d.txt)
        word_index: Word to index mapping from tokenizer
        embedding_dim: Dimension of embeddings (100, 200, or 300)
        vocab_size: Maximum vocabulary size to use (if None, uses len(word_index) + 2)
    
    Returns:
        Embedding matrix of shape (vocab_size, embedding_dim)
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"GloVe file not found: {filepath}")
    
    if vocab_size is None:
        vocab_size = len(word_index) + 2  # +1 for OOV, +1 for padding (index 0)
    else:
        # Ensure vocab_size doesn't exceed actual vocabulary
        vocab_size = min(vocab_size, len(word_index) + 2)
    embeddings_index: Dict[str, np.ndarray] = {}
    
    print(f"Loading GloVe embeddings from {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            if len(values) < embedding_dim + 1:
                continue
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            if len(coefs) == embedding_dim:
                embeddings_index[word] = coefs
    
    # Create embedding matrix
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    found_words = 0
    
    for word, i in word_index.items():
        if i >= vocab_size:
            continue
        embedding_vector = embeddings_index.get(word.lower())  # GloVe is lowercase
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            found_words += 1
    
    print(f"Found embeddings for {found_words}/{vocab_size - 2} words in vocabulary")
    return embedding_matrix


def load_word2vec_embeddings(
    filepath: Union[str, Path],
    word_index: Dict[str, int],
    binary: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Load pre-trained Word2Vec embeddings and create embedding matrix.
    
    Args:
        filepath: Path to Word2Vec embeddings file
        word_index: Word to index mapping from tokenizer
        binary: Whether the file is in binary format
    
    Returns:
        Tuple of (embedding_matrix, embedding_dim)
    """
    try:
        from gensim.models import KeyedVectors
    except ImportError:
        raise ImportError(
            "gensim is required for Word2Vec embeddings. Install with: pip install gensim"
        )
    
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Word2Vec file not found: {filepath}")
    
    print(f"Loading Word2Vec embeddings from {filepath}...")
    model = KeyedVectors.load_word2vec_format(str(filepath), binary=binary)
    
    vocab_size = len(word_index) + 2  # +1 for OOV, +1 for padding
    embedding_dim = model.vector_size
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    found_words = 0
    
    for word, i in word_index.items():
        if i >= vocab_size:
            continue
        if word.lower() in model:
            embedding_matrix[i] = model[word.lower()]
            found_words += 1
    
    print(f"Found embeddings for {found_words}/{vocab_size - 2} words in vocabulary")
    return embedding_matrix, embedding_dim


def preprocess_texts_for_lstm(
    texts: List[str],
    normalizer: Optional[Callable[[str], str]] = None,
    vocab_size: int = 10000,
    max_length: Optional[int] = None,
    oov_token: str = '<OOV>',
    padding: str = 'post',
    truncating: str = 'post',
) -> Tuple[LSTMPreprocessor, np.ndarray]:
    """
    Complete preprocessing pipeline for LSTM: normalize, tokenize, and pad.
    
    This function combines text normalization with LSTM tokenization/padding.
    It's a convenience wrapper that handles the full preprocessing flow.
    
    Args:
        texts: List of raw text strings
        normalizer: Optional normalization function (e.g., from build_normalizer)
                   If None, only basic lowercasing is applied
        vocab_size: Maximum vocabulary size
        max_length: Maximum sequence length (None to auto-detect)
        oov_token: Token for out-of-vocabulary words
        padding: Padding type ('pre' or 'post')
        truncating: Truncation type ('pre' or 'post')
    
    Returns:
        Tuple of (fitted LSTMPreprocessor, padded_sequences)
    """
    # Step 1: Normalize texts
    if normalizer:
        normalized_texts = [normalizer(str(text)) for text in texts]
    else:
        # Basic lowercasing if no normalizer provided
        normalized_texts = [str(text).lower() if text else "" for text in texts]
    
    # Step 2: Tokenize and pad
    preprocessor = LSTMPreprocessor(
        vocab_size=vocab_size,
        max_length=max_length,
        oov_token=oov_token,
        padding=padding,
        truncating=truncating,
    )
    padded_sequences = preprocessor.fit_transform(normalized_texts)
    
    return preprocessor, padded_sequences


__all__ = [
    "LSTMPreprocessor",
    "load_glove_embeddings",
    "load_word2vec_embeddings",
    "preprocess_texts_for_lstm",
]

