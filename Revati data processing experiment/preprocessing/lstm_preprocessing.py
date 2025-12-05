"""LSTM-specific preprocessing: tokenization and word embeddings."""

import pickle
from pathlib import Path
from typing import Optional, Union, Dict, Tuple

import numpy as np

try:
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except ImportError:
    try:
        from keras.preprocessing.text import Tokenizer
        from keras.preprocessing.sequence import pad_sequences
    except ImportError:
        raise ImportError(
            "TensorFlow or Keras is required for LSTM preprocessing. "
            "Install with: pip install tensorflow"
        )


class LSTMPreprocessor:
    """Preprocessor for LSTM models with tokenization and sequence padding."""
    
    def __init__(
        self,
        vocab_size: int = 10000,
        max_length: Optional[int] = None,
        oov_token: str = '<OOV>',
        padding: str = 'post',
        truncating: str = 'post',
    ):
        """
        Initialize LSTM preprocessor.
        
        Args:
            vocab_size: Maximum vocabulary size (most frequent words)
            max_length: Maximum sequence length (None to use max in data)
            oov_token: Token for out-of-vocabulary words
            padding: Padding type ('pre' or 'post')
            truncating: Truncation type ('pre' or 'post')
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.oov_token = oov_token
        self.padding = padding
        self.truncating = truncating
        self.tokenizer = Tokenizer(
            num_words=vocab_size,
            oov_token=oov_token,
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
            lower=True,
            split=' ',
        )
        self.is_fitted = False
        self.actual_max_length: Optional[int] = None
    
    def fit(self, texts):
        """
        Fit tokenizer on training texts.
        
        Args:
            texts: List or array of text strings
        """
        self.tokenizer.fit_on_texts(texts)
        self.is_fitted = True
        
        # Determine max_length if not specified
        if self.max_length is None:
            sequences = self.tokenizer.texts_to_sequences(texts)
            self.actual_max_length = max(len(seq) for seq in sequences)
        else:
            self.actual_max_length = self.max_length
        
        return self
    
    def transform(self, texts, max_length: Optional[int] = None):
        """
        Transform texts to padded sequences.
        
        Args:
            texts: List or array of text strings
            max_length: Override max_length for this transform (uses fitted value if None)
        
        Returns:
            Padded sequences as numpy array
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        sequences = self.tokenizer.texts_to_sequences(texts)
        length = max_length if max_length is not None else self.actual_max_length
        padded = pad_sequences(
            sequences,
            maxlen=length,
            padding=self.padding,
            truncating=self.truncating,
        )
        return padded
    
    def fit_transform(self, texts, max_length: Optional[int] = None):
        """
        Fit and transform texts.
        
        Args:
            texts: List or array of text strings
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
        """Get actual vocabulary size (may be less than vocab_size if fewer unique words)."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before getting vocab size")
        return min(self.vocab_size, len(self.tokenizer.word_index) + 1)  # +1 for OOV
    
    def save(self, filepath: Union[str, Path]):
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
            'is_fitted': self.is_fitted,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]):
        """Load preprocessor from disk."""
        filepath = Path(filepath)
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls(
            vocab_size=data['vocab_size'],
            max_length=data['max_length'],
            oov_token=data['oov_token'],
            padding=data['padding'],
            truncating=data['truncating'],
        )
        instance.tokenizer = data['tokenizer']
        instance.actual_max_length = data['actual_max_length']
        instance.is_fitted = data['is_fitted']
        
        return instance


def load_glove_embeddings(
    filepath: Union[str, Path],
    word_index: Dict[str, int],
    embedding_dim: int = 100,
) -> np.ndarray:
    """
    Load pre-trained GloVe embeddings and create embedding matrix.
    
    Args:
        filepath: Path to GloVe embeddings file
        word_index: Word to index mapping from tokenizer
        embedding_dim: Dimension of embeddings (100, 200, or 300)
    
    Returns:
        Embedding matrix of shape (vocab_size, embedding_dim)
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"GloVe file not found: {filepath}")
    
    vocab_size = len(word_index) + 1  # +1 for OOV
    embeddings_index = {}
    
    print(f"Loading GloVe embeddings from {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    
    # Create embedding matrix
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    found_words = 0
    
    for word, i in word_index.items():
        if i >= vocab_size:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            found_words += 1
    
    print(f"Found embeddings for {found_words}/{vocab_size} words")
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
        raise ImportError("gensim is required for Word2Vec embeddings. Install with: pip install gensim")
    
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Word2Vec file not found: {filepath}")
    
    print(f"Loading Word2Vec embeddings from {filepath}...")
    model = KeyedVectors.load_word2vec_format(str(filepath), binary=binary)
    
    vocab_size = len(word_index) + 1  # +1 for OOV
    embedding_dim = model.vector_size
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    found_words = 0
    
    for word, i in word_index.items():
        if i >= vocab_size:
            continue
        if word in model:
            embedding_matrix[i] = model[word]
            found_words += 1
    
    print(f"Found embeddings for {found_words}/{vocab_size} words")
    return embedding_matrix, embedding_dim

