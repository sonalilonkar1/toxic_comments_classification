"""BERT/DistilBERT-specific preprocessing with HuggingFace tokenizers."""

import pickle
from pathlib import Path
from typing import Optional, Union, Dict, List

import numpy as np
from transformers import AutoTokenizer


class BERTPreprocessor:
    """Preprocessor for BERT/DistilBERT models using HuggingFace tokenizers."""
    
    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        max_length: int = 128,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        return_tensors: Optional[str] = None,
    ):
        """
        Initialize BERT preprocessor.
        
        Args:
            model_name: HuggingFace model name (e.g., 'bert-base-uncased', 'distilbert-base-uncased')
            max_length: Maximum sequence length
            padding: Padding strategy (True, 'max_length', or False)
            truncation: Whether to truncate sequences
            return_tensors: Return format ('pt' for PyTorch, 'tf' for TensorFlow, None for numpy)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.return_tensors = return_tensors
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.is_fitted = True  # Tokenizers are pre-trained, so always "fitted"
    
    def transform(
        self,
        texts: Union[str, List[str]],
        max_length: Optional[int] = None,
        padding: Optional[Union[bool, str]] = None,
        truncation: Optional[bool] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Transform texts to tokenized format.
        
        Args:
            texts: Single text string or list of text strings
            max_length: Override max_length for this transform
            padding: Override padding for this transform
            truncation: Override truncation for this transform
        
        Returns:
            Dictionary with 'input_ids' and 'attention_mask' arrays
        """
        if isinstance(texts, str):
            texts = [texts]
        
        max_len = max_length if max_length is not None else self.max_length
        pad = padding if padding is not None else self.padding
        trunc = truncation if truncation is not None else self.truncation
        
        # Tokenize
        encoded = self.tokenizer(
            texts,
            max_length=max_len,
            padding=pad,
            truncation=trunc,
            return_tensors=self.return_tensors,
        )
        
        # Convert to numpy if return_tensors is None
        if self.return_tensors is None:
            result = {
                'input_ids': encoded['input_ids'].numpy() if hasattr(encoded['input_ids'], 'numpy') else np.array(encoded['input_ids']),
                'attention_mask': encoded['attention_mask'].numpy() if hasattr(encoded['attention_mask'], 'numpy') else np.array(encoded['attention_mask']),
            }
            return result
        else:
            return encoded
    
    def fit_transform(
        self,
        texts: Union[str, List[str]],
        max_length: Optional[int] = None,
        padding: Optional[Union[bool, str]] = None,
        truncation: Optional[bool] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Fit and transform (for BERT, fit is a no-op since tokenizer is pre-trained).
        
        Args:
            texts: Single text string or list of text strings
            max_length: Override max_length for this transform
            padding: Override padding for this transform
            truncation: Override truncation for this transform
        
        Returns:
            Dictionary with 'input_ids' and 'attention_mask' arrays
        """
        return self.transform(texts, max_length=max_length, padding=padding, truncation=truncation)
    
    def decode(self, token_ids: Union[List[int], np.ndarray]) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: Token IDs to decode
        
        Returns:
            Decoded text string
        """
        if isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.tokenizer)
    
    def get_special_tokens(self) -> Dict[str, int]:
        """Get special token IDs (CLS, SEP, PAD, UNK, MASK)."""
        return {
            'cls_token_id': self.tokenizer.cls_token_id,
            'sep_token_id': self.tokenizer.sep_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
            'unk_token_id': self.tokenizer.unk_token_id,
            'mask_token_id': self.tokenizer.mask_token_id,
        }
    
    def save(self, filepath: Union[str, Path]):
        """Save preprocessor to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save tokenizer
        tokenizer_path = filepath.parent / f"{filepath.stem}_tokenizer"
        self.tokenizer.save_pretrained(str(tokenizer_path))
        
        # Save metadata
        data = {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'padding': self.padding,
            'truncation': self.truncation,
            'return_tensors': self.return_tensors,
            'tokenizer_path': str(tokenizer_path),
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]):
        """Load preprocessor from disk."""
        filepath = Path(filepath)
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Load tokenizer
        tokenizer_path = Path(data['tokenizer_path'])
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        
        instance = cls(
            model_name=data['model_name'],
            max_length=data['max_length'],
            padding=data['padding'],
            truncation=data['truncation'],
            return_tensors=data['return_tensors'],
        )
        instance.tokenizer = tokenizer
        
        return instance


def get_bert_preprocessor(
    model_name: str = 'bert-base-uncased',
    max_length: int = 128,
    padding: Union[bool, str] = True,
    truncation: bool = True,
) -> BERTPreprocessor:
    """
    Factory function to create BERT preprocessor.
    
    Args:
        model_name: HuggingFace model name
        max_length: Maximum sequence length
        padding: Padding strategy
        truncation: Whether to truncate
    
    Returns:
        BERTPreprocessor instance
    """
    return BERTPreprocessor(
        model_name=model_name,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
    )

