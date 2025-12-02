"""TF-IDF feature extraction helpers and bucket-aware oversampling."""

from typing import Dict, List, Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def create_tfidf_vectorizer(
    max_features: int = 50000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 5,
    max_df: float = 0.95,
    lowercase: bool = True,
    strip_accents: str = "unicode",
) -> TfidfVectorizer:
    """Create and return a configured TF-IDF vectorizer."""
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        lowercase=lowercase,
        strip_accents=strip_accents,
    )


def oversample_buckets(
    train_df: pd.DataFrame,
    bucket_col: str,
    multipliers: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """Oversample training data based on bucket multipliers.

    Args:
        train_df: Training DataFrame with text and bucket columns
        bucket_col: Column name containing bucket tags (list of str)
        multipliers: Dict of bucket_name -> multiplier factor

    Returns:
        Augmented DataFrame with oversampled rows
    """
    parts = [train_df]
    if not multipliers:
        return train_df

    for bucket, factor in multipliers.items():
        if factor <= 1:
            continue
        bucket_rows = train_df[train_df[bucket_col].apply(lambda tags: bucket in tags)]
        if not bucket_rows.empty:
            extras = [bucket_rows] * (factor - 1)
            parts.extend(extras)

    augmented = pd.concat(parts).sample(frac=1.0, random_state=0).reset_index(drop=True)
    return augmented