"""TF-IDF + XGBoost trainers."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None  # Handle missing dependency gracefully

from src.features.tfidf import create_tfidf_vectorizer


def train_multilabel_tfidf_xgboost(
    X_train: List[str],
    y_train: np.ndarray,
    label_cols: List[str],
    vectorizer_params: Optional[Dict] = None,
    xgb_params: Optional[Dict] = None,
) -> Tuple[Union[TfidfVectorizer, FeatureUnion], Dict[str, XGBClassifier]]:
    """Train TF-IDF + XGBoost classifiers for each label (Binary Relevance)."""

    if XGBClassifier is None:
        raise ImportError(
            "XGBoost not installed. Please install with: pip install xgboost"
        )

    if vectorizer_params is None:
        vectorizer_params = {}
    if xgb_params is None:
        xgb_params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_jobs": -1,
            "random_state": 42,
            "scale_pos_weight": 1.0,  # Often tuned for imbalance
        }

    tfidf = create_tfidf_vectorizer(**vectorizer_params)
    X_train_vec = tfidf.fit_transform(X_train)

    models: Dict[str, XGBClassifier] = {}
    for idx, label in enumerate(label_cols):
        # Allow per-label overriding of scale_pos_weight if passed as a list/dict
        # For now, we assume global params or user handles it.
        clf = XGBClassifier(**xgb_params)
        clf.fit(X_train_vec, y_train[:, idx])
        models[label] = clf

    return tfidf, models


__all__ = ["train_multilabel_tfidf_xgboost"]

