"""TF-IDF + LogisticRegression trainers."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.features.tfidf import create_tfidf_vectorizer


def train_multilabel_tfidf_logistic(
    X_train: List[str],
    y_train: np.ndarray,
    label_cols: List[str],
    vectorizer_params: Optional[Dict] = None,
    model_params: Optional[Dict] = None,
) -> Tuple[TfidfVectorizer, Dict[str, LogisticRegression]]:
    """Train TF-IDF + LogisticRegression models for each label."""

    if vectorizer_params is None:
        vectorizer_params = {}
    if model_params is None:
        model_params = {
            "max_iter": 400,
            "class_weight": "balanced",
            "solver": "liblinear",
            "C": 1.0,
        }

    tfidf = create_tfidf_vectorizer(**vectorizer_params)
    X_train_vec = tfidf.fit_transform(X_train)

    models: Dict[str, LogisticRegression] = {}
    for idx, label in enumerate(label_cols):
        clf = LogisticRegression(**model_params)
        clf.fit(X_train_vec, y_train[:, idx])
        models[label] = clf

    return tfidf, models


__all__ = ["train_multilabel_tfidf_logistic"]
