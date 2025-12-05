"""TF-IDF + RandomForest trainers."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from src.features.tfidf import create_tfidf_vectorizer


def train_multilabel_tfidf_random_forest(
    X_train: List[str],
    y_train: np.ndarray,
    label_cols: List[str],
    vectorizer_params: Optional[Dict] = None,
    rf_params: Optional[Dict] = None,
) -> Tuple[Union[TfidfVectorizer, FeatureUnion], Dict[str, RandomForestClassifier]]:
    """Train TF-IDF + RandomForest classifiers for each label."""

    if vectorizer_params is None:
        vectorizer_params = {}
    if rf_params is None:
        rf_params = {
            "n_estimators": 300,
            "max_depth": None,
            "max_features": "sqrt",
            "class_weight": "balanced",
            "n_jobs": -1,
            "random_state": 42,
        }

    tfidf = create_tfidf_vectorizer(**vectorizer_params)
    X_train_vec = tfidf.fit_transform(X_train)

    models: Dict[str, RandomForestClassifier] = {}
    for idx, label in enumerate(label_cols):
        clf = RandomForestClassifier(**rf_params)
        clf.fit(X_train_vec, y_train[:, idx])
        models[label] = clf

    return tfidf, models


__all__ = ["train_multilabel_tfidf_random_forest"]
