"""TF-IDF + Multinomial Naive Bayes trainers."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion

from src.features.tfidf import create_tfidf_vectorizer


def train_multilabel_tfidf_naive_bayes(
    X_train: List[str],
    y_train: np.ndarray,
    label_cols: List[str],
    vectorizer_params: Optional[Dict] = None,
    nb_params: Optional[Dict] = None,
) -> Tuple[Union[TfidfVectorizer, FeatureUnion], Dict[str, MultinomialNB]]:
    """Train TF-IDF + MultinomialNB classifiers for each label.
    
    Note: Naive Bayes requires non-negative features. Ensure vectorizer
    uses appropriate settings (TF-IDF is usually non-negative).
    """

    if vectorizer_params is None:
        vectorizer_params = {}
    if nb_params is None:
        nb_params = {
            "alpha": 1.0,
            "fit_prior": True,
        }

    # Ensure vectorizer produces positive values (standard TF-IDF does)
    tfidf = create_tfidf_vectorizer(**vectorizer_params)
    X_train_vec = tfidf.fit_transform(X_train)

    models: Dict[str, MultinomialNB] = {}
    for idx, label in enumerate(label_cols):
        clf = MultinomialNB(**nb_params)
        clf.fit(X_train_vec, y_train[:, idx])
        models[label] = clf

    return tfidf, models


__all__ = ["train_multilabel_tfidf_naive_bayes"]




