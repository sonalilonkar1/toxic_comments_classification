"""TF-IDF + LogisticRegression trainers."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression

from src.features.tfidf import create_tfidf_vectorizer


def train_multilabel_tfidf_logistic(
    X_train: List[str],
    y_train: np.ndarray,
    label_cols: List[str],
    vectorizer_params: Optional[Dict] = None,
    model_params: Optional[Dict] = None,
    calibration_params: Optional[Dict] = None,
) -> Tuple[Union[TfidfVectorizer, FeatureUnion], Dict[str, Union[LogisticRegression, CalibratedClassifierCV]]]:
    """Train TF-IDF + LogisticRegression models for each label (optionally calibrated)."""

    if vectorizer_params is None:
        vectorizer_params = {}
    if model_params is None:
        model_params = {
            "max_iter": 400,
            "class_weight": "balanced",
            "solver": "liblinear",
            "C": 1.0,
        }
    
    # Check if calibration is requested
    use_calibration = calibration_params is not None and calibration_params.get("method") is not None

    tfidf = create_tfidf_vectorizer(**vectorizer_params)
    X_train_vec = tfidf.fit_transform(X_train)

    models: Dict[str, Union[LogisticRegression, CalibratedClassifierCV]] = {}
    for idx, label in enumerate(label_cols):
        clf = LogisticRegression(**model_params)
        
        if use_calibration:
            calibrated = CalibratedClassifierCV(
                estimator=clf,
                method=calibration_params.get("method", "sigmoid"),
                cv=calibration_params.get("cv", 3),
                n_jobs=calibration_params.get("n_jobs"),
            )
            calibrated.fit(X_train_vec, y_train[:, idx])
            models[label] = calibrated
        else:
            clf.fit(X_train_vec, y_train[:, idx])
            models[label] = clf

    return tfidf, models


__all__ = ["train_multilabel_tfidf_logistic"]
