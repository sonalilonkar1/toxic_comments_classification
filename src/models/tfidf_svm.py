"""TF-IDF + Linear SVM trainers with calibration."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

from src.features.tfidf import create_tfidf_vectorizer


def train_multilabel_tfidf_linear_svm(
    X_train: List[str],
    y_train: np.ndarray,
    label_cols: List[str],
    vectorizer_params: Optional[Dict] = None,
    svm_params: Optional[Dict] = None,
    calibration_params: Optional[Dict] = None,
) -> Tuple[Union[TfidfVectorizer, FeatureUnion], Dict[str, CalibratedClassifierCV]]:
    """Train TF-IDF + LinearSVC models per label with probability calibration."""

    if vectorizer_params is None:
        vectorizer_params = {}
    if svm_params is None:
        svm_params = {
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 2000,
        }
    if calibration_params is None:
        calibration_params = {"method": "sigmoid", "cv": 3}

    tfidf = create_tfidf_vectorizer(**vectorizer_params)
    X_train_vec = tfidf.fit_transform(X_train)

    models: Dict[str, CalibratedClassifierCV] = {}
    for idx, label in enumerate(label_cols):
        base_svm = LinearSVC(**svm_params)
        calibrated = CalibratedClassifierCV(
            estimator=base_svm,
            method=calibration_params.get("method", "sigmoid"),
            cv=calibration_params.get("cv", 3),
            n_jobs=calibration_params.get("n_jobs"),
        )
        calibrated.fit(X_train_vec, y_train[:, idx])
        models[label] = calibrated

    return tfidf, models


__all__ = ["train_multilabel_tfidf_linear_svm"]
