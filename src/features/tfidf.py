"""TF-IDF feature extraction and traditional model training."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


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


def train_multilabel_tfidf_logistic(
    X_train: List[str],
    y_train: np.ndarray,
    label_cols: List[str],
    vectorizer_params: Optional[Dict] = None,
    model_params: Optional[Dict] = None,
) -> tuple[TfidfVectorizer, Dict[str, LogisticRegression]]:
    """Train TF-IDF + LogisticRegression models for each label.

    Args:
        X_train: Training texts
        y_train: Multi-label targets (n_samples, n_labels)
        label_cols: Label column names
        vectorizer_params: Params for TfidfVectorizer
        model_params: Params for LogisticRegression

    Returns:
        Tuple of (fitted vectorizer, dict of label -> fitted model)
    """
    if vectorizer_params is None:
        vectorizer_params = {}
    if model_params is None:
        model_params = {"max_iter": 400, "class_weight": "balanced", "solver": "liblinear", "C": 1.0}

    # Fit vectorizer on all training text
    tfidf = create_tfidf_vectorizer(**vectorizer_params)
    X_train_vec = tfidf.fit_transform(X_train)

    # Train one model per label
    models = {}
    for idx, label in enumerate(label_cols):
        clf = LogisticRegression(**model_params)
        clf.fit(X_train_vec, y_train[:, idx])
        models[label] = clf

    return tfidf, models


def train_multilabel_tfidf_linear_svm(
    X_train: List[str],
    y_train: np.ndarray,
    label_cols: List[str],
    vectorizer_params: Optional[Dict] = None,
    svm_params: Optional[Dict] = None,
    calibration_params: Optional[Dict] = None,
) -> tuple[TfidfVectorizer, Dict[str, CalibratedClassifierCV]]:
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