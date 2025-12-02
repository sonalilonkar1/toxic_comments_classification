"""Reusable TF-IDF + logistic multi-label training pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import joblib
import numpy as np
import pandas as pd

from src.data.dataset import load_fold_frames
from src.data.preprocess import rich_normalize, toy_normalize
from src.features.tfidf import (
    oversample_buckets,
    train_multilabel_tfidf_linear_svm,
    train_multilabel_tfidf_logistic,
)
from src.utils.metrics import (
    compute_fairness_slices,
    compute_multilabel_metrics,
    probs_to_preds,
)


DEFAULT_LABELS: List[str] = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

DEFAULT_VECTORIZER: Dict[str, object] = {
    "max_features": 50000,
    "ngram_range": (1, 2),
    "min_df": 5,
    "max_df": 0.95,
    "lowercase": True,
    "strip_accents": "unicode",
}

DEFAULT_MODEL: Dict[str, object] = {
    "max_iter": 400,
    "class_weight": "balanced",
    "solver": "liblinear",
    "C": 1.0,
}

DEFAULT_SVM: Dict[str, object] = {
    "C": 1.0,
    "class_weight": "balanced",
    "max_iter": 2000,
}

DEFAULT_SVM_CALIBRATION: Dict[str, object] = {
    "method": "sigmoid",
    "cv": 3,
}

Normalizer = Optional[Callable[[str], str]]

NORMALIZERS: Dict[str, Normalizer] = {
    "raw": None,
    "toy": toy_normalize,
    "rich": rich_normalize,
}


@dataclass
class TrainConfig:
    """Configuration payload for the training pipeline."""

    data_path: Path = Path("data/raw/train.csv")
    splits_dir: Path = Path("data/splits")
    output_dir: Path = Path("experiments/tfidf_logreg")
    fold: Optional[str] = None
    label_cols: Optional[List[str]] = None
    text_col: str = "comment_text"
    normalization: str = "toy"
    threshold: float = 0.5
    fairness_min_support: int = 50
    seed: int = 42
    bucket_col: Optional[str] = None
    bucket_multipliers: Optional[Dict[str, int]] = None
    model_type: str = "logistic"
    vectorizer_params: Dict[str, object] = field(default_factory=lambda: DEFAULT_VECTORIZER.copy())
    model_params: Dict[str, object] = field(default_factory=lambda: DEFAULT_MODEL.copy())
    svm_params: Dict[str, object] = field(default_factory=lambda: DEFAULT_SVM.copy())
    svm_calibration_params: Dict[str, object] = field(default_factory=lambda: DEFAULT_SVM_CALIBRATION.copy())

    def __post_init__(self) -> None:
        self.data_path = Path(self.data_path)
        self.splits_dir = Path(self.splits_dir)
        self.output_dir = Path(self.output_dir)

    def resolve_label_cols(self, df: pd.DataFrame) -> List[str]:
        if self.label_cols:
            return self.label_cols
        inferred = [col for col in DEFAULT_LABELS if col in df.columns]
        if not inferred:
            raise ValueError("No label columns found in dataframe; please pass label_cols explicitly.")
        self.label_cols = inferred
        return inferred


def run_training_pipeline(config: TrainConfig) -> Dict[str, Dict[str, object]]:
    """Execute the TF-IDF + logistic baseline across one or more folds.

    Returns a dictionary keyed by fold containing artifact locations and metrics.
    """

    base_df, fold_frames, identity_cols, _ = load_fold_frames(
        seed=config.seed,
        data_path=config.data_path,
        splits_dir=config.splits_dir,
    )
    label_cols = config.resolve_label_cols(base_df)

    target_folds: Iterable[str]
    if config.fold:
        if config.fold not in fold_frames:
            raise ValueError(f"Fold '{config.fold}' not found in splits directory.")
        target_folds = [config.fold]
    else:
        target_folds = sorted(fold_frames.keys())

    model_type = config.model_type.lower()
    if model_type not in {"logistic", "svm"}:
        raise ValueError("model_type must be one of {'logistic', 'svm'}")

    results: Dict[str, Dict[str, object]] = {}
    for fold_name in target_folds:
        fold_dir = _prepare_fold_dir(config.output_dir, fold_name, config)
        fold_metrics = _train_single_fold(
            fold_name=fold_name,
            fold_splits=fold_frames[fold_name],
            identity_cols=identity_cols,
            label_cols=label_cols,
            fold_dir=fold_dir,
            config=config,
        )
        results[fold_name] = fold_metrics

    summary_path = config.output_dir / "summary_metrics.json"
    summary_payload = {
        fold: metrics["overall_metrics"] for fold, metrics in results.items()
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)

    return results


def _train_single_fold(
    fold_name: str,
    fold_splits: Dict[str, pd.DataFrame],
    identity_cols: List[str],
    label_cols: List[str],
    fold_dir: Path,
    config: TrainConfig,
) -> Dict[str, object]:
    """Train/evaluate a single fold and persist artifacts."""

    normalizer = _resolve_normalizer(config.normalization)
    text_col = config.text_col

    train_df = fold_splits["train"].reset_index(drop=True)
    train_df = _maybe_apply_bucket_augmentation(train_df, config)
    dev_df = fold_splits["dev"].reset_index(drop=True)
    test_df = fold_splits["test"].reset_index(drop=True)

    X_train = _normalize_series(train_df[text_col], normalizer)
    X_dev = _normalize_series(dev_df[text_col], normalizer)
    X_test = _normalize_series(test_df[text_col], normalizer)

    y_train = train_df[label_cols].values.astype(int)
    y_test = test_df[label_cols].values.astype(int)

    if config.model_type == "svm":
        tfidf, label_models = train_multilabel_tfidf_linear_svm(
            X_train.tolist(),
            y_train,
            label_cols,
            vectorizer_params=config.vectorizer_params,
            svm_params=config.svm_params,
            calibration_params=config.svm_calibration_params,
        )
    else:
        tfidf, label_models = train_multilabel_tfidf_logistic(
            X_train.tolist(),
            y_train,
            label_cols,
            vectorizer_params=config.vectorizer_params,
            model_params=config.model_params,
        )

    X_test_vec = tfidf.transform(X_test.tolist())
    test_probs = {
        label: model.predict_proba(X_test_vec)[:, 1]
        for label, model in label_models.items()
    }
    y_test_pred = probs_to_preds(test_probs, threshold=config.threshold)

    overall_metrics, per_label_df = compute_multilabel_metrics(y_test, y_test_pred, label_cols)
    fairness_df = compute_fairness_slices(
        test_df,
        y_test,
        y_test_pred,
        label_cols,
        identity_cols,
        min_support=config.fairness_min_support,
    )

    _persist_artifacts(
        fold_dir=fold_dir,
        fold_name=fold_name,
        config=config,
        label_cols=label_cols,
        overall_metrics=overall_metrics,
        per_label_df=per_label_df,
        fairness_df=fairness_df,
        test_df=test_df,
        test_probs=test_probs,
        y_test_pred=y_test_pred,
        tfidf=tfidf,
        label_models=label_models,
        text_col=text_col,
        model_type=config.model_type,
    )

    return {
        "overall_metrics": overall_metrics,
        "per_label_path": fold_dir / "per_label_metrics.csv",
        "config_path": fold_dir / "config.json",
    }


def _prepare_fold_dir(base_dir: Path, fold_name: str, config: TrainConfig) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_name = f"{fold_name}-seed{config.seed}-norm{config.normalization}-{timestamp}"
    fold_dir = base_dir / dir_name
    fold_dir.mkdir(parents=True, exist_ok=True)
    return fold_dir


def _resolve_normalizer(name: str) -> Normalizer:
    if name in (None, "raw"):
        return None
    if name not in NORMALIZERS:
        raise ValueError(f"Unknown normalization strategy '{name}'. Choices: {list(NORMALIZERS)}")
    return NORMALIZERS[name]


def _normalize_series(series: pd.Series, normalizer: Normalizer) -> pd.Series:
    series = series.fillna("").astype(str)
    if normalizer is None:
        return series
    return series.apply(normalizer)


def _persist_artifacts(
    fold_dir: Path,
    fold_name: str,
    config: TrainConfig,
    label_cols: List[str],
    overall_metrics: Dict[str, float],
    per_label_df: pd.DataFrame,
    fairness_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_probs: Dict[str, np.ndarray],
    y_test_pred: np.ndarray,
    tfidf,
    label_models,
    text_col: str,
    model_type: str,
) -> None:
    overall_path = fold_dir / "overall_metrics.json"
    per_label_path = fold_dir / "per_label_metrics.csv"
    fairness_path = fold_dir / "fairness_slices.csv"
    preds_path = fold_dir / "test_predictions.csv"
    config_path = fold_dir / "config.json"
    model_dir = fold_dir / "models"
    model_dir.mkdir(exist_ok=True)

    with open(overall_path, "w", encoding="utf-8") as handle:
        json.dump(overall_metrics, handle, indent=2)

    per_label_df.to_csv(per_label_path, index=False)

    if not fairness_df.empty:
        fairness_df.to_csv(fairness_path, index=False)

    preds_df = _build_predictions_frame(test_df, label_cols, test_probs, y_test_pred, text_col)
    preds_df.to_csv(preds_path, index=False)

    payload = asdict(config)
    payload.update({
        "data_path": str(config.data_path),
        "splits_dir": str(config.splits_dir),
        "output_dir": str(config.output_dir),
        "label_cols": label_cols,
        "active_fold": fold_name,
        "model_type": model_type,
    })
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    joblib.dump(tfidf, model_dir / "tfidf.joblib")
    for label, model in label_models.items():
        joblib.dump(model, model_dir / f"{label}.joblib")


def _build_predictions_frame(
    test_df: pd.DataFrame,
    label_cols: List[str],
    test_probs: Dict[str, np.ndarray],
    y_test_pred: np.ndarray,
    text_col: str,
) -> pd.DataFrame:
    payload = {}
    if "id" in test_df.columns:
        payload["id"] = test_df["id"].tolist()
    payload["row_index"] = test_df.index.tolist()
    text_key = text_col if text_col in test_df.columns else "comment_text"
    payload[text_key] = (
        test_df.get(text_key, pd.Series(dtype=str))
        .fillna("")
        .astype(str)
        .tolist()
    )

    for idx, label in enumerate(label_cols):
        payload[f"{label}_prob"] = test_probs[label]
        payload[f"{label}_pred"] = y_test_pred[:, idx]

    return pd.DataFrame(payload)


def _maybe_apply_bucket_augmentation(train_df: pd.DataFrame, config: TrainConfig) -> pd.DataFrame:
    """Optionally oversample training rows based on bucket multipliers."""

    if not config.bucket_multipliers:
        return train_df

    if not config.bucket_col:
        raise ValueError("bucket_col must be specified when bucket_multipliers are provided.")
    if config.bucket_col not in train_df.columns:
        print(
            f"[bucket-oversample] Column '{config.bucket_col}' missing; skipping bucket augmentation."
        )
        return train_df

    prepared = train_df.copy()
    prepared[config.bucket_col] = prepared[config.bucket_col].apply(_ensure_bucket_list)
    return oversample_buckets(prepared, config.bucket_col, config.bucket_multipliers)


def _ensure_bucket_list(raw_value) -> List[str]:
    """Convert serialized bucket tags to a list for downstream filtering."""

    if isinstance(raw_value, list):
        return raw_value
    if raw_value is None or (isinstance(raw_value, float) and np.isnan(raw_value)):
        return []
    if isinstance(raw_value, str):
        value = raw_value.strip()
        if not value:
            return []
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
        return [token for token in (token.strip() for token in value.split("|")) if token]
    return [str(raw_value)]