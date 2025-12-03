"""Reusable TF-IDF + logistic multi-label training pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import joblib
import numpy as np
import pandas as pd

from src.data.buckets import (
    DEFAULT_BUCKET_CONFIG_PATH,
    compute_bucket_hash,
    load_bucket_config,
)
from src.data.dataset import load_fold_frames
from src.data.normalization import (
    DEFAULT_NORMALIZATION_CONFIG_PATH,
    build_normalizer,
    compute_config_hash as compute_normalization_hash,
    load_normalization_config,
)
from src.data.preprocess import rich_normalize, toy_normalize
from src.features.tfidf import oversample_buckets
from src.models.tfidf_logistic import train_multilabel_tfidf_logistic
from src.models.tfidf_random_forest import train_multilabel_tfidf_random_forest
from src.models.tfidf_svm import train_multilabel_tfidf_linear_svm
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

DEFAULT_BERT: Dict[str, object] = {
    "model_name": "bert-base-uncased",
    "max_length": 256,
    "train_batch_size": 8,
    "eval_batch_size": 8,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "num_epochs": 3.0,
    "warmup_ratio": 0.06,
    "gradient_accumulation_steps": 1,
    "fp16": False,
    "logging_steps": 50,
    "save_total_limit": 1,
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

DEFAULT_RF: Dict[str, object] = {
    "n_estimators": 400,
    "max_depth": None,
    "max_features": "sqrt",
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": 42,
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
    normalization_config: Optional[Path] = DEFAULT_NORMALIZATION_CONFIG_PATH
    normalized_cache: Optional[Path] = None
    threshold: float = 0.5
    fairness_min_support: int = 50
    seed: int = 42
    bucket_col: Optional[str] = None
    bucket_multipliers: Optional[Dict[str, int]] = None
    bucket_config: Optional[Path] = DEFAULT_BUCKET_CONFIG_PATH
    bucket_cache: Optional[Path] = None
    bucket_cache_column: str = "bucket_tags"
    model_type: str = "logistic"
    vectorizer_params: Dict[str, object] = field(default_factory=lambda: DEFAULT_VECTORIZER.copy())
    model_params: Dict[str, object] = field(default_factory=lambda: DEFAULT_MODEL.copy())
    svm_params: Dict[str, object] = field(default_factory=lambda: DEFAULT_SVM.copy())
    svm_calibration_params: Dict[str, object] = field(default_factory=lambda: DEFAULT_SVM_CALIBRATION.copy())
    rf_params: Dict[str, object] = field(default_factory=lambda: DEFAULT_RF.copy())
    bert_params: Dict[str, object] = field(default_factory=lambda: DEFAULT_BERT.copy())

    def __post_init__(self) -> None:
        self.data_path = Path(self.data_path)
        self.splits_dir = Path(self.splits_dir)
        self.output_dir = Path(self.output_dir)
        if self.normalization_config is not None:
            self.normalization_config = Path(self.normalization_config)
        if self.normalized_cache is not None:
            self.normalized_cache = Path(self.normalized_cache)
        if self.bucket_config is not None:
            self.bucket_config = Path(self.bucket_config)
        if self.bucket_cache is not None:
            self.bucket_cache = Path(self.bucket_cache)

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
    if model_type not in {"logistic", "svm", "random_forest", "bert"}:
        raise ValueError("model_type must be one of {'logistic', 'svm', 'random_forest', 'bert'}")
    config.model_type = model_type

    normalizer, normalizer_hash = _prepare_normalizer(config)
    bucket_hash = _prepare_bucket_hash(config)
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
            normalizer=normalizer,
            normalizer_hash=normalizer_hash,
            bucket_hash=bucket_hash,
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
    normalizer: Normalizer,
    normalizer_hash: Optional[str],
    bucket_hash: Optional[str],
) -> Dict[str, object]:
    """Train/evaluate a single fold and persist artifacts."""

    text_col = config.text_col

    train_df = fold_splits["train"].reset_index(drop=True)
    train_df = _maybe_apply_bucket_augmentation(train_df, config, bucket_hash)
    dev_df = fold_splits["dev"].reset_index(drop=True)
    test_df = fold_splits["test"].reset_index(drop=True)

    X_train = _resolve_text_series(train_df, text_col, normalizer, config, normalizer_hash)
    X_dev = _resolve_text_series(dev_df, text_col, normalizer, config, normalizer_hash)
    X_test = _resolve_text_series(test_df, text_col, normalizer, config, normalizer_hash)

    y_train = train_df[label_cols].values.astype(int)
    y_dev = dev_df[label_cols].values.astype(int)
    y_test = test_df[label_cols].values.astype(int)

    tfidf = None
    label_models: Optional[Dict[str, object]] = None
    extra_metadata: Optional[Dict[str, object]] = None

    if config.model_type == "bert":
        from src.models.bert_transformer import train_multilabel_bert

        bert_result = train_multilabel_bert(
            train_texts=X_train.tolist(),
            train_labels=y_train,
            dev_texts=X_dev.tolist(),
            dev_labels=y_dev,
            test_texts=X_test.tolist(),
            label_cols=label_cols,
            model_dir=fold_dir / "models" / "bert",
            params=config.bert_params,
            seed=config.seed,
        )
        test_probs = bert_result.test_probs
        extra_metadata = {
            "trainer_metrics": bert_result.trainer_metrics,
            "transformer_model_path": str(bert_result.model_path),
            "transformer_tokenizer_path": str(bert_result.tokenizer_path),
        }
    elif config.model_type == "svm":
        tfidf, label_models = train_multilabel_tfidf_linear_svm(
            X_train.tolist(),
            y_train,
            label_cols,
            vectorizer_params=config.vectorizer_params,
            svm_params=config.svm_params,
            calibration_params=config.svm_calibration_params,
        )
    elif config.model_type == "random_forest":
        tfidf, label_models = train_multilabel_tfidf_random_forest(
            X_train.tolist(),
            y_train,
            label_cols,
            vectorizer_params=config.vectorizer_params,
            rf_params=config.rf_params,
        )
    else:
        tfidf, label_models = train_multilabel_tfidf_logistic(
            X_train.tolist(),
            y_train,
            label_cols,
            vectorizer_params=config.vectorizer_params,
            model_params=config.model_params,
        )

    if config.model_type != "bert" and tfidf is not None and label_models is not None:
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
        extra_metadata=extra_metadata,
    )

    return {
        "overall_metrics": overall_metrics,
        "per_label_path": fold_dir / "per_label_metrics.csv",
        "config_path": fold_dir / "config.json",
    }


def _prepare_fold_dir(base_dir: Path, fold_name: str, config: TrainConfig) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_name = (
        f"{fold_name}-seed{config.seed}-norm{config.normalization}-"
        f"model{config.model_type}-{timestamp}"
    )
    fold_dir = base_dir / dir_name
    fold_dir.mkdir(parents=True, exist_ok=True)
    return fold_dir


def _resolve_builtin_normalizer(name: str) -> Normalizer:
    if name in (None, "raw"):
        return None
    if name not in NORMALIZERS:
        raise ValueError(f"Unknown normalization strategy '{name}'. Choices: {list(NORMALIZERS)}")
    return NORMALIZERS[name]


def _prepare_normalizer(config: TrainConfig) -> tuple[Normalizer, Optional[str]]:
    if config.normalization == "config":
        if config.normalization_config is None:
            raise ValueError("normalization='config' requires --normalization-config.")
        profile = load_normalization_config(config.normalization_config)
        normalizer = build_normalizer(profile)
        return normalizer, compute_normalization_hash(profile)
    return _resolve_builtin_normalizer(config.normalization), None


def _prepare_bucket_hash(config: TrainConfig) -> Optional[str]:
    if not config.bucket_cache or not config.bucket_config:
        return None
    payload = load_bucket_config(config.bucket_config)
    return compute_bucket_hash(payload)


def _normalize_series(series: pd.Series, normalizer: Normalizer) -> pd.Series:
    series = series.fillna("").astype(str)
    if normalizer is None:
        return series
    return series.apply(normalizer)


def _resolve_text_series(
    df: pd.DataFrame,
    text_col: str,
    normalizer: Normalizer,
    config: TrainConfig,
    normalizer_hash: Optional[str],
) -> pd.Series:
    if config.normalized_cache is not None:
        if "row_index" not in df.columns:
            raise ValueError("row_index column missing; regenerate fold splits to use normalized cache.")
        series, cache_hash = _normalized_series_from_cache(config.normalized_cache, df["row_index"])
        _validate_cache_hash(normalizer_hash, cache_hash, "normalized-text", config.normalized_cache)
        return series
    return _normalize_series(df[text_col], normalizer)


def _normalized_series_from_cache(path: Path, row_index: pd.Series) -> tuple[pd.Series, Optional[str]]:
    cache_df = _read_parquet_cache(str(path))
    required = {"row_index", "normalized_text"}
    missing = required - set(cache_df.columns)
    if missing:
        raise ValueError(f"Normalized cache missing columns: {', '.join(sorted(missing))}")
    mapping = dict(zip(cache_df["row_index"], cache_df["normalized_text"]))
    series = row_index.map(mapping).fillna("")
    hash_value = _extract_cache_hash(cache_df, "config_hash")
    return series.astype(str), hash_value


def _bucket_tags_from_cache(path: Path, row_index: pd.Series) -> tuple[pd.Series, Optional[str]]:
    cache_df = _read_parquet_cache(str(path))
    required = {"row_index", "bucket_tags"}
    missing = required - set(cache_df.columns)
    if missing:
        raise ValueError(f"Bucket cache missing columns: {', '.join(sorted(missing))}")
    mapping = {}
    for key, raw_value in zip(cache_df["row_index"], cache_df["bucket_tags"]):
        mapping[int(key)] = _ensure_bucket_list(raw_value)

    def _lookup(idx: int) -> List[str]:
        value = mapping.get(int(idx))
        return list(value) if isinstance(value, list) else []

    series = row_index.map(_lookup)
    hash_value = _extract_cache_hash(cache_df, "config_hash")
    return series, hash_value


def _extract_cache_hash(df: pd.DataFrame, column: str) -> Optional[str]:
    if column not in df.columns:
        return None
    values = [val for val in df[column].dropna().unique() if isinstance(val, str)]
    return values[0] if values else None


def _validate_cache_hash(
    expected: Optional[str],
    actual: Optional[str],
    label: str,
    path: Path,
) -> None:
    if expected and actual and expected != actual:
        raise ValueError(
            f"{label} cache at {path} was produced with hash {actual}, "
            f"but expected {expected}. Regenerate the cache or pass the matching config."
        )


@lru_cache(maxsize=8)
def _read_parquet_cache(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


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
    extra_metadata: Optional[Dict[str, object]] = None,
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
        "normalization_config": str(config.normalization_config) if config.normalization_config else None,
        "normalized_cache": str(config.normalized_cache) if config.normalized_cache else None,
        "bucket_config": str(config.bucket_config) if config.bucket_config else None,
        "bucket_cache": str(config.bucket_cache) if config.bucket_cache else None,
        "bucket_cache_column": config.bucket_cache_column,
    })
    metadata_path = None
    if extra_metadata:
        metadata_path = fold_dir / "trainer_metadata.json"
        serializable = {}
        for key, value in extra_metadata.items():
            if isinstance(value, Path):
                serializable[key] = str(value)
            else:
                serializable[key] = value
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(serializable, handle, indent=2)
        payload["trainer_metadata"] = str(metadata_path)
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    if tfidf is not None:
        joblib.dump(tfidf, model_dir / "tfidf.joblib")
    if label_models:
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


def _maybe_apply_bucket_augmentation(
    train_df: pd.DataFrame,
    config: TrainConfig,
    bucket_hash: Optional[str],
) -> pd.DataFrame:
    """Optionally oversample training rows based on bucket multipliers."""

    if not config.bucket_multipliers:
        return train_df

    if "row_index" not in train_df.columns:
        raise ValueError("row_index column missing; cannot align bucket cache with training data.")

    prepared = train_df.copy()
    bucket_col = config.bucket_col

    if bucket_col == "auto":
        if config.bucket_cache is None:
            raise ValueError("bucket_col='auto' requires --bucket-cache to be set.")
        bucket_series, cache_hash = _bucket_tags_from_cache(config.bucket_cache, prepared["row_index"])
        _validate_cache_hash(bucket_hash, cache_hash, "bucket-tags", config.bucket_cache)
        bucket_col = config.bucket_cache_column
        prepared[bucket_col] = bucket_series

    if not bucket_col:
        raise ValueError("bucket_col must be specified when bucket_multipliers are provided.")
    if bucket_col not in prepared.columns:
        print(
            f"[bucket-oversample] Column '{bucket_col}' missing; skipping bucket augmentation."
        )
        return train_df

    prepared[bucket_col] = prepared[bucket_col].apply(_ensure_bucket_list)
    return oversample_buckets(prepared, bucket_col, config.bucket_multipliers)


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