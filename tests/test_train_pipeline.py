"""Tests for the reusable training pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.pipeline.train import TrainConfig, run_training_pipeline


@pytest.fixture()
def toy_data(tmp_path: Path) -> tuple[Path, Path]:
    data_dir = tmp_path / "data" / "raw"
    splits_dir = tmp_path / "data" / "splits"
    data_dir.mkdir(parents=True)
    splits_dir.mkdir(parents=True)

    rows = [
        {"id": 1, "comment_text": "nice", "toxic": 0, "obscene": 0, "bucket_tags_full": "[]"},
        {
            "id": 2,
            "comment_text": "mean rare",
            "toxic": 1,
            "obscene": 0,
            "bucket_tags_full": "[\"rare\"]",
        },
        {
            "id": 3,
            "comment_text": "awful rare",
            "toxic": 1,
            "obscene": 1,
            "bucket_tags_full": "[\"rare\"]",
        },
        {
            "id": 4,
            "comment_text": "awful",
            "toxic": 0,
            "obscene": 1,
            "bucket_tags_full": "[]",
        },
        {
            "id": 5,
            "comment_text": "plain",
            "toxic": 0,
            "obscene": 0,
            "bucket_tags_full": "[]",
        },
        {
            "id": 6,
            "comment_text": "nasty",
            "toxic": 1,
            "obscene": 1,
            "bucket_tags_full": "[\"rare\"]",
        },
    ]
    df = pd.DataFrame(rows)
    data_path = data_dir / "train.csv"
    df.to_csv(data_path, index=False)

    split_payload = {
        "train": [0, 1, 2, 3],
        "dev": [4],
        "test": [5],
    }
    (splits_dir / "fold1_seed42.json").write_text(json.dumps(split_payload), encoding="utf-8")

    return data_path, splits_dir


def test_run_pipeline_with_bucket_augmentation(tmp_path: Path, toy_data: tuple[Path, Path]) -> None:
    data_path, splits_dir = toy_data
    output_dir = tmp_path / "experiments"
    config = TrainConfig(
        data_path=data_path,
        splits_dir=splits_dir,
        output_dir=output_dir,
        fold="fold1_seed42",
        label_cols=["toxic", "obscene"],
        text_col="comment_text",
        normalization="raw",
        threshold=0.5,
        fairness_min_support=10,
        seed=42,
        bucket_col="bucket_tags_full",
        bucket_multipliers={"rare": 2},
        vectorizer_params={
            "max_features": 1000,
            "ngram_range": (1, 1),
            "min_df": 1,
            "max_df": 1.0,
            "lowercase": True,
            "strip_accents": None,
        },
        model_params={
            "max_iter": 100,
            "class_weight": None,
            "solver": "liblinear",
            "C": 1.0,
        },
    )

    results = run_training_pipeline(config)

    assert "fold1_seed42" in results
    metrics = results["fold1_seed42"]["overall_metrics"]
    assert "micro_f1" in metrics and "macro_f1" in metrics

    folds = list(output_dir.glob("fold1_seed42-*"))
    assert folds, "Pipeline should create a timestamped fold directory"
    fold_dir = folds[0]
    assert (fold_dir / "models" / "tfidf.joblib").exists()
    assert (fold_dir / "per_label_metrics.csv").exists()


def test_pipeline_requires_bucket_column(tmp_path: Path, toy_data: tuple[Path, Path]) -> None:
    data_path, splits_dir = toy_data
    output_dir = tmp_path / "experiments"
    config = TrainConfig(
        data_path=data_path,
        splits_dir=splits_dir,
        output_dir=output_dir,
        fold="fold1_seed42",
        label_cols=["toxic", "obscene"],
        bucket_col="missing",
        bucket_multipliers={"rare": 2},
    )

    with pytest.raises(ValueError):
        run_training_pipeline(config)


def test_run_pipeline_with_linear_svm(tmp_path: Path, toy_data: tuple[Path, Path]) -> None:
    data_path, splits_dir = toy_data
    output_dir = tmp_path / "experiments"
    config = TrainConfig(
        data_path=data_path,
        splits_dir=splits_dir,
        output_dir=output_dir,
        fold="fold1_seed42",
        label_cols=["toxic", "obscene"],
        model_type="svm",
        vectorizer_params={
            "max_features": 500,
            "ngram_range": (1, 1),
            "min_df": 1,
            "max_df": 1.0,
            "lowercase": True,
            "strip_accents": None,
        },
        svm_params={
            "C": 0.5,
            "class_weight": None,
            "max_iter": 1000,
        },
        svm_calibration_params={"method": "sigmoid", "cv": 2},
    )

    results = run_training_pipeline(config)
    assert "fold1_seed42" in results
    metrics = results["fold1_seed42"]["overall_metrics"]
    assert metrics["micro_f1"] >= 0.0

    fold_dirs = list(output_dir.glob("fold1_seed42-*"))
    assert fold_dirs, "SVM pipeline should emit a fold directory"


def test_run_pipeline_with_random_forest(tmp_path: Path, toy_data: tuple[Path, Path]) -> None:
    data_path, splits_dir = toy_data
    output_dir = tmp_path / "experiments"
    config = TrainConfig(
        data_path=data_path,
        splits_dir=splits_dir,
        output_dir=output_dir,
        fold="fold1_seed42",
        label_cols=["toxic", "obscene"],
        model_type="random_forest",
        vectorizer_params={
            "max_features": 500,
            "ngram_range": (1, 1),
            "min_df": 1,
            "max_df": 1.0,
            "lowercase": True,
            "strip_accents": None,
        },
        rf_params={
            "n_estimators": 25,
            "max_depth": 4,
            "max_features": 1,
            "class_weight": None,
            "n_jobs": 1,
            "random_state": 0,
        },
    )

    results = run_training_pipeline(config)
    assert "fold1_seed42" in results
    metrics = results["fold1_seed42"]["overall_metrics"]
    assert "macro_f1" in metrics

    rf_dirs = list(output_dir.glob("fold1_seed42-*modelrandom_forest*"))
    assert rf_dirs, "RandomForest pipeline should emit a model-specific directory"
