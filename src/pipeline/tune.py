"""Hyperparameter tuning utilities for the toxic comment classification pipeline."""

from __future__ import annotations

print("Starting script execution")

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score
import numpy as np

from src.pipeline.train import TrainConfig, run_training_pipeline
from src.data.dataset import load_fold_frames
from src.models.tfidf_logistic import train_multilabel_tfidf_logistic
from src.models.tfidf_svm import train_multilabel_tfidf_linear_svm
from src.models.tfidf_random_forest import train_multilabel_tfidf_random_forest
from src.features.tfidf import create_tfidf_vectorizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def tune_sklearn_model(
    model_type: str,
    param_grid: Dict[str, List[Any]],
    train_texts: List[str],
    train_labels: np.ndarray,
    dev_texts: List[str],
    dev_labels: np.ndarray,
    label_cols: List[str],
    vectorizer_params: Dict[str, Any],
    cv_folds: int = 3,
    scoring: str = "f1_macro",
    n_iter: Optional[int] = None,  # For RandomizedSearchCV
    random_state: int = 42,
) -> Dict[str, Any]:
    """Tune hyperparameters for sklearn-based TF-IDF models using cross-validation.

    Returns the best parameters and CV scores.
    """
    logging.info(f"Starting tuning for model: {model_type}")
    
    if model_type == "logistic":
        from sklearn.linear_model import LogisticRegression
        base_model = LogisticRegression(random_state=random_state)
        trainer_func = train_multilabel_tfidf_logistic
    elif model_type == "svm":
        from sklearn.svm import LinearSVC
        base_model = LinearSVC(random_state=random_state)
        trainer_func = train_multilabel_tfidf_linear_svm
    elif model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        base_model = RandomForestClassifier(random_state=random_state)
        trainer_func = train_multilabel_tfidf_random_forest
    else:
        raise ValueError(f"Tuning not supported for model_type: {model_type}")

    # For multilabel, we need to handle per-label tuning or use a custom scorer
    # For simplicity, we'll tune on a single representative label or use macro F1
    # In practice, you might want to tune per label or use a custom multilabel scorer

    # Fit TF-IDF on train data
    vectorizer = create_tfidf_vectorizer(**vectorizer_params)
    X_train = vectorizer.fit_transform(train_texts)
    X_dev = vectorizer.transform(dev_texts)

    # Use dev set for final evaluation, but tune on a subset of train for speed
    # For full rigor, you'd do nested CV, but this is a starting point

    best_params_per_label = {}
    best_scores = {}

    for label_idx, label in enumerate(label_cols):
        logging.info(f"Tuning hyperparameters for label: {label} (model: {model_type})")
        y_train_label = train_labels[:, label_idx]
        y_dev_label = dev_labels[:, label_idx]

        # Custom scorer for this label
        scorer = make_scorer(f1_score, average='macro')  # Or 'micro' depending on your needs

        if n_iter is not None:
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=n_iter,
                cv=cv_folds,
                scoring=scorer,
                random_state=random_state,
                n_jobs=-1,
            )
        else:
            search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv_folds,
                scoring=scorer,
                n_jobs=-1,
            )

        search.fit(X_train, y_train_label)
        best_params_per_label[label] = search.best_params_
        best_scores[label] = search.best_score_
        logging.info(f"Best params for {label}: {search.best_params_}, Best CV score: {search.best_score_:.4f}")

    logging.info(f"Completed tuning for model: {model_type}")
    return {
        "best_params": best_params_per_label,
        "best_scores": best_scores,
        "model_type": model_type,
    }


def run_hyperparameter_tuning(
    config: TrainConfig,
    param_grids: Dict[str, Dict[str, List[Any]]],
    tuning_output_dir: Path,
    cv_folds: int = 3,
    n_iter: Optional[int] = None,
) -> Dict[str, Any]:
    """Run hyperparameter tuning across folds and persist results."""
    logging.info(f"Starting hyperparameter tuning for seed: {config.seed}, folds: {config.fold or 'all'}")
    
    base_df, fold_frames, identity_cols, _ = load_fold_frames(
        seed=config.seed,
        data_path=config.data_path,
        splits_dir=config.splits_dir,
    )
    label_cols = config.resolve_label_cols(base_df)

    tuning_results = {}
    for fold_name in fold_frames.keys():
        if config.fold and config.fold != fold_name:
            continue

        logging.info(f"Processing fold: {fold_name}")
        fold_data = fold_frames[fold_name]
        train_df = fold_data["train"]
        dev_df = fold_data["dev"]

        # Prepare text data (simplified, without full normalization for tuning speed)
        train_texts = train_df[config.text_col].fillna("").tolist()
        dev_texts = dev_df[config.text_col].fillna("").tolist()
        train_labels = train_df[label_cols].values
        dev_labels = dev_df[label_cols].values

        fold_results = {}
        for model_type, param_grid in param_grids.items():
            logging.info(f"Starting tuning for model: {model_type} on fold: {fold_name}")
            config.model_type = model_type
            result = tune_sklearn_model(
                model_type=model_type,
                param_grid=param_grid,
                train_texts=train_texts,
                train_labels=train_labels,
                dev_texts=dev_texts,
                dev_labels=dev_labels,
                label_cols=label_cols,
                vectorizer_params=config.vectorizer_params,
                cv_folds=cv_folds,
                n_iter=n_iter,
                random_state=config.seed,
            )
            fold_results[model_type] = result
            logging.info(f"Completed tuning for model: {model_type} on fold: {fold_name}")

        tuning_results[fold_name] = fold_results

        # Save fold results
        fold_dir = tuning_output_dir / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)
        with open(fold_dir / "tuning_results.json", "w") as f:
            json.dump(fold_results, f, indent=2)
        logging.info(f"Saved tuning results for fold: {fold_name} to {fold_dir}")

    # Save summary
    summary = {}
    for fold, models in tuning_results.items():
        for model, result in models.items():
            if model not in summary:
                summary[model] = {}
            summary[model][fold] = {
                "best_params": result["best_params"],
                "best_scores": result["best_scores"],
            }

    with open(tuning_output_dir / "tuning_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Saved tuning summary to {tuning_output_dir / 'tuning_summary.json'}")

    logging.info("Hyperparameter tuning completed.")
    return tuning_results


# Example usage in a script:
if __name__ == "__main__":
    print("Starting tuning script")
    # Define parameter grids for tuning
    param_grids = {
        "logistic": {
            "C": [0.1, 1.0, 10.0],
            "max_iter": [200, 400, 1000],
        },
        "svm": {
            "C": [0.1, 1.0, 10.0],
            "max_iter": [1000, 2000],
        },
        "random_forest": {
            "n_estimators": [100, 200],
            "max_depth": [10, None],
        },
    }

    seeds = [42, 43, 44]  # Tune across all available seeds
    folds = ["fold1", "fold2", "fold3"]
    for fold in folds:
        for seed in seeds:
            logging.info(f"Starting hyperparameter tuning for {fold}_seed{seed}")
            config = TrainConfig(
                fold=f"{fold}_seed{seed}",
                model_type="logistic",  # Will be overridden per model
                seed=seed,
                output_dir=Path(f"experiments/hyperparameter_tuning_{fold}_seed{seed}"),
            )

            tuning_output_dir = Path(f"experiments/hyperparameter_tuning_{fold}_seed{seed}")
            results = run_hyperparameter_tuning(
                config=config,
                param_grids=param_grids,
                tuning_output_dir=tuning_output_dir,
                cv_folds=3,
                n_iter=None,  # Use GridSearchCV for exact search
            )

            logging.info(f"Tuning completed for {fold}_seed{seed}. Check {tuning_output_dir}/ for results.")