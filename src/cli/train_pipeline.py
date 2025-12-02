"""CLI entry point for the TF-IDF + logistic training pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from src.data.buckets import DEFAULT_BUCKET_CONFIG_PATH
from src.data.normalization import DEFAULT_NORMALIZATION_CONFIG_PATH
from src.pipeline.train import TrainConfig, run_training_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the reusable TF-IDF + logistic training pipeline."
    )
    parser.add_argument("--data-path", type=Path, default=Path("data/raw/train.csv"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/tfidf_logreg"),
        help="Directory where fold artifacts will be written.",
    )
    parser.add_argument(
        "--fold",
        type=str,
        default=None,
        help="Optional specific fold to run (default: run all folds).",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        help="Override label columns (space separated).",
    )
    parser.add_argument(
        "--text-col",
        type=str,
        default="comment_text",
        help="Name of the text column used for training.",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default="toy",
        choices=["raw", "toy", "rich", "config"],
        help="Text normalization strategy.",
    )
    parser.add_argument(
        "--normalization-config",
        type=Path,
        default=DEFAULT_NORMALIZATION_CONFIG_PATH,
        help="Normalization YAML used when --normalization config or cache builders are active.",
    )
    parser.add_argument(
        "--normalized-cache",
        type=Path,
        default=None,
        help="Optional parquet file from make_normalized_text; overrides on-the-fly normalization.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="logistic",
        choices=["logistic", "svm", "random_forest"],
        help="Model type to train on TF-IDF features.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for binarizing predictions.",
    )
    parser.add_argument(
        "--fairness-min-support",
        type=int,
        default=50,
        help="Minimum subgroup support for fairness slices.",
    )
    parser.add_argument(
        "--bucket-col",
        type=str,
        default=None,
        help="Column containing bucket tag lists for oversampling (set to 'auto' to read from cache).",
    )
    parser.add_argument(
        "--bucket-mult",
        action="append",
        default=None,
        metavar="BUCKET=FACTOR",
        help="Repeat to oversample specific buckets (e.g., rare=3).",
    )
    parser.add_argument(
        "--bucket-config",
        type=Path,
        default=DEFAULT_BUCKET_CONFIG_PATH,
        help="Bucket YAML used to validate cache hashes.",
    )
    parser.add_argument(
        "--bucket-cache",
        type=Path,
        default=None,
        help="Optional parquet file produced by make_bucket_tags.",
    )
    parser.add_argument(
        "--bucket-cache-column",
        type=str,
        default="bucket_tags",
        help="Column name inside the bucket cache to attach when --bucket-col auto.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed forwarded to fold loader.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=None,
        help="Override TF-IDF max_features.",
    )
    parser.add_argument(
        "--ngram-max",
        type=int,
        default=None,
        help="Override TF-IDF ngram upper bound (lower bound stays 1).",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=None,
        help="Override TF-IDF min_df setting.",
    )
    parser.add_argument(
        "--max-df",
        type=float,
        default=None,
        help="Override TF-IDF max_df setting.",
    )
    parser.add_argument(
        "--model-C",
        type=float,
        default=None,
        help="Override LogisticRegression C value.",
    )
    parser.add_argument(
        "--model-max-iter",
        type=int,
        default=None,
        help="Override LogisticRegression max_iter value.",
    )
    parser.add_argument(
        "--svm-C",
        type=float,
        default=None,
        help="Override LinearSVC C value (when --model svm).",
    )
    parser.add_argument(
        "--svm-max-iter",
        type=int,
        default=None,
        help="Override LinearSVC max_iter (when --model svm).",
    )
    parser.add_argument(
        "--svm-class-weight",
        type=str,
        default=None,
        help="Override LinearSVC class_weight (e.g., balanced).",
    )
    parser.add_argument(
        "--svm-calib-method",
        type=str,
        default=None,
        choices=["sigmoid", "isotonic"],
        help="Calibration method for SVM probabilities.",
    )
    parser.add_argument(
        "--svm-calib-cv",
        type=int,
        default=None,
        help="Number of folds for SVM calibration CV.",
    )
    parser.add_argument(
        "--rf-n-estimators",
        type=int,
        default=None,
        help="Number of trees for RandomForest (when --model random_forest).",
    )
    parser.add_argument(
        "--rf-max-depth",
        type=int,
        default=None,
        help="Max tree depth for RandomForest.",
    )
    parser.add_argument(
        "--rf-max-features",
        type=str,
        default=None,
        help="Max features per split (e.g., sqrt, log2, auto).",
    )
    parser.add_argument(
        "--rf-class-weight",
        type=str,
        default=None,
        help="Class weight strategy for RandomForest (e.g., balanced).",
    )
    parser.add_argument(
        "--rf-min-samples-split",
        type=int,
        default=None,
        help="Minimum samples to split for RandomForest.",
    )
    parser.add_argument(
        "--rf-min-samples-leaf",
        type=int,
        default=None,
        help="Minimum samples per leaf for RandomForest.",
    )
    parser.add_argument(
        "--rf-n-jobs",
        type=int,
        default=None,
        help="Parallel jobs for RandomForest (-1 uses all cores).",
    )
    return parser.parse_args()


def _parse_bucket_multipliers(values: Optional[list[str]]) -> Optional[dict[str, int]]:
    if not values:
        return None
    multipliers: dict[str, int] = {}
    for entry in values:
        if "=" not in entry:
            raise ValueError(f"Bucket multiplier '{entry}' must be in NAME=FACTOR format.")
        name, factor = entry.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError("Bucket name cannot be empty.")
        try:
            multipliers[name] = int(factor)
        except ValueError as exc:
            raise ValueError(f"Bucket multiplier for '{name}' must be an integer: {factor}") from exc
    return multipliers


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        data_path=args.data_path,
        splits_dir=args.splits_dir,
        output_dir=args.output_dir,
        fold=args.fold,
        label_cols=args.labels,
        text_col=args.text_col,
        normalization=args.normalization,
        normalization_config=args.normalization_config,
        normalized_cache=args.normalized_cache,
        model_type=args.model,
        threshold=args.threshold,
        fairness_min_support=args.fairness_min_support,
        seed=args.seed,
        bucket_col=args.bucket_col,
        bucket_multipliers=_parse_bucket_multipliers(args.bucket_mult),
        bucket_config=args.bucket_config,
        bucket_cache=args.bucket_cache,
        bucket_cache_column=args.bucket_cache_column,
    )

    if args.max_features is not None:
        config.vectorizer_params["max_features"] = args.max_features
    if args.ngram_max is not None:
        config.vectorizer_params["ngram_range"] = (1, args.ngram_max)
    if args.min_df is not None:
        config.vectorizer_params["min_df"] = args.min_df
    if args.max_df is not None:
        config.vectorizer_params["max_df"] = args.max_df
    if args.model_C is not None:
        config.model_params["C"] = args.model_C
    if args.model_max_iter is not None:
        config.model_params["max_iter"] = args.model_max_iter
    if args.svm_C is not None:
        config.svm_params["C"] = args.svm_C
    if args.svm_max_iter is not None:
        config.svm_params["max_iter"] = args.svm_max_iter
    if args.svm_class_weight is not None:
        config.svm_params["class_weight"] = args.svm_class_weight
    if args.svm_calib_method is not None:
        config.svm_calibration_params["method"] = args.svm_calib_method
    if args.svm_calib_cv is not None:
        config.svm_calibration_params["cv"] = args.svm_calib_cv
    if args.rf_n_estimators is not None:
        config.rf_params["n_estimators"] = args.rf_n_estimators
    if args.rf_max_depth is not None:
        config.rf_params["max_depth"] = args.rf_max_depth
    if args.rf_max_features is not None:
        config.rf_params["max_features"] = args.rf_max_features
    if args.rf_class_weight is not None:
        config.rf_params["class_weight"] = args.rf_class_weight
    if args.rf_min_samples_split is not None:
        config.rf_params["min_samples_split"] = args.rf_min_samples_split
    if args.rf_min_samples_leaf is not None:
        config.rf_params["min_samples_leaf"] = args.rf_min_samples_leaf
    if args.rf_n_jobs is not None:
        config.rf_params["n_jobs"] = args.rf_n_jobs

    results = run_training_pipeline(config)
    for fold_name, payload in results.items():
        metrics = payload["overall_metrics"]
        print(
            f"Fold {fold_name}: micro F1={metrics['micro_f1']:.4f}, "
            f"macro F1={metrics['macro_f1']:.4f}, hamming_loss={metrics['hamming_loss']:.4f}"
        )


if __name__ == "__main__":
    main()
