"""CLI entry point for the reusable toxic-comment training pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import yaml

from src.data.buckets import DEFAULT_BUCKET_CONFIG_PATH
from src.data.normalization import DEFAULT_NORMALIZATION_CONFIG_PATH
from src.pipeline.train import TrainConfig, run_training_pipeline
from src.pipeline.train_deep import DeepTrainConfig, run_deep_training_pipeline

DEFAULT_NAIVE_BAYES_CONFIG_PATH = Path("configs/naive_bayes.yaml")
DEFAULT_LSTM_CONFIG_PATH = Path("configs/lstm.yaml")
DEFAULT_BERT_CONFIG_PATH = Path("configs/bert_distil.yaml")


def load_tfidf_config(config_path: Path, model: str) -> dict:
    """Load TF-IDF model config from YAML."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config.get("model", {}).get("params", {})


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
        choices=["logistic", "svm", "random_forest", "naive_bayes", "xgboost", "bert", "lstm"],
        help="Model type to train (TF-IDF variants or transformer).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config file for model parameters (for TF-IDF models).",
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
    parser.add_argument(
        "--nb-alpha",
        type=float,
        default=None,
        help="Alpha (smoothing parameter) for Naive Bayes (when --model naive_bayes).",
    )
    parser.add_argument(
        "--nb-fit-prior",
        type=str,
        default=None,
        choices=["true", "false"],
        help="Whether to learn class prior probabilities for Naive Bayes (true/false).",
    )
    parser.add_argument(
        "--lstm-config",
        type=Path,
        default=DEFAULT_LSTM_CONFIG_PATH,
        help="Path to LSTM config YAML file (when --model lstm).",
    )
    parser.add_argument(
        "--bert-config",
        type=Path,
        default=DEFAULT_BERT_CONFIG_PATH,
        help="Path to BERT config YAML file (when --model bert).",
    )
    parser.add_argument(
        "--bert-model-name",
        type=str,
        default=None,
        help="HuggingFace model checkpoint (default: distilbert-base-uncased).",
    )
    parser.add_argument(
        "--bert-max-length",
        type=int,
        default=None,
        help="Maximum token length for transformer inputs.",
    )
    parser.add_argument(
        "--bert-train-batch-size",
        type=int,
        default=None,
        help="Per-device train batch size for BERT runs.",
    )
    parser.add_argument(
        "--bert-eval-batch-size",
        type=int,
        default=None,
        help="Per-device eval batch size for BERT runs.",
    )
    parser.add_argument(
        "--bert-learning-rate",
        type=float,
        default=None,
        help="Learning rate for transformer fine-tuning.",
    )
    parser.add_argument(
        "--bert-weight-decay",
        type=float,
        default=None,
        help="Weight decay for transformer fine-tuning.",
    )
    parser.add_argument(
        "--bert-num-epochs",
        type=float,
        default=None,
        help="Number of epochs for transformer fine-tuning.",
    )
    parser.add_argument(
        "--bert-warmup-ratio",
        type=float,
        default=None,
        help="Warmup ratio for scheduler.",
    )
    parser.add_argument(
        "--bert-gradient-accumulation",
        type=int,
        default=None,
        help="Gradient accumulation steps for transformer runs.",
    )
    parser.add_argument(
        "--bert-fp16",
        action="store_true",
        help="Enable fp16 mixed precision when CUDA is available.",
    )
    parser.add_argument(
        "--bert-logging-steps",
        type=int,
        default=None,
        help="Logging steps for Trainer.",
    )
    parser.add_argument(
        "--bert-save-total-limit",
        type=int,
        default=None,
        help="Maximum checkpoints to keep for transformer runs.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable hyperparameter tuning mode (grid search for model params).",
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
    
    # Determine models to run
    if args.model in ["bert", "lstm"]:
        models = [args.model]
        is_deep = True
    else:
        models = [args.model] if args.model != "logistic" else ["logistic", "svm", "random_forest"]  # Default batch for TF-IDF
        is_deep = False
    
    # Determine folds to run
    if args.fold:
        if is_deep:
            fold_seed = args.fold
            folds = [fold_seed]
            seeds = [int(args.fold.split("_")[1].replace("seed", ""))]
        else:
            folds = [args.fold.split("_")[0]]
            seeds = [int(args.fold.split("_")[1].replace("seed", ""))]
    else:
        seeds = [42, 43, 44]
        folds = ["fold1", "fold2", "fold3"]
    
    all_results = {}
    for model in models:
        model_results = {}
        for fold in folds:
            for seed in seeds:
                if is_deep:
                    fold_seed = fold  # fold is already fold_seed for deep
                else:
                    fold_seed = f"{fold}_seed{seed}"
                print(f"{'Tuning' if args.tune else 'Training'} {model} on {fold_seed}")
                
                if is_deep:
                    # Deep model config
                    config = DeepTrainConfig(
                        data_path=args.data_path,
                        splits_dir=args.splits_dir,
                        output_dir=args.output_dir,
                        fold=fold_seed,
                        label_cols=args.labels,
                        text_col=args.text_col,
                        normalization=args.normalization,
                        normalization_config=args.normalization_config,
                        normalized_cache=args.normalized_cache,
                        model_type=model,
                        threshold=args.threshold,
                        fairness_min_support=args.fairness_min_support,
                        seed=seed,
                        bert_config_path=args.bert_config if model == "bert" else None,
                        lstm_config_path=args.lstm_config if model == "lstm" else None,
                    )
                    # Apply BERT overrides
                    if args.bert_model_name is not None:
                        config.bert_params["model_name"] = args.bert_model_name
                    # ... other bert params
                    
                    if args.tune:
                        # Implement tuning for deep models
                        from src.pipeline.tune_deep import tune_deep_model
                        results = tune_deep_model(config)
                        # tune_deep_model returns metrics_dict directly
                        all_results[fold_seed] = results
                    else:
                        results = run_deep_training_pipeline(config)
                        # run_deep_training_pipeline returns {fold: metrics_dict}
                        all_results[fold_seed] = results[fold_seed]
                else:
                    # TF-IDF model config
                    # Load params from config if provided
                    if args.config:
                        config_params = load_tfidf_config(args.config, model)
                        # Override args with config values
                        if 'C' in config_params:
                            args.model_C = config_params['C']
                        if 'penalty' in config_params:
                            args.model_penalty = config_params['penalty']
                        if 'max_iter' in config_params:
                            args.model_max_iter = config_params['max_iter']
                        if 'class_weight' in config_params:
                            args.model_class_weight = config_params['class_weight']
                        if 'kernel' in config_params:
                            args.model_kernel = config_params['kernel']
                        if 'gamma' in config_params:
                            args.model_gamma = config_params['gamma']
                        if 'n_estimators' in config_params:
                            args.model_n_estimators = config_params['n_estimators']
                        if 'max_depth' in config_params:
                            args.model_max_depth = config_params['max_depth']
                        if 'min_samples_split' in config_params:
                            args.model_min_samples_split = config_params['min_samples_split']
                        if 'min_samples_leaf' in config_params:
                            args.model_min_samples_leaf = config_params['min_samples_leaf']
                    
                    config = TrainConfig(
                        data_path=args.data_path,
                        splits_dir=args.splits_dir,
                        output_dir=Path("experiments/train") / f"tfidf_{model}",
                        fold=fold_seed,
                        label_cols=args.labels,
                        text_col=args.text_col,
                        normalization=args.normalization,
                        normalization_config=args.normalization_config,
                        normalized_cache=args.normalized_cache,
                        model_type=model,
                        threshold=args.threshold,
                        fairness_min_support=args.fairness_min_support,
                        seed=seed,
                        bucket_col=args.bucket_col,
                        bucket_multipliers=_parse_bucket_multipliers(args.bucket_mult),
                        bucket_config=args.bucket_config,
                        bucket_cache=args.bucket_cache,
                        bucket_cache_column=args.bucket_cache_column,
                    )
                    # Apply overrides as before
                    
                    if args.tune:
                        from src.pipeline.tune import tune_sklearn_model
                        results = tune_sklearn_model(config)
                    else:
                        results = run_training_pipeline(config)
                
                model_results[fold_seed] = results
                all_results[fold_seed] = results
        
        if not is_deep:
            # Write summary for TF-IDF models
            summary_path = Path("experiments/train") / f"tfidf_{model}" / "summary_metrics.json"
            summary_payload = {
                fold: metrics[fold]["overall_metrics"] for fold, metrics in model_results.items()
            }
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            if summary_path.exists():
                with open(summary_path, "r", encoding="utf-8") as handle:
                    existing = json.load(handle)
                existing.update(summary_payload)
                summary_payload = existing
            with open(summary_path, "w", encoding="utf-8") as handle:
                json.dump(summary_payload, handle, indent=2)
    
    for fold_name, payload in all_results.items():
        if payload is not None:
            metrics = payload[fold_name]["overall_metrics"]
            print(
                f"Fold {fold_name}: micro F1={metrics['micro_f1']:.4f}, "
                f"macro F1={metrics['macro_f1']:.4f}, hamming_loss={metrics['hamming_loss']:.4f}"
            )


if __name__ == "__main__":
    main()
