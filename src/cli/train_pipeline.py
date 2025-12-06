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
    
    # For training all models on all folds with all seeds
    seeds = [42, 43, 44]
    folds = ["fold1", "fold2", "fold3"]
    models = ["logistic", "svm", "random_forest"]
    
    all_results = {}
    for model in models:
        model_results = {}
        for fold in folds:
            for seed in seeds:
                print(f"Training {model} on {fold}_seed{seed}")
                config = TrainConfig(
                    data_path=args.data_path,
                    splits_dir=args.splits_dir,
                    output_dir=Path("experiments/train") / f"tfidf_{model}",
                    fold=f"{fold}_seed{seed}",
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

                # Apply overrides
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
                # Load Naive Bayes config from YAML automatically (only for naive_bayes model)
                if config.model_type == "naive_bayes":
                    config_path = DEFAULT_NAIVE_BAYES_CONFIG_PATH
                    # Resolve path relative to project root
                    if not config_path.is_absolute():
                        project_root = Path(__file__).resolve().parents[2]
                        config_path = project_root / config_path
                    # Load YAML and extract hyperparameters if config file exists
                    if config_path.exists():
                        with open(config_path, "r", encoding="utf-8") as f:
                            nb_config = yaml.safe_load(f)
                        config.nb_params.update({
                            "alpha": nb_config.get("alpha", 1.0),
                            "fit_prior": nb_config.get("fit_prior", True),
                        })
                
                # CLI arguments override config file values
                if args.nb_alpha is not None:
                    config.nb_params["alpha"] = args.nb_alpha
                if args.nb_fit_prior is not None:
                    config.nb_params["fit_prior"] = args.nb_fit_prior.lower() == "true"
                if args.bert_model_name is not None:
                    config.bert_params["model_name"] = args.bert_model_name
                if args.bert_max_length is not None:
                    config.bert_params["max_length"] = args.bert_max_length
                if args.bert_train_batch_size is not None:
                    config.bert_params["train_batch_size"] = args.bert_train_batch_size
                if args.bert_eval_batch_size is not None:
                    config.bert_params["eval_batch_size"] = args.bert_eval_batch_size
                if args.bert_learning_rate is not None:
                    config.bert_params["learning_rate"] = args.bert_learning_rate
                if args.bert_weight_decay is not None:
                    config.bert_params["weight_decay"] = args.bert_weight_decay
                if args.bert_num_epochs is not None:
                    config.bert_params["num_epochs"] = args.bert_num_epochs
                if args.bert_warmup_ratio is not None:
                    config.bert_params["warmup_ratio"] = args.bert_warmup_ratio
                if args.bert_gradient_accumulation is not None:
                    config.bert_params["gradient_accumulation_steps"] = args.bert_gradient_accumulation
                if args.bert_fp16:
                    config.bert_params["fp16"] = True
                if args.bert_logging_steps is not None:
                    config.bert_params["logging_steps"] = args.bert_logging_steps
                if args.bert_save_total_limit is not None:
                    config.bert_params["save_total_limit"] = args.bert_save_total_limit

                results = run_training_pipeline(config)
                model_results.update(results)
                all_results.update(results)
        
        # Write summary for this model
        summary_path = Path("experiments/train") / f"tfidf_{model}" / "summary_metrics.json"
        summary_payload = {
            fold: metrics["overall_metrics"] for fold, metrics in model_results.items()
        }
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        # Load existing summary if it exists and merge
        if summary_path.exists():
            with open(summary_path, "r", encoding="utf-8") as handle:
                existing = json.load(handle)
            existing.update(summary_payload)
            summary_payload = existing
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary_payload, handle, indent=2)
    
    for fold_name, payload in all_results.items():
        metrics = payload["overall_metrics"]
        print(
            f"Fold {fold_name}: micro F1={metrics['micro_f1']:.4f}, "
            f"macro F1={metrics['macro_f1']:.4f}, hamming_loss={metrics['hamming_loss']:.4f}"
        )


if __name__ == "__main__":
    main()
