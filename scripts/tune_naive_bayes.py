"""Hyperparameter tuning script for Naive Bayes alpha values."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

import pandas as pd

from src.data.dataset import load_fold_frames
from src.data.normalization import DEFAULT_NORMALIZATION_CONFIG_PATH, build_normalizer, load_normalization_config
from src.data.preprocess import toy_normalize
from src.features.tfidf import create_tfidf_vectorizer
from src.models.tfidf_naive_bayes import train_multilabel_tfidf_naive_bayes
from src.pipeline.train import TrainConfig, _prepare_normalizer, _resolve_text_series
from src.utils.metrics import compute_multilabel_metrics, probs_to_preds

DEFAULT_NAIVE_BAYES_CONFIG_PATH = Path("configs/naive_bayes.yaml")
DEFAULT_ALPHAS = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]


def tune_alpha(
    config: TrainConfig,
    alphas: List[float],
    fold_name: str,
) -> pd.DataFrame:
    """Tune alpha on dev set, return results for all alphas."""
    
    # Load data
    base_df, fold_frames, identity_cols, _ = load_fold_frames(
        seed=config.seed,
        data_path=config.data_path,
        splits_dir=config.splits_dir,
    )
    label_cols = config.resolve_label_cols(base_df)
    
    if fold_name not in fold_frames:
        raise ValueError(f"Fold '{fold_name}' not found. Available: {list(fold_frames.keys())}")
    
    fold_splits = fold_frames[fold_name]
    
    # Prepare data
    train_df = fold_splits["train"].reset_index(drop=True)
    dev_df = fold_splits["dev"].reset_index(drop=True)
    
    normalizer, _ = _prepare_normalizer(config)
    X_train = _resolve_text_series(train_df, config.text_col, normalizer, config, None)
    X_dev = _resolve_text_series(dev_df, config.text_col, normalizer, config, None)
    
    y_train = train_df[label_cols].values.astype(int)
    y_dev = dev_df[label_cols].values.astype(int)
    
    # Create TF-IDF
    tfidf = create_tfidf_vectorizer(**config.vectorizer_params)
    X_train_vec = tfidf.fit_transform(X_train.tolist())
    X_dev_vec = tfidf.transform(X_dev.tolist())
    
    # Tune each alpha
    results = []
    for alpha in alphas:
        print(f"Testing alpha={alpha}...")
        
        # Train models
        nb_params = {"alpha": alpha, "fit_prior": True}
        _, label_models = train_multilabel_tfidf_naive_bayes(
            X_train.tolist(),
            y_train,
            label_cols,
            vectorizer_params=config.vectorizer_params,
            nb_params=nb_params,
        )
        
        # Evaluate on dev set
        dev_probs = {
            label: model.predict_proba(X_dev_vec)[:, 1]
            for label, model in label_models.items()
        }
        y_dev_pred = probs_to_preds(dev_probs, threshold=config.threshold)
        
        metrics, _ = compute_multilabel_metrics(y_dev, y_dev_pred, label_cols)
        
        results.append({
            "alpha": alpha,
            "micro_f1": metrics["micro_f1"],
            "macro_f1": metrics["macro_f1"],
            "micro_precision": metrics["micro_precision"],
            "micro_recall": metrics["micro_recall"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "hamming_loss": metrics["hamming_loss"],
            "subset_accuracy": metrics["subset_accuracy"],
        })
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description="Tune Naive Bayes alpha hyperparameter on dev set"
    )
    parser.add_argument(
        "--fold",
        type=str,
        default="fold1_seed42",
        help="Fold to tune on (e.g., fold1_seed42)",
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=None,
        help="Alpha values to test. If not specified, loads from configs/naive_bayes.yaml",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/raw/train.csv"),
        help="Path to training data",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data/splits"),
        help="Directory containing fold JSON files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default="toy",
        choices=["raw", "toy", "rich", "config"],
        help="Text normalization strategy",
    )
    parser.add_argument(
        "--normalization-config",
        type=Path,
        default=DEFAULT_NORMALIZATION_CONFIG_PATH,
        help="Normalization config path (when --normalization config)",
    )
    parser.add_argument(
        "--normalized-cache",
        type=Path,
        default=None,
        help="Optional normalized text cache",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=None,
        help="TF-IDF max_features",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for predictions",
    )
    parser.add_argument(
        "--nb-config-path",
        type=Path,
        default=None,
        help="Path to Naive Bayes config YAML file (default: configs/naive_bayes.yaml)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/nb_tuning/results.csv"),
        help="Output CSV file for results",
    )
    args = parser.parse_args()
    
    # Load alpha values from config file if not specified
    if args.alphas is None:
        config_path = args.nb_config_path or DEFAULT_NAIVE_BAYES_CONFIG_PATH
        # Resolve path relative to project root
        if not config_path.is_absolute():
            project_root = Path(__file__).parent.parent
            config_path = project_root / config_path
        # Load YAML and extract tuning alphas
        with open(config_path, "r", encoding="utf-8") as f:
            nb_config = yaml.safe_load(f)
        args.alphas = nb_config.get("tuning_alphas", DEFAULT_ALPHAS)
        print(f"Loaded alpha values from config: {args.alphas}")
    
    # Create config
    config = TrainConfig(
        data_path=args.data_path,
        splits_dir=args.splits_dir,
        seed=args.seed,
        normalization=args.normalization,
        normalization_config=args.normalization_config,
        normalized_cache=args.normalized_cache,
        threshold=args.threshold,
    )
    
    if args.max_features is not None:
        config.vectorizer_params["max_features"] = args.max_features
    
    print(f"Tuning Naive Bayes alpha values: {args.alphas}")
    print(f"Fold: {args.fold}")
    print(f"Normalization: {args.normalization}")
    print()
    
    # Run tuning
    results_df = tune_alpha(config, args.alphas, args.fold)
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.output, index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("TUNING RESULTS")
    print("="*60)
    print(results_df.to_string(index=False))
    print()
    
    # Find best alpha
    best_idx = results_df["micro_f1"].idxmax()
    best_alpha = results_df.loc[best_idx, "alpha"]
    best_micro_f1 = results_df.loc[best_idx, "micro_f1"]
    best_macro_f1 = results_df.loc[best_idx, "macro_f1"]
    
    print(f"Best alpha: {best_alpha}")
    print(f"  Micro F1: {best_micro_f1:.4f}")
    print(f"  Macro F1: {best_macro_f1:.4f}")
    print()
    print(f"Results saved to: {args.output}")
    print()
    print(f"To train with best alpha, run:")
    print(f"  ./scripts/run_python.sh -m src.cli.train_pipeline \\")
    print(f"    --model naive_bayes \\")
    print(f"    --nb-alpha {best_alpha} \\")
    print(f"    --fold {args.fold}")


if __name__ == "__main__":
    main()
