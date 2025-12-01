"""CLI entry point for the TF-IDF + logistic training pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

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
        choices=["raw", "toy", "rich"],
        help="Text normalization strategy.",
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
        help="Column containing bucket tag lists for oversampling.",
    )
    parser.add_argument(
        "--bucket-mult",
        action="append",
        default=None,
        metavar="BUCKET=FACTOR",
        help="Repeat to oversample specific buckets (e.g., rare=3).",
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
        threshold=args.threshold,
        fairness_min_support=args.fairness_min_support,
        seed=args.seed,
        bucket_col=args.bucket_col,
        bucket_multipliers=_parse_bucket_multipliers(args.bucket_mult),
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

    results = run_training_pipeline(config)
    for fold_name, payload in results.items():
        metrics = payload["overall_metrics"]
        print(
            f"Fold {fold_name}: micro F1={metrics['micro_f1']:.4f}, "
            f"macro F1={metrics['macro_f1']:.4f}, hamming_loss={metrics['hamming_loss']:.4f}"
        )


if __name__ == "__main__":
    main()
