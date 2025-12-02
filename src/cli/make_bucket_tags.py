"""CLI for generating reusable bucket tags."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.data.buckets import (
    DEFAULT_BUCKET_CONFIG_PATH,
    apply_bucket_rules,
    compile_bucket_rules,
    compute_bucket_hash,
    ensure_normalized_series,
    load_bucket_config,
)
from src.data.normalization import DEFAULT_NORMALIZATION_CONFIG_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create bucket tag caches for oversampling.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/raw/train.csv"),
        help="Dataset containing the raw comments.",
    )
    parser.add_argument(
        "--text-col",
        type=str,
        default="comment_text",
        help="Name of the text column.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("artifacts/buckets/train.parquet"),
        help="Destination for the bucket cache.",
    )
    parser.add_argument(
        "--normalized-cache",
        type=Path,
        default=None,
        help="Optional path to a precomputed normalized-text cache.",
    )
    parser.add_argument(
        "--normalization-config",
        type=Path,
        default=DEFAULT_NORMALIZATION_CONFIG_PATH,
        help="Fallback normalization profile when --normalized-cache is absent.",
    )
    parser.add_argument(
        "--bucket-config",
        type=Path,
        default=DEFAULT_BUCKET_CONFIG_PATH,
        help="YAML file describing the bucket tagging rules.",
    )
    return parser.parse_args()


def _load_normalized_from_cache(df: pd.DataFrame, cache_path: Path) -> pd.Series:
    cache_df = pd.read_parquet(cache_path)
    if "row_index" not in cache_df.columns or "normalized_text" not in cache_df.columns:
        raise ValueError("Normalized cache must contain 'row_index' and 'normalized_text'.")
    mapping: Dict[int, str] = dict(zip(cache_df["row_index"], cache_df["normalized_text"]))
    return df["row_index"].map(mapping).fillna("")


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.data_path).reset_index(drop=True)
    df["row_index"] = df.index

    if args.normalized_cache is not None:
        normalized_series = _load_normalized_from_cache(df, args.normalized_cache)
    else:
        normalized_series = ensure_normalized_series(
            df=df,
            text_col=args.text_col,
            normalized_series=None,
            normalization_config=args.normalization_config,
        )

    bucket_config = load_bucket_config(args.bucket_config)
    compiled = compile_bucket_rules(bucket_config)
    config_hash = compute_bucket_hash(bucket_config)

    raw_series = df[args.text_col].fillna("").astype(str)
    normalized_series = normalized_series.fillna("").astype(str)
    row_dicts = df.to_dict("records")

    tags = [
        apply_bucket_rules(raw, norm, row, compiled)
        for raw, norm, row in zip(raw_series, normalized_series, row_dicts)
    ]

    out_df = pd.DataFrame(
        {
            "row_index": df["row_index"].astype(int),
            "bucket_tags": [json.dumps(tag_list) for tag_list in tags],
            "bucket_count": [len(tag_list) for tag_list in tags],
            "config_hash": config_hash,
        }
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.output_path, index=False)

    meta = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "source_path": str(args.data_path.resolve()),
        "text_col": args.text_col,
        "rows": len(out_df),
        "bucket_rules_path": str(args.bucket_config.resolve()),
        "bucket_config_hash": config_hash,
        "normalized_cache": str(args.normalized_cache) if args.normalized_cache else None,
        "normalization_config": str(args.normalization_config.resolve()),
    }
    meta_path = args.output_path.with_suffix(args.output_path.suffix + ".meta.json")
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    print(
        f"[bucket-tags] Wrote {len(out_df):,} rows to {args.output_path} (hash={config_hash})."
    )


if __name__ == "__main__":
    main()
