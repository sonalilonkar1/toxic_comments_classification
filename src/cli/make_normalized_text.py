"""CLI to precompute normalized text caches based on a YAML profile."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.normalization import (
    DEFAULT_NORMALIZATION_CONFIG_PATH,
    build_normalizer,
    compute_config_hash,
    load_normalization_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize normalized text for downstream pipelines."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/raw/train.csv"),
        help="Source CSV containing the raw comments.",
    )
    parser.add_argument(
        "--text-col",
        type=str,
        default="comment_text",
        help="Column containing the raw text.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("artifacts/normalized/train.parquet"),
        help="Destination parquet file for the normalized text cache.",
    )
    parser.add_argument(
        "--normalization-config",
        type=Path,
        default=DEFAULT_NORMALIZATION_CONFIG_PATH,
        help="YAML profile describing the normalization steps.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_payload = load_normalization_config(args.normalization_config)
    normalizer = build_normalizer(config_payload)
    config_hash = compute_config_hash(config_payload)

    df = pd.read_csv(args.data_path).reset_index(drop=True)
    df["row_index"] = df.index
    normalized = (
        df[args.text_col].fillna("").astype(str).apply(normalizer)
    )

    out_df = pd.DataFrame(
        {
            "row_index": df["row_index"].astype(int),
            "normalized_text": normalized.astype(str),
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
        "config_path": str(args.normalization_config.resolve()),
        "config_hash": config_hash,
    }
    meta_path = args.output_path.with_suffix(args.output_path.suffix + ".meta.json")
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    print(
        f"[normalized-text] Wrote {len(out_df):,} rows to {args.output_path} (hash={config_hash})."
    )


if __name__ == "__main__":
    main()
