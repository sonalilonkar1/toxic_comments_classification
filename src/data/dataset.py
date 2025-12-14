"""Data loading utilities."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def load_fold_frames(
    seed: int,
    data_path: Path,
    splits_dir: Path,
    fold: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, pd.DataFrame]], List[str], List[str]]:
    """Load raw data and split it into fold frames.

    Args:
        seed: Random seed (unused here but kept for API compatibility).
        data_path: Path to the raw CSV data.
        splits_dir: Directory containing JSON split indices.
        fold: Optional specific fold to load (returns dict with just that fold).

    Returns:
        Tuple of:
            - base_df: The full raw dataframe.
            - fold_frames: Dict mapping fold_name -> {train, dev, test} dataframes.
            - identity_cols: List of identity columns found in the data.
            - label_cols: List of label columns found in the data (standard Jigsaw labels).
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")

    df = pd.read_csv(data_path)
    
    # Ensure row_index exists for joining
    if "id" in df.columns and "row_index" not in df.columns:
         # Use existing index as row_index if not present
         df["row_index"] = df.index

    # Identity columns from Jigsaw 2.0 (if present)
    identity_cols = [
        "male", "female", "transgender", "other_gender", "heterosexual",
        "homosexual_gay_or_lesbian", "christian", "jewish", "muslim", "hindu",
        "buddhist", "atheist", "intellectual_or_learning_disability",
        "psychiatric_or_mental_illness", "physical_disability", "white", "black",
        "asian", "latino", "other_race_or_ethnicity"
    ]
    found_identities = [c for c in identity_cols if c in df.columns]

    # Standard toxic labels
    label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    found_labels = [c for c in label_cols if c in df.columns]

    fold_frames = {}
    
    # List all split files
    split_files = sorted(list(splits_dir.glob("fold*.json")))
    if not split_files:
        raise ValueError(f"No split files found in {splits_dir}. Run make_splits.py first.")

    for split_file in split_files:
        fold_name = split_file.stem  # e.g., "fold1"
        if fold and fold != fold_name:
            continue
            
        with open(split_file, "r") as f:
            indices = json.load(f)
            
        fold_frames[fold_name] = {
            "train": df.iloc[indices["train"]].copy(),
            "dev": df.iloc[indices["dev"]].copy(),
            "test": df.iloc[indices["test"]].copy(),
        }

    return df, fold_frames, found_identities, found_labels




