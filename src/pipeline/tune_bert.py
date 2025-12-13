#!/usr/bin/env python3
"""Script to tune BERT model across all folds and seeds."""

import subprocess
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent

# Folds and seeds
FOLDS = ["fold1", "fold2", "fold3"]
SEEDS = [42, 43, 44]

def run_bert_training(fold: str, seed: int):
    """Run BERT training for a specific fold and seed."""
    fold_name = f"{fold}_seed{seed}"
    cmd = [
        sys.executable, "-m", "src.cli.train_pipeline",
        "--model", "bert",
        "--tune",  # Enable tuning
        "--fold", fold_name,
        "--data-path", str(PROJECT_ROOT / "data" / "raw" / "train.csv"),
        "--splits-dir", str(PROJECT_ROOT / "data" / "splits"),
        "--output-dir", str(PROJECT_ROOT / "experiments" / "train" / "bert"),
        "--seed", str(seed),
        "--labels", "toxic,severe_toxic,obscene,threat,insult,identity_hate",
        "--text-col", "comment_text",
    ]

    print(f"Running BERT training for {fold_name}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"Failed for {fold_name}")
    else:
        print(f"Completed for {fold_name}")

if __name__ == "__main__":
    for fold in FOLDS:
        for seed in SEEDS:
            run_bert_training(fold, seed)
    print("All BERT trainings completed!")