"""Script to tune BERT hyperparameters for a specific fold and seed."""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Tune BERT hyperparameters for a specific fold and seed.")
    parser.add_argument("--fold", type=str, required=True, help="Fold to tune (e.g., fold1)")
    parser.add_argument("--seed", type=int, required=True, help="Seed to tune (e.g., 42)")
    args = parser.parse_args()

    fold = args.fold
    seed = args.seed
    fold_seed = f"{fold}_seed{seed}"

    # Run the CLI for tuning
    cmd = [
        sys.executable, "-m", "src.cli.train_pipeline",
        "--model", "bert",
        "--fold", fold_seed,
        "--tune",
        "--output-dir", str(PROJECT_ROOT / "experiments" / "train" / "bert"),
    ]

    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"Tuning failed for {fold_seed}")
        sys.exit(1)


if __name__ == "__main__":
    main()