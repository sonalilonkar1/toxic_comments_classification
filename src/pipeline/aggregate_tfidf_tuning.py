#!/usr/bin/env python3
"""Aggregate TF-IDF tuning results into a summary JSON."""

import json
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

FOLDS = ["fold1", "fold2", "fold3"]
SEEDS = [42, 43, 44]
MODELS = ["logistic", "svm", "random_forest"]

def aggregate_tuning_results():
    """Aggregate tuning results from all folds and seeds."""
    summary = {}
    
    for fold in FOLDS:
        for seed in SEEDS:
            fold_seed = f"{fold}_seed{seed}"
            tuning_file = PROJECT_ROOT / "experiments" / "hyperparameter_tuning" / "tfidf_tuning" / fold_seed / "tuning_results.json"
            
            if not tuning_file.exists():
                print(f"Warning: Tuning results not found for {fold_seed}")
                continue
            
            with open(tuning_file, "r") as f:
                data = json.load(f)
            
            summary[fold_seed] = data
    
    # Save summary
    summary_file = PROJECT_ROOT / "experiments" / "hyperparameter_tuning" / "tfidf_tuning" / "tuning_results_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Tuning results summary saved to {summary_file}")

if __name__ == "__main__":
    aggregate_tuning_results()