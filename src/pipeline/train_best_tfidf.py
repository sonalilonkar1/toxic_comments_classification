#!/usr/bin/env python3
"""Train TF-IDF models with best hyperparameters from tuning on all folds and seeds."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Models and their config files
MODELS = {
    "logistic": "configs/logistic_regression.yaml",
    "svm": "configs/svm.yaml",
    "random_forest": "configs/random_forest.yaml"
}

FOLDS = ["fold1", "fold2", "fold3"]
SEEDS = [42, 43, 44]

def load_best_params(model: str, fold_seed: str) -> dict:
    """Load best params from tuning summary."""
    tuning_file = PROJECT_ROOT / "experiments" / "hyperparameter_tuning" / "tfidf_tuning" / fold_seed / "tuning_summary.json"
    if not tuning_file.exists():
        raise FileNotFoundError(f"Tuning summary not found: {tuning_file}")
    
    with open(tuning_file, "r") as f:
        data = json.load(f)
    
    # For TF-IDF, best_params is nested under model and then fold
    model_data = data.get(model, {}).get(fold_seed, {})
    best_params_nested = model_data.get("best_params", {})
    best_scores = model_data.get("best_scores", {})
    if not best_params_nested or not best_scores:
        raise ValueError(f"No best params or scores found for {model} on {fold_seed}")
    
    # Find the label with the highest score
    best_label = max(best_scores, key=best_scores.get)
    return best_params_nested[best_label]

def update_config_with_params(config_path: str, param_dict: dict) -> str:
    """Update YAML config with best params and return temp file path."""
    import tempfile
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Ensure model.params exists
    if "model" not in config:
        config["model"] = {}
    if "params" not in config["model"]:
        config["model"]["params"] = {}
    
    # Update params based on model
    for key, value in param_dict.items():
        config["model"]["params"][key] = value
    
    # Write to temp file
    import io
    yaml_str = yaml.dump(config)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_str)
        temp_path = f.name
    return temp_path

def run_training(model: str, fold_seed: str, config_path: str, params: dict):
    """Run training for a model on a fold_seed."""
    output_dir = PROJECT_ROOT / "experiments" / "train" / "tfidf" / fold_seed
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        sys.executable, "-m", "src.cli.train_pipeline",
        "--model", model,
        "--fold", fold_seed,
        "--data-path", str(PROJECT_ROOT / "data" / "raw" / "train.csv"),
        "--splits-dir", str(PROJECT_ROOT / "data" / "splits"),
        "--output-dir", str(output_dir),
        "--text-col", "comment_text",
        "--config", config_path,
    ] + ["--labels"] + "toxic,severe_toxic,obscene,threat,insult,identity_hate".split(",")
    
    print(f"Running {model} training for {fold_seed} with params: {params}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"Failed for {model} on {fold_seed}")
    else:
        print(f"Completed for {model} on {fold_seed}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TF-IDF models with best params")
    parser.add_argument("--model", choices=list(MODELS.keys()), help="Specific model to train (default: all)")
    args = parser.parse_args()
    
    models_to_run = [args.model] if args.model else list(MODELS.keys())
    
    for model in models_to_run:
        config_file = MODELS[model]
        print(f"Starting training for model: {model}")
        for fold in FOLDS:
            for seed in SEEDS:
                fold_seed = f"{fold}_seed{seed}"
                try:
                    # Load best params
                    best_params = load_best_params(model, fold_seed)
                    print(f"Best params for {model} on {fold_seed}: {best_params}")
                    
                    # Update config
                    temp_config = update_config_with_params(config_file, best_params)
                    
                    # Run training
                    run_training(model, fold_seed, temp_config, best_params)
                    
                    # Clean up temp file
                    Path(temp_config).unlink()
                    
                except Exception as e:
                    print(f"Error for {model} on {fold_seed}: {e}")
        print(f"Completed training for model: {model}")
    
    print("All requested TF-IDF trainings completed!")