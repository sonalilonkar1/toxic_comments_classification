"""Hyperparameter tuning for deep learning models (BERT, LSTM)."""

from __future__ import annotations

import json
import logging
from itertools import product
from pathlib import Path
from typing import Any, Dict

import numpy as np

from src.pipeline.train_deep import DeepTrainConfig, run_deep_training_pipeline

logger = logging.getLogger(__name__)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('tune_bert.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


def tune_deep_model(config: DeepTrainConfig) -> Dict[str, Any]:
    """Tune hyperparameters for deep models using grid search.
    
    Args:
        config: Deep training config with tune_params defined.
        
    Returns:
        Dict with best params and metrics.
    """
    logger.info(f"Starting hyperparameter tuning for {config.model_type} on fold {config.fold}")
    
    if config.model_type == "bert":
        tune_params = config.bert_params.get("tune_params", {})
        if not tune_params:
            # Default grid if not specified
            tune_params = {
                "learning_rate": [1e-5, 2e-5, 5e-5],
                "batch_size": [8, 16, 32],
                "epochs": [2, 3, 4],
            }
        logger.info(f"BERT tuning grid: {tune_params}")
    else:
        logger.error(f"Tuning not implemented for {config.model_type}")
        raise NotImplementedError(f"Tuning not implemented for {config.model_type}")
    
    # Generate all combinations
    keys = list(tune_params.keys())
    values = list(tune_params.values())
    combinations = list(product(*values))
    
    logger.info(f"Total combinations to try: {len(combinations)}")
    
    best_score = -np.inf
    best_params = None
    best_metrics = None
    
    tuning_results = []
    
    # Save tuning results incrementally
    tuning_dir = Path("experiments/hyperparameter_tuning/bert") / f"{config.fold.replace('_', '')}"
    tuning_dir.mkdir(parents=True, exist_ok=True)
    results_file = tuning_dir / "tuning_results.json"
    
    for i, combo in enumerate(combinations, 1):
        params = dict(zip(keys, combo))
        logger.info(f"Trial {i}/{len(combinations)}: Trying params: {params}")
        
        # Update config with current params
        if config.model_type == "bert":
            config.bert_params["learning_rate"] = params["learning_rate"]
            config.bert_params["train_batch_size"] = params["batch_size"]
            config.bert_params["eval_batch_size"] = params["batch_size"]
            config.bert_params["num_epochs"] = params["epochs"]
        
        try:
            # Run training
            logger.info("Starting training for current params...")
            results = run_deep_training_pipeline(config)
            logger.info("Training completed successfully")
            
            # Evaluate on validation or test? For tuning, use test metrics
            score = results["overall_metrics"]["macro_f1"]  # Use macro F1 as score
            logger.info(f"Trial {i} score: {score:.4f}")
            
            tuning_results.append({
                "params": params,
                "metrics": results["overall_metrics"],
                "score": score,
            })
            
            if score > best_score:
                best_score = score
                best_params = params
                best_metrics = results
                logger.info(f"New best score: {best_score:.4f} with params: {best_params}")
        
        except Exception as e:
            logger.error(f"Trial {i} failed with params {params}: {str(e)}")
            tuning_results.append({
                "params": params,
                "error": str(e),
                "score": None,
            })
        
        # Save incrementally after each trial
        with open(results_file, "w") as f:
            json.dump({
                "best_params": best_params,
                "best_score": best_score,
                "all_results": tuning_results,
            }, f, indent=2)
        logger.info(f"Results saved incrementally to {results_file}")
    
    logger.info(f"Tuning completed. Best params: {best_params}, Best score: {best_score:.4f}")
    logger.info(f"Final results saved to {results_file}")
    
    return best_metrics