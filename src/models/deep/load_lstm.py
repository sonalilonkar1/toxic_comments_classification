"""Helper functions to load saved LSTM model artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch

from src.features.lstm_preprocessing import LSTMPreprocessor
from src.models.deep.lstm import MultiLabelLSTM


def load_lstm_artifacts(
    model_dir: Path,
    device: Optional[str] = None,
) -> Tuple[MultiLabelLSTM, LSTMPreprocessor, Dict]:
    """
    Load saved LSTM model, preprocessor (tokenizer), and config.
    
    Args:
        model_dir: Directory containing saved artifacts (should have models/ subdirectory)
        device: Device to load model on ('cuda', 'cpu', or None for auto-detect)
    
    Returns:
        Tuple of (model, preprocessor, config_dict)
    
    Example:
        >>> model, preprocessor, config = load_lstm_artifacts(
        ...     Path("experiments/lstm/fold1_seed42/models")
        ... )
        >>> # Now you can use the model for predictions
        >>> probs, preds = predict_lstm(model, preprocessor, ["some text"])
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Load config
    config_path = model_dir / "lstm_config.json"
    if not config_path.exists():
        # Fallback to parent directory config.json
        config_path = model_dir.parent / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found in {model_dir} or parent")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    # Extract LSTM-specific config
    if "model_architecture" in config:
        # New format (lstm_config.json)
        arch = config["model_architecture"]
        vocab_size = config["preprocessing"]["vocab_size"]
        embedding_dim = arch["embedding_dim"]
        hidden_dim = arch["hidden_dim"]
        num_layers = arch["num_layers"]
        num_labels = arch["num_labels"]
        dropout = arch["dropout"]
        bidirectional = arch["bidirectional"]
    else:
        # Fallback to config.json format
        lstm_params = config.get("lstm_params", {})
        vocab_size = lstm_params.get("vocab_size", 10000)
        embedding_dim = lstm_params.get("embedding_dim", 128)
        hidden_dim = lstm_params.get("hidden_dim", 128)
        num_layers = lstm_params.get("num_layers", 2)
        num_labels = len(config.get("label_cols", []))
        dropout = lstm_params.get("dropout", 0.3)
        bidirectional = lstm_params.get("bidirectional", True)
    
    # Load preprocessor (tokenizer)
    preprocessor_path = model_dir / "lstm_preprocessor.pkl"
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")
    preprocessor = LSTMPreprocessor.load(preprocessor_path)
    
    # Load model
    # Try full model first (easier to load)
    model_path = model_dir / "lstm_model_full.pt"
    if model_path.exists():
        model = torch.load(model_path, map_location=device)
        model = model.to(device)
    else:
        # Fallback to state dict + model info
        state_dict_path = model_dir / "lstm_model.pt"
        model_info_path = model_dir / "lstm_model_info.pt"
        
        if not state_dict_path.exists():
            raise FileNotFoundError(f"Model file not found: {state_dict_path}")
        
        # Reconstruct model
        model = MultiLabelLSTM(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_labels=num_labels,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        model.load_state_dict(torch.load(state_dict_path, map_location=device))
        model = model.to(device)
    
    model.eval()  # Set to evaluation mode
    
    return model, preprocessor, config


def load_lstm_for_prediction(
    experiment_dir: Path,
    fold_name: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[MultiLabelLSTM, LSTMPreprocessor, Dict]:
    """
    Convenience function to load LSTM artifacts from experiment directory.
    
    Args:
        experiment_dir: Root experiment directory (e.g., "experiments/lstm")
        fold_name: Optional specific fold name (if None, finds first fold)
        device: Device to load model on
    
    Returns:
        Tuple of (model, preprocessor, config_dict)
    
    Example:
        >>> model, preprocessor, config = load_lstm_for_prediction(
        ...     Path("experiments/lstm"),
        ...     fold_name="fold1_seed42"
        ... )
    """
    experiment_dir = Path(experiment_dir)
    
    if fold_name:
        # Look for specific fold
        fold_dirs = list(experiment_dir.glob(f"{fold_name}*"))
        if not fold_dirs:
            raise FileNotFoundError(f"Fold {fold_name} not found in {experiment_dir}")
        fold_dir = sorted(fold_dirs)[-1]  # Get most recent if multiple
    else:
        # Find first available fold
        fold_dirs = list(experiment_dir.glob("fold*"))
        if not fold_dirs:
            raise FileNotFoundError(f"No folds found in {experiment_dir}")
        fold_dir = sorted(fold_dirs)[-1]  # Get most recent
    
    model_dir = fold_dir / "models"
    return load_lstm_artifacts(model_dir, device=device)


__all__ = [
    "load_lstm_artifacts",
    "load_lstm_for_prediction",
]




