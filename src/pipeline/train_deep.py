"""Training pipeline for Deep Learning models (BERT, LSTM)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Any

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.dataset import load_fold_frames
from src.data.normalization import (
    DEFAULT_NORMALIZATION_CONFIG_PATH,
    build_normalizer,
    load_normalization_config,
)

DEFAULT_LSTM_CONFIG_PATH = Path("configs/lstm.yaml")
from src.data.preprocess import rich_normalize, toy_normalize
from src.models.deep.bert import train_bert_model
from src.models.deep.lstm import train_lstm_model, predict_lstm
from src.features.lstm_preprocessing import (
    LSTMPreprocessor,
    load_glove_embeddings,
    load_word2vec_embeddings,
    preprocess_texts_for_lstm,
)
from src.utils.metrics import (
    compute_fairness_slices,
    compute_multilabel_metrics,
    compute_top_k_metrics,
    find_precision_thresholds,
    probs_to_preds,
)

DEFAULT_LABELS: List[str] = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

DEFAULT_BERT_PARAMS: Dict[str, Any] = {
    "model_name": "distilbert-base-uncased",
    "max_length": 128,
    "batch_size": 16,
    "epochs": 3,
    "learning_rate": 2e-5,
    "fp16": False,
}

DEFAULT_LSTM_PARAMS: Dict[str, Any] = {
    "vocab_size": 10000,
    "max_length": 200,
    "embedding_dim": 128,
    "hidden_dim": 128,
    "num_layers": 2,
    "dropout": 0.3,
    "bidirectional": True,
    "batch_size": 32,
    "epochs": 10,
    "learning_rate": 0.001,
    "embedding_path": None,  # Path to pre-trained embeddings (GloVe/Word2Vec)
    "freeze_embeddings": False,
    "resume_from": None,  # Path to checkpoint to resume from
    "checkpoint_interval": 1,  # Save checkpoint every N epochs
}

@dataclass
class DeepTrainConfig:
    """Configuration for Deep Learning training pipeline."""
    
    data_path: Path = Path("data/raw/train.csv")
    splits_dir: Path = Path("data/splits")
    output_dir: Path = Path("experiments/deep_learning")
    fold: Optional[str] = None
    label_cols: Optional[List[str]] = None
    text_col: str = "comment_text"
    
    # Preprocessing
    normalization: str = "toy"
    normalization_config: Optional[Path] = DEFAULT_NORMALIZATION_CONFIG_PATH
    
    # Model config
    model_type: str = "bert" # 'bert' or 'lstm'
    bert_params: Dict[str, Any] = field(default_factory=lambda: DEFAULT_BERT_PARAMS.copy())
    lstm_params: Dict[str, Any] = field(default_factory=lambda: DEFAULT_LSTM_PARAMS.copy())
    lstm_config_path: Optional[Path] = DEFAULT_LSTM_CONFIG_PATH  # Path to LSTM YAML config
    loss_type: str = "bce"
    loss_params: Dict[str, float] = field(default_factory=dict)
    
    # Evaluation
    target_precision: Optional[float] = 0.90
    top_k: int = 1000
    fairness_min_support: int = 50
    seed: int = 42

    def __post_init__(self) -> None:
        self.data_path = Path(self.data_path)
        self.splits_dir = Path(self.splits_dir)
        self.output_dir = Path(self.output_dir)
        if self.normalization_config:
            self.normalization_config = Path(self.normalization_config)
        if self.lstm_config_path:
            self.lstm_config_path = Path(self.lstm_config_path)
        
        # Load LSTM config from YAML if model_type is lstm and config file exists
        if self.model_type == "lstm" and self.lstm_config_path:
            # Resolve path relative to project root if not absolute
            if not self.lstm_config_path.is_absolute():
                # Try to find project root (go up from src/pipeline)
                project_root = Path(__file__).resolve().parents[2]
                config_path = project_root / self.lstm_config_path
            else:
                config_path = self.lstm_config_path
            
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    lstm_config = yaml.safe_load(f)
                
                # Update lstm_params with values from YAML (preserve defaults if not in YAML)
                self.lstm_params.update({
                    "vocab_size": lstm_config.get("vocab_size", self.lstm_params.get("vocab_size", 10000)),
                    "max_length": lstm_config.get("max_length", self.lstm_params.get("max_length", 200)),
                    "embedding_dim": lstm_config.get("embedding_dim", self.lstm_params.get("embedding_dim", 128)),
                    "hidden_dim": lstm_config.get("hidden_dim", self.lstm_params.get("hidden_dim", 128)),
                    "num_layers": lstm_config.get("num_layers", self.lstm_params.get("num_layers", 2)),
                    "dropout": lstm_config.get("dropout", self.lstm_params.get("dropout", 0.3)),
                    "bidirectional": lstm_config.get("bidirectional", self.lstm_params.get("bidirectional", True)),
                    "batch_size": lstm_config.get("batch_size", self.lstm_params.get("batch_size", 32)),
                    "epochs": lstm_config.get("epochs", self.lstm_params.get("epochs", 10)),
                    "learning_rate": lstm_config.get("learning_rate", self.lstm_params.get("learning_rate", 0.001)),
                    "embedding_path": lstm_config.get("embedding_path", self.lstm_params.get("embedding_path")),
                    "freeze_embeddings": lstm_config.get("freeze_embeddings", self.lstm_params.get("freeze_embeddings", False)),
                    "resume_from": lstm_config.get("resume_from", self.lstm_params.get("resume_from")),
                    "checkpoint_interval": lstm_config.get("checkpoint_interval", self.lstm_params.get("checkpoint_interval", 1)),
                })

    def resolve_label_cols(self, df: pd.DataFrame) -> List[str]:
        if self.label_cols:
            return self.label_cols
        inferred = [col for col in DEFAULT_LABELS if col in df.columns]
        if not inferred:
            raise ValueError("No label columns found.")
        self.label_cols = inferred
        return inferred


def run_deep_training_pipeline(config: DeepTrainConfig) -> Dict[str, Any]:
    """Execute Deep Learning training pipeline."""
    
    # Load Data
    base_df, fold_frames, identity_cols, _ = load_fold_frames(
        seed=config.seed,
        data_path=config.data_path,
        splits_dir=config.splits_dir,
    )
    label_cols = config.resolve_label_cols(base_df)
    
    target_folds: Iterable[str]
    if config.fold:
        if config.fold not in fold_frames:
            raise ValueError(f"Fold {config.fold} not found.")
        target_folds = [config.fold]
    else:
        target_folds = sorted(fold_frames.keys())

    # Resolve Normalizer
    if config.normalization == "config" and config.normalization_config:
        norm_profile = load_normalization_config(config.normalization_config)
        normalizer = build_normalizer(norm_profile)
    elif config.normalization == "rich":
        normalizer = rich_normalize
    else:
        normalizer = toy_normalize

    results = {}

    for fold_name in target_folds:
        print(f"Starting Fold: {fold_name}")
        fold_dir = _prepare_fold_dir(config.output_dir, fold_name, config)
        
        metrics = _train_single_deep_fold(
            fold_name=fold_name,
            fold_splits=fold_frames[fold_name],
            identity_cols=identity_cols,
            label_cols=label_cols,
            fold_dir=fold_dir,
            config=config,
            normalizer=normalizer,
        )
        results[fold_name] = metrics

    # Save summary
    summary_path = config.output_dir / "summary_metrics.json"
    summary_payload = {k: v["overall_metrics"] for k, v in results.items()}
    with open(summary_path, "w") as f:
        json.dump(summary_payload, f, indent=2)

    return results


def _train_single_deep_fold(
    fold_name: str,
    fold_splits: Dict[str, pd.DataFrame],
    identity_cols: List[str],
    label_cols: List[str],
    fold_dir: Path,
    config: DeepTrainConfig,
    normalizer: Any,
) -> Dict[str, Any]:
    
    # Data Prep
    train_df = fold_splits["train"].reset_index(drop=True)
    dev_df = fold_splits["dev"].reset_index(drop=True)
    test_df = fold_splits["test"].reset_index(drop=True)
    
    # Normalize Text
    # Note: Deep models handle raw text reasonably well, but we still apply 
    # the requested normalization (e.g. for obfuscation)
    X_train = train_df[config.text_col].apply(lambda x: normalizer(str(x))).tolist()
    X_dev = dev_df[config.text_col].apply(lambda x: normalizer(str(x))).tolist()
    X_test = test_df[config.text_col].apply(lambda x: normalizer(str(x))).tolist()
    
    y_train = train_df[label_cols].values.astype(int)
    y_dev = dev_df[label_cols].values.astype(int)
    y_test = test_df[label_cols].values.astype(int)
    
    # Train Model
    if config.model_type == "bert":
        trainer, tokenizer = train_bert_model(
            model_name=config.bert_params["model_name"],
            X_train_text=X_train,
            y_train=y_train,
            X_val_text=X_dev,
            y_val=y_dev,
            output_dir=str(fold_dir / "checkpoints"),
            num_labels=len(label_cols),
            max_length=config.bert_params.get("max_length", 128),
            batch_size=config.bert_params.get("batch_size", 16),
            epochs=config.bert_params.get("epochs", 3),
            learning_rate=config.bert_params.get("learning_rate", 2e-5),
            fp16=config.bert_params.get("fp16", False),
            seed=config.seed,
            loss_type=config.loss_type,
            loss_params=config.loss_params,
        )
        
        # Predict on Dev to set thresholds
        # Note: Trainer.predict returns named tuple
        dev_out = trainer.predict(trainer.eval_dataset)
        dev_logits = dev_out.predictions
        # Apply sigmoid
        dev_probs_arr = 1.0 / (1.0 + np.exp(-dev_logits))
        
        # Create dict of probs for metric helpers
        dev_probs = {label: dev_probs_arr[:, i] for i, label in enumerate(label_cols)}
        
        # Determine thresholds
        if config.target_precision:
            thresholds = find_precision_thresholds(
                y_dev, dev_probs, label_cols, target_precision=config.target_precision
            )
        else:
            thresholds = 0.5

        # Predict on Test
        # We need to re-tokenize test data using the *trained* tokenizer (though for BERT it's static)
        test_encodings = tokenizer(
            X_test, truncation=True, padding=True, max_length=config.bert_params.get("max_length", 128)
        )
        from src.models.deep.bert import ToxicDataset
        test_dataset = ToxicDataset(test_encodings) # No labels needed for prediction, but nice to have
        
        test_out = trainer.predict(test_dataset)
        test_logits = test_out.predictions
        test_probs_arr = 1.0 / (1.0 + np.exp(-test_logits))
        test_probs = {label: test_probs_arr[:, i] for i, label in enumerate(label_cols)}
        
        # Save model
        trainer.save_model(str(fold_dir / "final_model"))
        tokenizer.save_pretrained(str(fold_dir / "final_model"))

    elif config.model_type == "lstm":
        # Preprocess texts for LSTM (tokenize and pad)
        preprocessor = LSTMPreprocessor(
            vocab_size=config.lstm_params.get("vocab_size", 10000),
            max_length=config.lstm_params.get("max_length", 200),
            oov_token="<OOV>",
            padding="post",
            truncating="post",
        )
        
        # Fit preprocessor on training data
        X_train_seq = preprocessor.fit_transform(X_train)
        X_dev_seq = preprocessor.transform(X_dev)
        X_test_seq = preprocessor.transform(X_test)
        
        # Load pre-trained embeddings if provided
        embedding_matrix = None
        embedding_path = config.lstm_params.get("embedding_path")
        if embedding_path and Path(embedding_path).exists():
            embedding_path = Path(embedding_path)
            word_index = preprocessor.get_word_index()
            
            # Try to detect embedding type from file extension or content
            if embedding_path.suffix == ".txt" or "glove" in embedding_path.name.lower():
                # Assume GloVe format
                embedding_dim = config.lstm_params.get("embedding_dim", 128)
                embedding_matrix = load_glove_embeddings(
                    embedding_path, word_index, embedding_dim=embedding_dim
                )
            elif "word2vec" in embedding_path.name.lower() or embedding_path.suffix in [".bin", ".model"]:
                # Assume Word2Vec format
                embedding_matrix, embedding_dim = load_word2vec_embeddings(
                    embedding_path, word_index, binary=True
                )
                # Update embedding_dim in config if it was auto-detected
                config.lstm_params["embedding_dim"] = embedding_dim
        
        # Train LSTM model
        resume_from = config.lstm_params.get("resume_from")
        if resume_from:
            resume_from = str(Path(resume_from))  # Convert to string path
        
        model, preprocessor = train_lstm_model(
            X_train_text=X_train,  # Already normalized
            y_train=y_train,
            X_val_text=X_dev,  # Already normalized
            y_val=y_dev,
            preprocessor=preprocessor,
            output_dir=str(fold_dir / "checkpoints"),
            num_labels=len(label_cols),
            embedding_dim=config.lstm_params.get("embedding_dim", 128),
            hidden_dim=config.lstm_params.get("hidden_dim", 128),
            num_layers=config.lstm_params.get("num_layers", 2),
            dropout=config.lstm_params.get("dropout", 0.3),
            bidirectional=config.lstm_params.get("bidirectional", True),
            batch_size=config.lstm_params.get("batch_size", 32),
            epochs=config.lstm_params.get("epochs", 10),
            learning_rate=config.lstm_params.get("learning_rate", 0.001),
            embedding_matrix=embedding_matrix,
            freeze_embeddings=config.lstm_params.get("freeze_embeddings", False),
            seed=config.seed,
            resume_from=resume_from,
            checkpoint_interval=config.lstm_params.get("checkpoint_interval", 1),
            loss_type=config.loss_type,
            loss_params=config.loss_params,
        )
        
        # Save preprocessor (tokenizer) and model
        model_dir = fold_dir / "models"
        model_dir.mkdir(exist_ok=True)
        preprocessor.save(model_dir / "lstm_preprocessor.pkl")
        
        # Save full model (not just state dict) for easier loading
        torch.save(model, model_dir / "lstm_model_full.pt")
        
        # Save LSTM-specific config JSON with all hyperparameters
        lstm_config_path = model_dir / "lstm_config.json"
        lstm_config = {
            "model_type": "lstm",
            "fold": fold_name,
            "preprocessing": {
                "vocab_size": config.lstm_params.get("vocab_size", 10000),
                "max_length": config.lstm_params.get("max_length", 200),
            },
            "model_architecture": {
                "embedding_dim": config.lstm_params.get("embedding_dim", 128),
                "hidden_dim": config.lstm_params.get("hidden_dim", 128),
                "num_layers": config.lstm_params.get("num_layers", 2),
                "dropout": config.lstm_params.get("dropout", 0.3),
                "bidirectional": config.lstm_params.get("bidirectional", True),
                "num_labels": len(label_cols),
            },
            "training": {
                "batch_size": config.lstm_params.get("batch_size", 32),
                "epochs": config.lstm_params.get("epochs", 10),
                "learning_rate": config.lstm_params.get("learning_rate", 0.001),
                "loss_type": config.loss_type,
                "loss_params": config.loss_params,
            },
            "embeddings": {
                "embedding_path": str(config.lstm_params.get("embedding_path")) if config.lstm_params.get("embedding_path") else None,
                "freeze_embeddings": config.lstm_params.get("freeze_embeddings", False),
            },
            "label_cols": label_cols,
            "seed": config.seed,
        }
        with open(lstm_config_path, "w", encoding="utf-8") as f:
            json.dump(lstm_config, f, indent=2)
        
        # Predict on Dev to set thresholds
        dev_probs_arr, _ = predict_lstm(model, preprocessor, X_dev)
        dev_probs = {label: dev_probs_arr[:, i] for i, label in enumerate(label_cols)}
        
        # Determine thresholds
        if config.target_precision:
            thresholds = find_precision_thresholds(
                y_dev, dev_probs, label_cols, target_precision=config.target_precision
            )
        else:
            thresholds = 0.5
        
        # Predict on Test
        test_probs_arr, _ = predict_lstm(model, preprocessor, X_test)
        test_probs = {label: test_probs_arr[:, i] for i, label in enumerate(label_cols)}
        
        # Save model state dict (for compatibility)
        torch.save(model.state_dict(), model_dir / "lstm_model.pt")
        
        # Also save model architecture info for reconstruction
        model_info = {
            "vocab_size": preprocessor.get_vocab_size(),
            "embedding_dim": config.lstm_params.get("embedding_dim", 128),
            "hidden_dim": config.lstm_params.get("hidden_dim", 128),
            "num_layers": config.lstm_params.get("num_layers", 2),
            "num_labels": len(label_cols),
            "dropout": config.lstm_params.get("dropout", 0.3),
            "bidirectional": config.lstm_params.get("bidirectional", True),
        }
        torch.save(model_info, model_dir / "lstm_model_info.pt")
    else:
        raise ValueError(f"Unknown deep model type: {config.model_type}")

    # Calculate Metrics
    y_test_pred = probs_to_preds(test_probs, threshold=thresholds)
    
    overall_metrics, per_label_df = compute_multilabel_metrics(y_test, y_test_pred, label_cols, prob_dict=test_probs)
    top_k_metrics = compute_top_k_metrics(y_test, test_probs, label_cols, k=config.top_k)
    overall_metrics.update(top_k_metrics)
    
    fairness_df = compute_fairness_slices(
        test_df, y_test, y_test_pred, label_cols, identity_cols, min_support=config.fairness_min_support
    )
    
    # Save Artifacts
    _persist_deep_artifacts(
        fold_dir=fold_dir,
        fold_name=fold_name,
        config=config,
        label_cols=label_cols,
        overall_metrics=overall_metrics,
        per_label_df=per_label_df,
        fairness_df=fairness_df,
        test_df=test_df,
        test_probs=test_probs,
        y_test_pred=y_test_pred,
    )
    
    return {
        "overall_metrics": overall_metrics,
        "per_label_path": fold_dir / "per_label_metrics.csv",
    }


def _prepare_fold_dir(base_dir: Path, fold_name: str, config: DeepTrainConfig) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_name = (
        f"{fold_name}-seed{config.seed}-norm{config.normalization}-"
        f"model{config.model_type}-{timestamp}"
    )
    fold_dir = base_dir / dir_name
    fold_dir.mkdir(parents=True, exist_ok=True)
    return fold_dir


def _convert_paths_to_strings(obj):
    """Recursively convert Path objects to strings in a dictionary or list."""
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: _convert_paths_to_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_paths_to_strings(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_paths_to_strings(item) for item in obj)
    else:
        return obj


def _persist_deep_artifacts(
    fold_dir: Path,
    fold_name: str,
    config: DeepTrainConfig,
    label_cols: List[str],
    overall_metrics: Dict[str, float],
    per_label_df: pd.DataFrame,
    fairness_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_probs: Dict[str, np.ndarray],
    y_test_pred: np.ndarray,
) -> None:
    overall_path = fold_dir / "overall_metrics.json"
    per_label_path = fold_dir / "per_label_metrics.csv"
    fairness_path = fold_dir / "fairness_slices.csv"
    preds_path = fold_dir / "test_predictions.csv"
    config_path = fold_dir / "config.json"

    with open(overall_path, "w") as f:
        json.dump(overall_metrics, f, indent=2)
    
    per_label_df.to_csv(per_label_path, index=False)
    
    if not fairness_df.empty:
        fairness_df.to_csv(fairness_path, index=False)
        
    # Save Predictions
    preds_df = test_df.copy()
    for i, label in enumerate(label_cols):
        preds_df[f"{label}_prob"] = test_probs[label]
        preds_df[f"{label}_pred"] = y_test_pred[:, i]
        # Ground truth is already in test_df copy, so no extra work needed
    preds_df.to_csv(preds_path, index=False)
    
    # Save Config (includes all config including lstm_params)
    payload = asdict(config)
    # Convert all Path objects to strings recursively
    payload = _convert_paths_to_strings(payload)
    # Explicitly convert Path fields (in case they weren't caught)
    payload["data_path"] = str(config.data_path)
    payload["splits_dir"] = str(config.splits_dir)
    payload["output_dir"] = str(config.output_dir)
    if payload.get("normalization_config"):
        payload["normalization_config"] = str(payload["normalization_config"])
    if payload.get("lstm_config_path"):
        payload["lstm_config_path"] = str(payload["lstm_config_path"])
    payload["fold_name"] = fold_name
    payload["label_cols"] = label_cols
    
    # Ensure lstm_params are properly serialized (already converted by _convert_paths_to_strings)
    if config.model_type == "lstm" and "lstm_params" in payload:
        lstm_params = payload["lstm_params"]
        # Double-check Path objects are strings
        if "embedding_path" in lstm_params and lstm_params["embedding_path"]:
            lstm_params["embedding_path"] = str(lstm_params["embedding_path"])
        if "resume_from" in lstm_params and lstm_params["resume_from"]:
            lstm_params["resume_from"] = str(lstm_params["resume_from"])
        payload["lstm_params"] = lstm_params
    
    if config.normalization_config:
        payload["normalization_config"] = str(config.normalization_config)
        
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
