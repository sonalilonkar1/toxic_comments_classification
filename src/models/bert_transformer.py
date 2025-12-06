"""HuggingFace Transformer trainers for multi-label toxic comment classification."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

logger = logging.getLogger(__name__)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def _compute_metrics(eval_pred):
    """Compute metrics for multi-label classification."""
    predictions, labels = eval_pred
    predictions = _sigmoid(predictions)
    
    # Calculate metrics per label
    f1_micro = f1_score(labels, predictions >= 0.5, average='micro')
    f1_macro = f1_score(labels, predictions >= 0.5, average='macro')
    
    return {
        'micro_f1': f1_micro,
        'macro_f1': f1_macro,
    }


class _MultilabelTextDataset(Dataset):
    """Simple dataset wrapper that tokenizes texts once for efficiency."""

    def __init__(
        self,
        texts: Sequence[str],
        labels: np.ndarray,
        tokenizer,
        max_length: int,
    ) -> None:
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        self.labels = labels.astype(np.float32)

    def __len__(self) -> int:  # noqa: D401 - trivial override
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item


@dataclass
class BertTrainingResult:
    """Container for transformer training artifacts."""

    test_probs: Dict[str, np.ndarray]
    dev_probs: Dict[str, np.ndarray]  # Added for threshold tuning
    model_path: Path
    tokenizer_path: Path
    trainer_metrics: Dict[str, float]
    trainer: Trainer  # Added to allow further predictions


def train_multilabel_bert(
    train_texts: Sequence[str],
    train_labels: np.ndarray,
    dev_texts: Sequence[str],
    dev_labels: np.ndarray,
    test_texts: Sequence[str],
    label_cols: List[str],
    model_dir: Path,
    params: Dict[str, object],
    seed: int = 42,
) -> BertTrainingResult:
    """Fine-tune a transformer using HuggingFace Trainer for multi-label toxic comments."""

    logger.info(f"Starting BERT training with model: {params.get('model_name', 'bert-base-uncased')}")
    logger.info(f"Training params: lr={params.get('learning_rate', 2e-5)}, batch_size={params.get('train_batch_size', 8)}, epochs={params.get('num_epochs', 3.0)}")
    logger.info(f"Dataset sizes: train={len(train_texts)}, dev={len(dev_texts)}, test={len(test_texts)}")
    logger.info(f"Labels: {label_cols}")

    model_name = str(params.get("model_name", "bert-base-uncased"))
    max_length = int(params.get("max_length", 256))
    train_batch_size = int(params.get("train_batch_size", 8))
    eval_batch_size = int(params.get("eval_batch_size", 8))
    learning_rate = float(params.get("learning_rate", 2e-5))
    weight_decay = float(params.get("weight_decay", 0.01))
    num_epochs = float(params.get("num_epochs", 3.0))
    warmup_ratio = float(params.get("warmup_ratio", 0.06))
    grad_accum = int(params.get("gradient_accumulation_steps", 1))
    fp16_requested = bool(params.get("fp16", False))
    logging_steps = int(params.get("logging_steps", 50))
    save_total_limit = int(params.get("save_total_limit", 1))

    model_dir = Path(model_dir)
    checkpoints_dir = model_dir / "checkpoints"
    tokenizer_dir = model_dir / "tokenizer"
    final_model_dir = model_dir / "model"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading tokenizer and model from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_cols),
        problem_type="multi_label_classification",
    )

    set_seed(seed)
    logger.info(f"Set random seed to {seed}")

    logger.info("Tokenizing datasets...")
    train_dataset = _MultilabelTextDataset(train_texts, train_labels, tokenizer, max_length)
    dev_dataset = _MultilabelTextDataset(dev_texts, dev_labels, tokenizer, max_length)
    zero_test_labels = np.zeros((len(test_texts), len(label_cols)), dtype=np.float32)
    test_dataset = _MultilabelTextDataset(test_texts, zero_test_labels, tokenizer, max_length)

    logger.info("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=grad_accum,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=logging_steps,
        seed=seed,
        report_to=[]
        if not params.get("report_to")
        else params.get("report_to"),
        fp16=bool(fp16_requested and torch.cuda.is_available()),
    )

    if fp16_requested and not torch.cuda.is_available():
        logger.warning("FP16 requested but CUDA not available, using FP32")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        compute_metrics=_compute_metrics,
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training completed")

    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(tokenizer_dir))
    logger.info(f"Model and tokenizer saved to {final_model_dir} and {tokenizer_dir}")

    logger.info("Predicting on dev set...")
    dev_predictions = trainer.predict(dev_dataset)
    dev_probs_arr = _sigmoid(dev_predictions.predictions)
    dev_probs = {label: dev_probs_arr[:, idx] for idx, label in enumerate(label_cols)}

    logger.info("Predicting on test set...")
    predictions = trainer.predict(test_dataset)
    probs = _sigmoid(predictions.predictions)
    test_probs = {label: probs[:, idx] for idx, label in enumerate(label_cols)}

    metrics = predictions.metrics or {}
    metrics = {key: float(value) for key, value in metrics.items()}
    logger.info(f"Final metrics: {metrics}")

    return BertTrainingResult(
        test_probs=test_probs,
        dev_probs=dev_probs,
        model_path=final_model_dir,
        tokenizer_path=tokenizer_dir,
        trainer_metrics=metrics,
        trainer=trainer,
    )
