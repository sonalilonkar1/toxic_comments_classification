"""BERT/DistilBERT model definitions and training utilities."""

import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EvalPrediction,
)
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

class ToxicDataset(Dataset):
    """PyTorch Dataset for Toxic Comments."""
    
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def compute_metrics(p: EvalPrediction) -> Dict[str, float]:
    """Compute metrics for HuggingFace Trainer."""
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    # Apply sigmoid to logits
    probs = 1.0 / (1.0 + np.exp(-preds))
    y_true = p.label_ids
    
    # Use 0.5 threshold for F1 logic during training (can be tuned later)
    y_pred = (probs >= 0.5).astype(int)
    
    metrics = {
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y_true, probs, average="macro"),
        "pr_auc": average_precision_score(y_true, probs, average="macro")
    }
    return metrics


def train_bert_model(
    model_name: str,
    X_train_text: List[str],
    y_train: np.ndarray,
    X_val_text: List[str],
    y_val: np.ndarray,
    output_dir: str,
    num_labels: int = 6,
    max_length: int = 128,
    batch_size: int = 16,
    epochs: int = 3,
    learning_rate: float = 2e-5,
    seed: int = 42,
    fp16: bool = False,
) -> Tuple[Trainer, AutoTokenizer]:
    """Fine-tune a BERT/DistilBERT model for multi-label classification."""
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize data
    train_encodings = tokenizer(
        X_train_text, truncation=True, padding=True, max_length=max_length
    )
    val_encodings = tokenizer(
        X_val_text, truncation=True, padding=True, max_length=max_length
    )
    
    train_dataset = ToxicDataset(train_encodings, y_train)
    val_dataset = ToxicDataset(val_encodings, y_val)
    
    # Load model with correct classification head
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification"
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, config=config
    )
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="pr_auc",  # Optimize for PR-AUC as per proposal
        learning_rate=learning_rate,
        seed=seed,
        fp16=fp16 and torch.cuda.is_available(), # Use mixed precision if requested and available
        push_to_hub=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    
    return trainer, tokenizer

