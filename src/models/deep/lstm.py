"""LSTM model definitions and training utilities for multi-label classification."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

from src.features.lstm_preprocessing import LSTMPreprocessor
from src.models.deep.loss import FocalLoss


class SequenceDataset(Dataset):
    """PyTorch Dataset for padded sequences."""
    
    def __init__(self, sequences: np.ndarray, labels: Optional[np.ndarray] = None):
        """
        Args:
            sequences: Padded sequences of shape (n_samples, seq_length)
            labels: Optional labels of shape (n_samples, n_labels)
        """
        self.sequences = torch.LongTensor(sequences)
        self.labels = torch.FloatTensor(labels) if labels is not None else None
    
    def __getitem__(self, idx):
        item = {"input_ids": self.sequences[idx]}
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item
    
    def __len__(self):
        return len(self.sequences)


class MultiLabelLSTM(nn.Module):
    """LSTM model for multi-label text classification."""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_labels: int = 6,
        dropout: float = 0.3,
        bidirectional: bool = True,
        embedding_matrix: Optional[np.ndarray] = None,
        freeze_embeddings: bool = False,
    ):
        """
        Initialize LSTM model.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            num_labels: Number of output labels
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
            embedding_matrix: Pre-trained embedding matrix (vocab_size, embedding_dim)
            freeze_embeddings: Whether to freeze embedding layer during training
        """
        super(MultiLabelLSTM, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_labels = num_labels
        self.bidirectional = bidirectional
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Load pre-trained embeddings if provided
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        
        # Output layer
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_dim, num_labels)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Input sequences of shape (batch_size, seq_length)
        
        Returns:
            Logits of shape (batch_size, num_labels)
        """
        # Embedding
        embedded = self.embedding(input_ids)  # (batch_size, seq_length, embedding_dim)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use the last hidden state (or concatenate last states if bidirectional)
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        
        # Dropout and classification
        output = self.dropout(hidden)
        logits = self.fc(output)
        
        return logits


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray) -> Dict[str, float]:
    """Compute multi-label classification metrics."""
    metrics = {
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    
    # ROC-AUC and PR-AUC (can be slow for large datasets)
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_probs, average="macro")
        metrics["pr_auc"] = average_precision_score(y_true, y_probs, average="macro")
    except Exception:
        metrics["roc_auc"] = 0.0
        metrics["pr_auc"] = 0.0
    
    return metrics


def train_lstm_model(
    X_train_text: List[str],
    y_train: np.ndarray,
    X_val_text: List[str],
    y_val: np.ndarray,
    preprocessor: LSTMPreprocessor,
    output_dir: str,
    num_labels: int = 6,
    embedding_dim: int = 128,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
    bidirectional: bool = True,
    batch_size: int = 32,
    epochs: int = 10,
    learning_rate: float = 0.001,
    embedding_matrix: Optional[np.ndarray] = None,
    freeze_embeddings: bool = False,
    seed: int = 42,
    device: Optional[str] = None,
    resume_from: Optional[str] = None,
    checkpoint_interval: int = 1,
    loss_type: str = "bce",
    loss_params: Optional[Dict[str, float]] = None,
) -> Tuple[MultiLabelLSTM, LSTMPreprocessor]:
    """
    Train an LSTM model for multi-label classification.
    
    Args:
        X_train_text: Training text (already normalized)
        y_train: Training labels of shape (n_samples, num_labels)
        X_val_text: Validation text (already normalized)
        y_val: Validation labels of shape (n_samples, num_labels)
        preprocessor: Fitted LSTMPreprocessor
        output_dir: Directory to save model checkpoints
        num_labels: Number of output labels
        embedding_dim: Dimension of word embeddings
        hidden_dim: Hidden dimension of LSTM
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        bidirectional: Whether to use bidirectional LSTM
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Learning rate
        embedding_matrix: Pre-trained embedding matrix (optional)
        freeze_embeddings: Whether to freeze embedding layer
        seed: Random seed
        device: Device to use ('cuda', 'cpu', or None for auto-detect)
        resume_from: Path to checkpoint file to resume training from (optional)
        checkpoint_interval: Save checkpoint every N epochs (default: 1, saves every epoch)
        loss_type: 'bce' or 'focal'
        loss_params: Dictionary of parameters for loss function (e.g., {'alpha': 0.25, 'gamma': 2.0})
    
    Returns:
        Tuple of (trained model, preprocessor)
    """
    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Transform texts to sequences
    X_train_seq = preprocessor.transform(X_train_text)
    X_val_seq = preprocessor.transform(X_val_text)
    
    # Create datasets
    train_dataset = SequenceDataset(X_train_seq, y_train)
    val_dataset = SequenceDataset(X_val_seq, y_val)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)
    
    # Initialize model
    vocab_size = preprocessor.get_vocab_size()
    model = MultiLabelLSTM(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_labels=num_labels,
        dropout=dropout,
        bidirectional=bidirectional,
        embedding_matrix=embedding_matrix,
        freeze_embeddings=freeze_embeddings,
    )
    model = model.to(device)
    
    # Loss and optimizer
    if loss_type == "focal":
        params = loss_params or {}
        criterion = FocalLoss(**params)
        print(f"Using Focal Loss with params: {params}")
    else:
        criterion = nn.BCEWithLogitsLoss()
        
    criterion = criterion.to(device) # Ensure criterion is on device (though usually stateless)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_val_f1 = 0.0
    
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming training from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Resume from next epoch
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_f1 = checkpoint.get('val_f1', 0.0)
        
        print(f"Resumed from epoch {start_epoch}, best val F1: {best_val_f1:.4f}")
    
    # Training loop
    os.makedirs(output_dir, exist_ok=True)
    
    for epoch in range(start_epoch, epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_val_logits = []
        all_val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                
                logits = model(input_ids)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                all_val_logits.append(logits.cpu().numpy())
                all_val_labels.append(labels.cpu().numpy())
        
        # Compute metrics
        val_logits = np.vstack(all_val_logits)
        val_labels = np.vstack(all_val_labels)
        val_probs = 1.0 / (1.0 + np.exp(-val_logits))
        val_preds = (val_probs >= 0.5).astype(int)
        
        metrics = compute_metrics(val_labels, val_preds, val_probs)
        val_f1 = metrics["f1_macro"]
        
        print(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_loss/len(train_loader):.4f} - "
            f"Val Loss: {val_loss/len(val_loader):.4f} - "
            f"Val F1 Macro: {val_f1:.4f}"
        )
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'metrics': metrics,
                'vocab_size': vocab_size,
                'embedding_dim': embedding_dim,
                'hidden_dim': hidden_dim,
                'num_layers': num_layers,
                'num_labels': num_labels,
                'dropout': dropout,
                'bidirectional': bidirectional,
            }
            torch.save(best_checkpoint, os.path.join(output_dir, 'best_model.pt'))
            print(f"  -> New best model saved (F1: {val_f1:.4f})")
        
        # Save periodic checkpoint
        if (epoch + 1) % checkpoint_interval == 0 or epoch == epochs - 1:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pt')
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'metrics': metrics,
                'train_loss': train_loss / len(train_loader),
                'val_loss': val_loss / len(val_loader),
                'vocab_size': vocab_size,
                'embedding_dim': embedding_dim,
                'hidden_dim': hidden_dim,
                'num_layers': num_layers,
                'num_labels': num_labels,
                'dropout': dropout,
                'bidirectional': bidirectional,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"  -> Checkpoint saved: {checkpoint_path}")
    
    # Load best model for return
    best_checkpoint_path = os.path.join(output_dir, 'best_model.pt')
    if os.path.exists(best_checkpoint_path):
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\nLoaded best model from epoch {checkpoint['epoch']+1} with F1: {checkpoint['val_f1']:.4f}")
    else:
        print("\nWarning: No best model checkpoint found, using final epoch model")
    
    return model, preprocessor


def predict_lstm(
    model: MultiLabelLSTM,
    preprocessor: LSTMPreprocessor,
    X_text: List[str],
    batch_size: int = 32,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions with trained LSTM model.
    
    Args:
        model: Trained MultiLabelLSTM model
        preprocessor: Fitted LSTMPreprocessor
        X_text: Text to predict on (already normalized)
        batch_size: Batch size for prediction
        device: Device to use
    
    Returns:
        Tuple of (probabilities, predictions) both of shape (n_samples, num_labels)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    # Transform texts to sequences
    X_seq = preprocessor.transform(X_text)
    
    # Create dataset and loader
    dataset = SequenceDataset(X_seq)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Predict
    model.eval()
    all_logits = []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            logits = model(input_ids)
            all_logits.append(logits.cpu().numpy())
    
    logits = np.vstack(all_logits)
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    
    return probs, preds


__all__ = [
    "MultiLabelLSTM",
    "SequenceDataset",
    "train_lstm_model",
    "predict_lstm",
    "compute_metrics",
]

# Re-export load functions for convenience
try:
    from src.models.deep.load_lstm import load_lstm_artifacts, load_lstm_for_prediction
    __all__.extend(["load_lstm_artifacts", "load_lstm_for_prediction"])
except ImportError:
    pass

