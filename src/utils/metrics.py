"""Evaluation metrics for multi-label classification."""

import numpy as np
import pandas as pd
from typing import Union, Optional, List, Dict, Tuple
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_recall_curve,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    brier_score_loss,
)


def probs_to_preds(
    prob_dict: Dict[str, np.ndarray],
    threshold: Union[float, Dict[str, float]] = 0.5
) -> np.ndarray:
    """Convert per-label probabilities to binary predictions using threshold(s)."""
    preds = []
    for label in prob_dict.keys():
        probs = prob_dict[label]
        thresh = threshold.get(label, 0.5) if isinstance(threshold, dict) else threshold
        preds.append((probs >= thresh).astype(int))
    return np.column_stack(preds)


def find_precision_thresholds(
    y_true: np.ndarray,
    prob_dict: Dict[str, np.ndarray],
    label_cols: List[str],
    target_precision: float = 0.90,
) -> Dict[str, float]:
    """Find threshold for each label that achieves at least target_precision."""
    thresholds = {}
    for idx, label in enumerate(label_cols):
        y_label = y_true[:, idx]
        probs = prob_dict[label]
        
        precisions, recalls, thresh_values = precision_recall_curve(y_label, probs)
        
        # Filter for precisions >= target
        valid_indices = np.where(precisions[:-1] >= target_precision)[0]
        
        if len(valid_indices) > 0:
            best_idx = valid_indices[np.argmax(recalls[valid_indices])]
            thresholds[label] = float(thresh_values[best_idx])
        else:
            thresholds[label] = 0.5
            
    return thresholds


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece


def compute_top_k_metrics(
    y_true: np.ndarray,
    prob_dict: Dict[str, np.ndarray],
    label_cols: List[str],
    k: int,
) -> Dict[str, float]:
    """Simulate a review queue of capacity K."""
    all_probs = []
    all_true = []
    
    for idx, label in enumerate(label_cols):
        probs = prob_dict[label]
        true_vals = y_true[:, idx]
        all_probs.extend(probs)
        all_true.extend(true_vals)
        
    all_probs = np.array(all_probs)
    all_true = np.array(all_true)
    
    sorted_indices = np.argsort(-all_probs)
    top_k_indices = sorted_indices[:k]
    
    top_k_true = all_true[top_k_indices]
    
    precision_at_k = np.mean(top_k_true)
    total_positives = np.sum(all_true)
    recall_at_k = np.sum(top_k_true) / total_positives if total_positives > 0 else 0.0
    
    return {
        f"precision_at_{k}": float(precision_at_k),
        f"recall_at_{k}": float(recall_at_k),
        f"hits_at_{k}": int(np.sum(top_k_true)),
    }


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_cols: List[str],
    prob_dict: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[Dict, pd.DataFrame]:
    """Compute overall and per-label metrics for multi-label classification.

    Returns:
        Tuple of (overall_metrics_dict, per_label_df)
    """
    # Overall metrics
    overall_metrics = {
        "micro_precision": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "micro_recall": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "hamming_loss": hamming_loss(y_true, y_pred),
        "subset_accuracy": accuracy_score(y_true, y_pred),
    }

    # Per-label metrics
    per_label_rows = []
    brier_scores = []
    ece_scores = []

    for idx, label in enumerate(label_cols):
        prec, rec, f1, support = precision_recall_fscore_support(
            y_true[:, idx], y_pred[:, idx], average="binary", zero_division=0
        )
        
        row = {
            "label": label,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": int(y_true[:, idx].sum()),
        }

        # Calibration metrics
        if prob_dict is not None and label in prob_dict:
            probs = prob_dict[label]
            bs = brier_score_loss(y_true[:, idx], probs)
            ece = expected_calibration_error(y_true[:, idx], probs)
            row["brier_score"] = bs
            row["ece"] = ece
            brier_scores.append(bs)
            ece_scores.append(ece)
        
        per_label_rows.append(row)

    if brier_scores:
        overall_metrics["macro_brier"] = np.mean(brier_scores)
        overall_metrics["macro_ece"] = np.mean(ece_scores)

    per_label_df = pd.DataFrame(per_label_rows).sort_values("f1", ascending=False)

    return overall_metrics, per_label_df


def compute_fairness_slices(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_cols: List[str],
    identity_cols: List[str],
    min_support: int = 50,
) -> pd.DataFrame:
    """Compute fairness metrics by identity subgroups."""
    if not identity_cols:
        return pd.DataFrame()

    overall_f1_lookup = dict(zip(label_cols, [f1_score(y_true[:, idx], y_pred[:, idx], zero_division=0) for idx in range(len(label_cols))]))

    identity_records = []
    for identity in identity_cols:
        mask = test_df.get(identity, pd.Series(dtype=float)).fillna(0) >= 0.5
        support = int(mask.sum())
        if support < min_support:
            continue
        for idx, label in enumerate(label_cols):
            y_true_slice = y_true[mask.values, idx]
            y_pred_slice = y_pred[mask.values, idx]
            if y_true_slice.size == 0:
                continue
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true_slice, y_pred_slice, average="binary", zero_division=0
            )
            identity_records.append({
                "identity": identity,
                "label": label,
                "support": support,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "overall_f1": overall_f1_lookup.get(label, np.nan),
            })

    return pd.DataFrame(identity_records)
