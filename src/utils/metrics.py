"""Evaluation metrics for multi-label classification."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)


def probs_to_preds(prob_dict: dict[str, np.ndarray], threshold: float = 0.5) -> np.ndarray:
    """Convert per-label probabilities to binary predictions using threshold."""
    return np.column_stack([(prob_dict[label] >= threshold).astype(int) for label in prob_dict.keys()])


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_cols: list[str],
) -> tuple[dict, pd.DataFrame]:
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
    for idx, label in enumerate(label_cols):
        prec, rec, f1, support = precision_recall_fscore_support(
            y_true[:, idx], y_pred[:, idx], average="binary", zero_division=0
        )
        per_label_rows.append({
            "label": label,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": int(y_true[:, idx].sum()),
        })

    import pandas as pd
    per_label_df = pd.DataFrame(per_label_rows).sort_values("f1", ascending=False)

    return overall_metrics, per_label_df


def compute_fairness_slices(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_cols: list[str],
    identity_cols: list[str],
    min_support: int = 50,
) -> pd.DataFrame:
    """Compute fairness metrics by identity subgroups.

    Returns:
        DataFrame with fairness gaps per identity/label
    """
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