"""Calibration analysis tools."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.calibration import calibration_curve

def plot_reliability_diagram(
    y_true: np.ndarray, 
    y_prob: np.ndarray, 
    output_path: Path, 
    label: str = "Calibration", 
    n_bins: int = 10
) -> None:
    """Plot reliability diagram (calibration curve)."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
    plt.plot(prob_pred, prob_true, "s-", label=label)
    
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(f"Reliability Diagram: {label}")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_multilabel_reliability(
    y_true: np.ndarray, 
    prob_dict: Dict[str, np.ndarray], 
    label_cols: List[str], 
    output_dir: Path
) -> None:
    """Generate reliability diagrams for all labels."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, label in enumerate(label_cols):
        if label not in prob_dict:
            continue
            
        probs = prob_dict[label]
        true_vals = y_true[:, i]
        
        plot_reliability_diagram(
            true_vals, 
            probs, 
            output_dir / f"reliability_{label}.png", 
            label=label
        )

