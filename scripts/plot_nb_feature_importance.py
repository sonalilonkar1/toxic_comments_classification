"""Plot top-k words learned by Naive Bayes using feature likelihood ratios.

This script visualizes the top 10 words per label that Naive Bayes uses for classification,
based on the likelihood ratio: P(word | toxic) / P(word | non-toxic).
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.interpretability import get_feature_names

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 8)


def extract_likelihood_ratios(
    model: MultinomialNB, feature_names: List[str]
) -> Tuple[np.ndarray, List[str]]:
    """Extract feature likelihood ratios from a trained MultinomialNB model.
    
    For binary classification:
    - P(word | label=1) / P(word | label=0) = exp(log_prob[1] - log_prob[0])
    
    Args:
        model: Trained MultinomialNB model
        feature_names: List of feature names from vectorizer
        
    Returns:
        Tuple of (likelihood_ratios, feature_names)
    """
    # feature_log_prob_ shape: (n_classes, n_features)
    # For binary: [0] = negative class, [1] = positive class
    log_probs = model.feature_log_prob_
    
    if log_probs.shape[0] != 2:
        raise ValueError(
            f"Expected binary classification, but model has {log_probs.shape[0]} classes"
        )
    
    # Calculate likelihood ratio: P(word | positive) / P(word | negative)
    # = exp(log P(word | positive) - log P(word | negative))
    log_ratios = log_probs[1] - log_probs[0]
    likelihood_ratios = np.exp(log_ratios)
    
    return likelihood_ratios, feature_names


def get_top_k_words(
    likelihood_ratios: np.ndarray,
    feature_names: List[str],
    k: int = 10
) -> List[Tuple[str, float]]:
    """Get top-k words with highest likelihood ratios.
    
    Args:
        likelihood_ratios: Array of likelihood ratios
        feature_names: List of feature names
        k: Number of top words to return
        
    Returns:
        List of (word, ratio) tuples, sorted by ratio (descending)
    """
    # Get top-k indices
    top_indices = np.argsort(likelihood_ratios)[::-1][:k]
    
    # Return word and ratio pairs
    top_words = [(feature_names[i], likelihood_ratios[i]) for i in top_indices]
    
    return top_words


def plot_top_words_per_label(
    fold_dir: Path,
    output_path: Path,
    top_k: int = 10,
    label_cols: List[str] = None
):
    """Plot top-k words per label using Naive Bayes likelihood ratios.
    
    Args:
        fold_dir: Directory containing trained models
        output_path: Path to save the plot
        top_k: Number of top words to show per label
        label_cols: List of label columns (if None, auto-detect from model files)
    """
    model_dir = fold_dir / "models"
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Load vectorizer
    tfidf_path = model_dir / "tfidf.joblib"
    if not tfidf_path.exists():
        raise FileNotFoundError(f"TF-IDF vectorizer not found: {tfidf_path}")
    
    tfidf = joblib.load(tfidf_path)
    feature_names = get_feature_names(tfidf)
    
    # Auto-detect labels if not provided
    if label_cols is None:
        label_cols = []
        for model_path in model_dir.glob("*.joblib"):
            if model_path.name != "tfidf.joblib":
                label_cols.append(model_path.stem)
        label_cols = sorted(label_cols)
    
    # Collect top words for each label
    all_top_words: Dict[str, List[Tuple[str, float]]] = {}
    
    for label in label_cols:
        model_path = model_dir / f"{label}.joblib"
        if not model_path.exists():
            print(f"Warning: Model for {label} not found, skipping...")
            continue
        
        model = joblib.load(model_path)
        
        if not isinstance(model, MultinomialNB):
            print(f"Warning: Model for {label} is not MultinomialNB, skipping...")
            continue
        
        # Extract likelihood ratios
        likelihood_ratios, _ = extract_likelihood_ratios(model, feature_names)
        
        # Get top-k words
        top_words = get_top_k_words(likelihood_ratios, feature_names, k=top_k)
        all_top_words[label] = top_words
    
    if not all_top_words:
        raise ValueError("No valid Naive Bayes models found to analyze")
    
    # Create visualization
    n_labels = len(all_top_words)
    n_cols = 3
    n_rows = (n_labels + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    if n_labels == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Plot each label
    for idx, (label, top_words) in enumerate(all_top_words.items()):
        ax = axes[idx]
        
        words = [w for w, _ in top_words]
        ratios = [r for _, r in top_words]
        
        # Create bar chart
        bars = ax.barh(range(len(words)), ratios, color='steelblue', alpha=0.7)
        
        # Customize
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontsize=10)
        ax.set_xlabel('Likelihood Ratio\nP(word | toxic) / P(word | non-toxic)', fontsize=11)
        ax.set_title(f'{label.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.invert_yaxis()  # Top word at top
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for i, (bar, ratio) in enumerate(zip(bars, ratios)):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{ratio:.2f}', 
                   ha='left', va='center', fontsize=9)
    
    # Hide unused subplots
    for idx in range(len(all_top_words), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(
        f'Top {top_k} Words Learned by Naive Bayes (Feature Likelihood Ratios)',
        fontsize=16,
        fontweight='bold',
        y=0.995
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot top-k words per label using Naive Bayes likelihood ratios"
    )
    parser.add_argument(
        "--fold-dir",
        type=Path,
        required=True,
        help="Path to experiment fold directory containing models/"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for plot (default: fold_dir/nb_top_words.png)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top words to show per label (default: 10)"
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Label columns to analyze (default: auto-detect from model files)"
    )
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = args.fold_dir / "nb_top_words.png"
    
    plot_top_words_per_label(
        fold_dir=args.fold_dir,
        output_path=args.output,
        top_k=args.top_k,
        label_cols=args.labels
    )


if __name__ == "__main__":
    main()




