"""revati's exploratory data analysis for the Jigsaw toxic comment dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    import networkx as nx
except ImportError:  # pragma: no cover
    nx = None

LABEL_COLUMNS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "revati_exploratory_analysis"
PLOTS_DIR = ANALYSIS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find dataset at {csv_path}. Make sure you have downloaded "
            "the Jigsaw data via scripts/02_download_kaggle.sh."
        )
    df = pd.read_csv(csv_path)
    expected_cols = {"comment_text", *LABEL_COLUMNS}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")
    return df


def basic_overview(df: pd.DataFrame) -> Dict[str, int]:
    info = {
        "num_rows": int(df.shape[0]),
        "num_columns": int(df.shape[1]),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 ** 2), 2),
    }
    return info


def null_analysis(df: pd.DataFrame) -> pd.DataFrame:
    null_counts = df.isnull().sum().rename("null_count")
    null_pct = (df.isnull().mean() * 100).rename("null_pct")
    empty_comments = (df["comment_text"].astype(str).str.strip() == "").sum()
    summary = pd.concat([null_counts, null_pct], axis=1)
    summary.loc["empty_comment_text", :] = [empty_comments, empty_comments / len(df) * 100]
    return summary


def label_statistics(df: pd.DataFrame) -> Dict[str, object]:
    labels_df = df[LABEL_COLUMNS]
    label_counts = labels_df.sum().sort_values(ascending=False)
    total = len(df)
    label_pct = (label_counts / total * 100).round(3)
    label_matrix = labels_df.to_numpy(dtype=int)
    labels_per_sample = label_matrix.sum(axis=1)
    no_label = int((labels_per_sample == 0).sum())
    at_least_one = total - no_label
    combination_counts = Counter(tuple(row) for row in label_matrix)
    top_combos = combination_counts.most_common(10)
    co_occurrence = labels_df.T.dot(labels_df)
    return {
        "label_counts": label_counts.to_dict(),
        "label_percentages": label_pct.to_dict(),
        "no_label_rows": no_label,
        "at_least_one_label_rows": at_least_one,
        "labels_per_row_distribution": Counter(labels_per_sample).most_common(),
        "top_label_combinations": top_combos,
        "co_occurrence": co_occurrence,
    }


def text_statistics(df: pd.DataFrame) -> Dict[str, object]:
    text_series = df["comment_text"].astype(str)
    char_lengths = text_series.str.len()
    word_lengths = text_series.str.split().map(len)
    stats = {
        "char_length_summary": char_lengths.describe().to_dict(),
        "word_length_summary": word_lengths.describe().to_dict(),
        "num_duplicates": int(text_series.duplicated().sum()),
        "longest_comment": text_series.iloc[char_lengths.idxmax()],
        "shortest_comment": text_series.iloc[char_lengths.idxmin()],
    }
    df["char_length"] = char_lengths
    df["word_length"] = word_lengths
    return stats


def plot_nulls(null_df: pd.DataFrame):
    subset = null_df[null_df["null_count"] > 0].copy()
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=subset.index, y="null_count", data=subset, ax=ax)
    ax.set_title("Null Values per Column")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "null_counts.png", dpi=200)
    plt.close(fig)


def plot_label_counts(label_counts: Dict[str, int]):
    labels, counts = zip(*label_counts.items())
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=list(labels), y=list(counts), ax=ax)
    ax.set_title("Label Distribution")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "label_distribution.png", dpi=200)
    plt.close(fig)


def plot_label_hist(labels_per_row: List[Tuple[int, int]]):
    counts = pd.Series(dict(labels_per_row))
    counts = counts.sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=counts.index, y=counts.values, ax=ax)
    ax.set_xlabel("Number of Labels per Comment")
    ax.set_ylabel("Count")
    ax.set_title("Label Cardinality Distribution")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "label_cardinality.png", dpi=200)
    plt.close(fig)


def plot_co_occurrence(co_matrix: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(co_matrix, annot=True, fmt=".0f", cmap="Blues", ax=ax)
    ax.set_title("Label Co-occurrence Heatmap")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "label_cooccurrence_heatmap.png", dpi=200)
    plt.close(fig)


def plot_length_distributions(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df["char_length"], bins=50, ax=axes[0])
    axes[0].set_title("Character Length Distribution")
    sns.boxplot(x=df["char_length"], ax=axes[1])
    axes[1].set_title("Character Length Boxplot")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "char_length_distribution.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df["word_length"], bins=50, ax=axes[0])
    axes[0].set_title("Word Length Distribution")
    sns.boxplot(x=df["word_length"], ax=axes[1])
    axes[1].set_title("Word Length Boxplot")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "word_length_distribution.png", dpi=200)
    plt.close(fig)


def build_label_network(co_matrix: pd.DataFrame):
    if nx is None:
        print("Skipping label co-occurrence network (networkx not installed).")
        return
    G = nx.Graph()
    for label in LABEL_COLUMNS:
        G.add_node(label)
    for i, lbl_i in enumerate(LABEL_COLUMNS):
        for j in range(i + 1, len(LABEL_COLUMNS)):
            lbl_j = LABEL_COLUMNS[j]
            weight = co_matrix.loc[lbl_i, lbl_j]
            if weight > 0:
                G.add_edge(lbl_i, lbl_j, weight=weight)
    pos = nx.spring_layout(G, seed=42)
    weights = [G[u][v]["weight"] for u, v in G.edges]
    if weights:
        norm = np.array(weights) / np.max(weights)
    else:
        norm = [1]
    fig, ax = plt.subplots(figsize=(6, 6))
    nx.draw(G, pos, with_labels=True, width=norm, ax=ax)
    ax.set_title("Label Co-occurrence Network")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "label_network.png", dpi=200)
    plt.close(fig)


def convert_to_native(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_native(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_native(item) for item in obj]
    return obj

def save_summary(summary: Dict[str, object]):
    output_path = ANALYSIS_DIR / "summary_stats.json"
    # Convert numpy types to native Python types
    summary_native = convert_to_native(summary)
    with open(output_path, "w") as f:
        json.dump(summary_native, f, indent=2)
    print(f"Saved summary statistics to {output_path}")


def run_eda(data_path: Path):
    df = load_dataset(data_path)
    overview = basic_overview(df)
    nulls = null_analysis(df)
    label_stats = label_statistics(df)
    text_stats = text_statistics(df)

    plot_nulls(nulls)
    plot_label_counts(label_stats["label_counts"])
    plot_label_hist(label_stats["labels_per_row_distribution"])
    plot_co_occurrence(label_stats["co_occurrence"])
    plot_length_distributions(df)
    build_label_network(label_stats["co_occurrence"])

    summary = {
        "overview": overview,
        "nulls": nulls.fillna(0).to_dict(),
        "label_stats": {
            "label_counts": label_stats["label_counts"],
            "label_percentages": label_stats["label_percentages"],
            "no_label_rows": label_stats["no_label_rows"],
            "at_least_one_label_rows": label_stats["at_least_one_label_rows"],
            "labels_per_row_distribution": label_stats["labels_per_row_distribution"],
            "top_label_combinations": [
                {"combination": combo, "count": count}
                for combo, count in label_stats["top_label_combinations"]
            ],
        },
        "text_stats": text_stats,
    }
    save_summary(summary)
    print("EDA complete. Check the plots directory for visualizations.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run revati's toxic comment EDA")
    parser.add_argument(
        "--data-path",
        default=str(ANALYSIS_DIR / "jigsaw-toxic-comment-classification-challenge" / "train.csv"),
        help="Path to the train.csv file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data_path)
    try:
        run_eda(data_path)
    except Exception as exc:  # pragma: no cover
        print(f"EDA failed: {exc}")
        raise


if __name__ == "__main__":
    main()
