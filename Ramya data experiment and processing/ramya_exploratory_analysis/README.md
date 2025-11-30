# Ramya's Exploratory Data Analysis

This directory contains the exploratory data analysis (EDA) assets for the Jigsaw toxic comment classification dataset.

## Key Deliverables

- `eda_ramya.ipynb`: interactive notebook capturing all analyses
- `eda_ramya.py`: script variant for programmatic execution
- `plots/`: saved visualizations generated during the EDA workflow
- `summary_stats.json`: comprehensive statistics and metrics

## Prerequisites

- Python 3.8+ with required packages: pandas, numpy, matplotlib, seaborn, networkx
- Jigsaw toxic comment dataset (`train.csv`) located in `jigsaw-toxic-comment-classification-challenge/train.csv`

## How to Run

### Option 1: Run Python Script (Recommended)

From the project root:
```bash
cd /Users/ramyag/Desktop/toxic_comments_classification
python3 "Ramya data experiment and processing/ramya_exploratory_analysis/eda_ramya.py"
```

Or from inside this directory:
```bash
cd "Ramya data experiment and processing/ramya_exploratory_analysis"
python3 eda_ramya.py
```

### Option 2: Run with Custom Data Path

If your `train.csv` is in a different location:
```bash
python3 eda_ramya.py --data-path "path/to/your/train.csv"
```

### Option 3: Use Jupyter Notebook

Open `eda_ramya.ipynb` in Jupyter Notebook or JupyterLab and run all cells.

## Output

After running, you'll find:
- **Plots** in `plots/` directory:
  - `char_length_distribution.png` - Character length analysis
  - `word_length_distribution.png` - Word count analysis
  - `label_distribution.png` - Distribution of each toxicity label
  - `label_cardinality.png` - Number of labels per comment
  - `label_cooccurrence_heatmap.png` - Label co-occurrence matrix
  - `label_network.png` - Label co-occurrence network graph
- **Statistics** in `summary_stats.json`:
  - Dataset overview (rows, columns, memory usage)
  - Null value analysis
  - Label statistics and distributions
  - Text statistics (lengths, duplicates)
