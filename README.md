# Toxic Comment Classification

This repository now contains a reproducible scaffold **plus** the first wave of
exploratory assets for the Jigsaw toxic-comment dataset. In addition to the
original foundations, we now have:

- a `notebooks/data_explore.ipynb` workbook that benchmarks TF-IDF + Logistic
	baselines, normalization strategies, temporal drift, and bucket-aware
	augmentation with logging to `experiments/bucket_augmentation/outputs/`;
- a `notebooks/multilabel_analysis.ipynb` outline ready to run multi-label
	baselines, drift diagnostics, fairness slices, and SHAP interpretability;
- standardized experiment directories under `experiments/` for artifacts and
	metric traces.

The bootstrap scripts remain unchanged, so you can still spin up the project
from scratch in one command and then open the notebooks for richer analysis.


## Prerequisites

- Python 3.8+ with either Conda/Miniforge or the ability to create a virtualenv
- `git`, `curl`/`wget`, and the Kaggle CLI (`pip install kaggle`)
- Kaggle API token saved as `~/.kaggle/kaggle.json` or copied into the project root

## Quick Start

1. **Bootstrap everything (recommended):**

	 ```bash
	 bash scripts/00_bootstrap_project.sh
	 ```

	 Use `--skip-env`, `--skip-torch`, or `--skip-data` to turn off individual
	 steps if you rerun the bootstrapper later.

2. **Activate the environment:**

	 - If Conda: `conda activate toxbench`
	 - If the script created `.venv/`: `source .venv/bin/activate`

	 > Tip: `scripts/run_python.sh <args>` is a safe wrapper that automatically
	 > targets whichever environment the bootstrapper created.

3. **Download the Kaggle dataset manually (optional):**

	 ```bash
	 bash scripts/02_download_kaggle.sh
	 ```

	 This step is already part of the bootstrapper, but the standalone command is
	 handy if you need to refresh the data only.

4. **Generate chronological folds:**

	 ```bash
	 ./scripts/run_python.sh -m src.cli.make_splits --folds 3
	 ```

	 This reads `data/raw/train.csv` and writes JSON index files to
	 `data/splits/`.

## What's in `scripts/`

- `00_bootstrap_project.sh` – orchestrates env creation, optional PyTorch
	install, and data download.
- `01_make_env_macos.sh` – idempotent macOS-friendly environment creator
	(prefers Conda, falls back to `python -m venv`).
- `02_install_torch_macos.sh` – installs the right PyTorch build for Apple
	Silicon/Intel macOS; skip if you only need the data utilities.
- `02_download_kaggle.sh` – wraps the Kaggle CLI download and unzip workflow,
	honoring a repo-local `kaggle.json` when present.
- `run_python.sh` – helper to execute any Python command inside the managed
	environment (`conda run` or `.venv`).

## Source + Notebook Layout

- `src/cli/download_data.py` – logic shared by the download script
- `src/cli/make_splits.py` – chronological fold generation
- `notebooks/data_explore.ipynb` – end-to-end exploratory workbook (label
	prevalence, normalization, TF-IDF logistic baseline, error taxonomy, bucket
	augmentation; writes logs to `experiments/bucket_augmentation/outputs/`)
- `notebooks/multilabel_analysis.ipynb` – structured notebook for multi-label
	baselines, fairness slices, SHAP, and artifact persistence under
	`experiments/multilabel_analysis/`
- `src/pipeline/train.py` – reusable TF-IDF pipeline that mirrors the notebook flow and logs artifacts under `experiments/tfidf_logreg/`

Other packages (`src/data`, `src/features`, `src/models`, `src/pipeline`,
`src/utils`) remain as placeholders so the directory structure is ready when you
add new functionality from the notebooks or CLI utilities.

## Data & Experiment Locations

- `data/raw/` – Kaggle CSVs (`train.csv`, `test.csv`, etc.) once downloaded
- `data/splits/` – JSON index files produced by the split CLI
- `experiments/bucket_augmentation/` – CSV logs for oversampling sweeps driven
	from `data_explore.ipynb`
- `experiments/multilabel_analysis/` – placeholder for multi-label artifacts
- `experiments/tfidf_logreg/` – CLI-driven TF-IDF + logistic pipeline outputs
- `artifacts/` – general-purpose directory for future model checkpoints or
	reports

Raw datasets can be large; keep them out of Git unless absolutely required. Use
`.gitignore` (already configured) to avoid accidental commits.

## Current Status & Next Steps

- ✅ Environment bootstrap, Kaggle download, and chronological splits are in
	place.
- ✅ Exploratory notebook covers normalization, TF-IDF baseline, error taxonomy,
	and bucket-aware augmentation with logging.
- 🟡 Multi-label notebook is authored and ready to run once you execute the
	cells; results will land under `experiments/multilabel_analysis/`.
- ✅ Reusable TF-IDF pipeline lives in `src/pipeline/train.py` with a CLI 	(`python -m src.cli.train_pipeline`) for running experiments outside the notebooks, including optional bucket-aware
	oversampling controls.
- 🔜 Integrate bucket-aware augmentation knobs into the eventual training
	pipeline and schedule regular drift checks using the chronological folds.

## Run the reusable pipeline

Execute the TF-IDF + logistic baseline across all folds (or a specific fold)
and persist metrics/models under `experiments/tfidf_logreg/`:

```bash
./scripts/run_python.sh -m src.cli.train_pipeline --output-dir experiments/tfidf_logreg
```

To target a single fold with richer normalization:

```bash
./scripts/run_python.sh -m src.cli.train_pipeline \
  --fold fold1_seed42 \
	--model svm \
  --normalization rich \
  --max-features 75000 \
  --output-dir experiments/tfidf_logreg
```

Run a Random Forest baseline with custom hyperparameters:

```bash
./scripts/run_python.sh -m src.cli.train_pipeline \
	--fold fold1_seed42 \
	--model random_forest \
	--rf-n-estimators 600 \
	--rf-max-depth 40 \
	--rf-class-weight balanced \
	--output-dir experiments/tfidf_logreg
```

Apply bucket-aware oversampling by pointing to a bucket column (list-like or JSON
encoded) and specifying repeatable multipliers:

```bash
./scripts/run_python.sh -m src.cli.train_pipeline \
	--bucket-col bucket_tags_full \
	--bucket-mult rare=3 \
	--bucket-mult misogyny=2 \
	--output-dir experiments/tfidf_logreg
```

This repo now gets you an environment, the datasets, deterministic splits, and
actionable notebooks to evaluate baselines and prep the production pipeline.

---

