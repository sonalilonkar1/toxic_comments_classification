# Toxic Comment Classification – Project Overview

This document gives new collaborators a guided tour of the repository, the project goals, and how the major components fit together. It complements the main `README.md` (quick start) and `proposal.md` (research plan) by spelling out the file layout and responsibilities in detail.

---
## 1. Project Goals
- Benchmark multiple model families (classical TF-IDF + linear/trees, LSTM, DistilBERT/BERT) on the Jigsaw multi-label toxic comment dataset.
- Standardize preprocessing (normalization, bucket tagging), time-aware splits, and experiment logging so results are reproducible.
- Address class imbalance via weighting/oversampling, calibrate probabilities, evaluate fairness gaps, and capture interpretable explanations (SHAP, attention, casebook).
- Provide both notebook-based exploration and CLI/pipeline tooling suitable for automated runs.

---
## 2. Repository Structure & Key Files

| Path | Purpose |
| --- | --- |
| `README.md` | Quick-start instructions, current capabilities, pipeline usage examples, experiment directories. |
| `proposal.md` | Full CMPE-255 project proposal describing motivation, model lineup, evaluation plan, timelines. |
| `plan.md` | Implementation roadmap that tracks what’s complete and what remains per proposal section. |
| `requirements.txt` | Python dependencies (PyData, scikit-learn, shap, transformers, etc.). PyTorch is installed separately via script. |
| `kaggle.json` (optional) | Local Kaggle API credentials; `.gitignore`-d. |
| `.gitignore` | Excludes virtualenvs, data, experiments, and other generated artifacts. |
| `scripts/` | Bootstrap shell scripts: `00_bootstrap_project.sh` (full setup), `01_make_env_macos.sh`, `02_download_kaggle.sh`, `02_install_torch_macos.sh`, and `run_python.sh` (utility wrapper that executes Python inside the managed env). |
| `data/` | `raw/` holds Kaggle CSVs (train/test). `splits/` stores chronological fold JSON indices produced by `src.cli.make_splits`. |
| `experiments/` | Canonical home for reproducible outputs. Subfolders include `bucket_augmentation/` (notebook logs), `multilabel_analysis/` (EDA artifacts), and `tfidf_logreg/` (CLI pipeline runs). Additional experiments should mirror this structure. |
| `artifacts/` | Reserved for model checkpoints or reports beyond the main experiments directory. |
| `notebooks/` | Primary exploratory assets: `data_explore.ipynb` (normalization, TF-IDF baseline, bucket augmentation) and `multilabel_analysis.ipynb` (fold-driven training, metrics, fairness, SHAP). Both notebooks import shared logic from `src/`. |
| `reports/` | Empty `figures/` and `tables/` directories ready for final plots/tables. |
| `configs/` | Placeholder YAMLs (`bert_distil.yaml`, `linear_svm.yaml`, `random_forest.yaml`) for future experiment configs. |
| `tests/` | Pytest suite validating shared modules (`test_preprocess.py`, `test_train_pipeline.py`). |
| `src/` | Python package containing reusable modules (see section 3). |

### Repository History (optional legacy)
- `Ramya data experiment and processing/`: Legacy exploratory scripts/configs from earlier EDA work. Useful for reference but not part of the current pipeline.

---
## 3. Source Package (`src/`)

```
src/
├── cli/
│   ├── download_data.py        # Kaggle download helper invoked by scripts/02_download_kaggle.sh
│   ├── make_splits.py          # CLI to create chronological fold JSON files
│   └── train_pipeline.py       # CLI entry for reusable TF-IDF + logistic pipeline
├── data/
│   ├── dataset.py              # `load_fold_frames`, fold summaries
│   ├── preprocess.py           # `toy_normalize`, `rich_normalize` text utilities
│   └── ...
├── features/
│   └── tfidf.py                # TF-IDF vectorizer factory plus bucket oversampling helper
├── models/
│   ├── tfidf_logistic.py       # Multi-label TF-IDF + LogisticRegression trainer
│   ├── tfidf_svm.py            # Calibrated TF-IDF + LinearSVC trainer
│   ├── tfidf_random_forest.py  # TF-IDF + RandomForestClassifier trainer
│   ├── bert_model.py           # Placeholder for transformer models
│   └── traditional.py          # Placeholder for NB/LR/SVM/RF/XGB implementations
├── pipeline/
│   ├── train.py                # High-level training pipeline orchestrating data loading, normalization, TF-IDF+logistic training, metrics, fairness, artifact logging
│   └── __init__.py
├── utils/
│   ├── metrics.py              # Multi-label metrics, probabilities→predictions, fairness slice computation
│   └── ...
└── ...
```

### Highlights
- **`src/pipeline/train.py`**: Encapsulates the TF-IDF + logistic baseline. It loads folds via `load_fold_frames`, applies optional normalization/bucket oversampling, trains per-label logistic heads, computes metrics/fairness, and writes artifacts (metrics JSON/CSV, predictions, serialized models) under timestamped `experiments/tfidf_logreg/` directories. Configured via the `TrainConfig` dataclass.
- **`src/cli/train_pipeline.py`**: Argparse front-end for the pipeline. Supports selecting folds, normalization strategy, TF-IDF/model hyperparameters, bucket oversampling (`--bucket-col`, `--bucket-mult`), and fairness thresholds. Prints summary micro/macro F1 and hamming loss per fold.
- **`src/features/tfidf.py`**: Houses TF-IDF vectorizer creation plus `oversample_buckets`. Shared between notebooks and the pipeline.
- **`src/models/tfidf_logistic.py`**: Implements the multi-label TF-IDF + logistic trainer used across notebooks and the pipeline.
- **`src/models/tfidf_svm.py`**: Provides the calibrated TF-IDF + LinearSVC trainer (probability estimates for downstream metrics).
- **`src/models/tfidf_random_forest.py`**: Offers a TF-IDF + RandomForest baseline compatible with the same interfaces.
- **`src/utils/metrics.py`**: Provides `compute_multilabel_metrics`, `compute_fairness_slices`, and `probs_to_preds`. Enables consistent evaluation across notebooks, pipeline runs, and future models.
- **`src/data/preprocess.py`**: Contains normalization routines referenced by notebooks and future data loaders. Tests in `tests/test_preprocess.py` ensure deterministic behavior.
- **`src/data/dataset.py`**: Handles reading `train.csv`, injecting mock timestamps, and hydrating fold splits from JSON index files. It returns combined data structures (base dataframe, fold map, identity column list, fold size table).

---
## 4. Notebooks & Experiments

### `notebooks/data_explore.ipynb`
- Imports normalization and TF-IDF helpers from `src/`.
- Runs dataset profiling, label prevalence, drift analysis, and TF-IDF + logistic baseline.
- Demonstrates bucket tagging + oversampling sweeps; writes results to `experiments/bucket_augmentation/outputs/bucket_aug_runs.csv`.
- Use this notebook to prototype new preprocessing ideas before porting to `src/`.

### `notebooks/multilabel_analysis.ipynb`
- Loads fold splits via `src.data.dataset.load_fold_frames`.
- Builds train/dev/test matrices, trains TF-IDF + logistic models, evaluates metrics/fairness, and runs SHAP explainability.
- Persists artifacts (metrics, fairness slices, SHAP tables) under `experiments/multilabel_analysis/`.
- Acts as a human-readable report and sanity check for the CLI pipeline outputs.

### `experiments/`
- **`tfidf_logreg/`**: Each CLI run creates a timestamped folder `fold*-seed*-norm*-<timestamp>/` containing metrics JSON, per-label metrics, fairness slices, predictions, serialized TF-IDF + per-label models, and a config snapshot.
- **`bucket_augmentation/outputs/`**: CSV logs of bucket oversampling sweeps initiated from the exploratory notebook.
- **`multilabel_analysis/`**: Notebook-generated CSV/JSON artifacts for drift stats, metrics, SHAP feature importance, etc.

---
## 5. Tests & Quality Gates
- `tests/test_preprocess.py`: Ensures `toy_normalize` and `rich_normalize` behave as expected.
- `tests/test_train_pipeline.py`: Spins up a tiny synthetic dataset to verify the pipeline runs end-to-end (including bucket augmentation validation) and writes artifact files.
- Run the suite via `./scripts/run_python.sh -m pytest -v` or target specific files for faster feedback.

---
## 6. Typical Workflows
1. **Setup**: `bash scripts/00_bootstrap_project.sh` (creates environment, installs deps, downloads data, generates folds).
2. **Exploration**: Open `notebooks/data_explore.ipynb` or `notebooks/multilabel_analysis.ipynb`, run cells sequentially, and inspect generated artifacts.
3. **Pipeline Run**: Use `./scripts/run_python.sh -m src.cli.train_pipeline --fold fold1_seed42 --output-dir experiments/tfidf_logreg` (add `--bucket-col`/`--bucket-mult` when bucket tags exist).
4. **Testing**: `./scripts/run_python.sh -m pytest tests/test_preprocess.py tests/test_train_pipeline.py -v` before committing changes.
5. **Review Artifacts**: Explore new subdirectories under `experiments/` for metrics, fairness slices, SHAP tables, and serialized models.

---
## 7. Roadmap Snapshot
- Classical model implementations (NB/SVM/RF/XGB) and calibrators still need to be wired into `src/models/` and the CLI.
- Neural (LSTM) and Transformer (DistilBERT/BERT) training code is planned but not yet implemented.
- Decision-policy tooling (fixed-precision, top-K) and fairness/interpretability reports will expand once additional models are in place.
- See `plan.md` for the detailed execution backlog.

---
## 8. Getting Help / Next Steps
- Read `plan.md` to understand immediate priorities.
- Ping the original authors (per proposal) for context on pending features or to coordinate GPU resources.
- Use this overview plus the README to onboard quickly and start contributing.
