# Toxic Comment Classification (Starter Scaffold)

This repository now contains a minimal, data-ready scaffold for the Jigsaw
toxic-comment dataset. The focus is strictly on:

- provisioning a reproducible Python environment,
- downloading and storing the Kaggle data locally, and
- generating chronological fold splits for future experiments.

All training pipelines, notebooks, and model code were intentionally removed to
keep the repo lightweight. (A full copy of the previous project state lives
outside this repository.)

## Prerequisites

- Python 3.8+ with either Conda/Miniforge or the ability to create a virtualenv
- `git`, `curl`/`wget`, and the Kaggle CLI (`pip install kaggle`)
- Kaggle API token saved as `~/.kaggle/kaggle.json` or copied into the project
	root

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

## Minimal Source Layout

The `src/` tree currently exposes only two CLI entry points:

- `src/cli/download_data.py` – logic shared by the download script
- `src/cli/make_splits.py` – chronological fold generation

Other packages (`src/data`, `src/features`, `src/models`, `src/pipeline`,
`src/utils`) remain as placeholders so the directory structure is ready when you
add new functionality. They intentionally do **not** contain any implementation
yet.

## Data Locations

- `data/raw/` – Kaggle CSVs (`train.csv`, `test.csv`, etc.) once downloaded
- `data/splits/` – JSON index files produced by the split CLI
- `artifacts/` – currently empty; reserved for future experiment outputs

Raw datasets can be large; keep them out of Git unless absolutely required. Use
`.gitignore` (already configured) to avoid accidental commits.

## Next Steps

- Extend `src/data/` with normalization utilities or feature builders.
- Implement training/inference modules under `src/models/` and
	`src/pipeline/` when ready.
- Add notebooks or reports back once you start experimentation (they were
	removed from this slim scaffold).

Until then, this repo is deliberately lean: it gets you an environment, the
datasets, and deterministic splits—nothing more.

---

