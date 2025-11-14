# Toxic Comment Classification

Lightweight project scaffold for experiments and coursework (CMPE-255).

This repository contains code, notebooks, and helper scripts to set up a Python
environment, download the Kaggle dataset, and install macOS-specific PyTorch
builds. The `scripts/` folder contains convenient shell scripts to automate
common setup steps.

**Prerequisites**
- Python 3.8+ or compatible (recommended: use the provided environment scripts)
- `git`, `curl`/`wget`, and optionally `conda` or `venv`
- Kaggle account and API token (if you want to download the dataset via the
	Kaggle CLI)

**Quick Setup (recommended: use scripts)**

1. Make the scripts executable (one-time):

```bash
chmod +x scripts/*.sh
```

2a. Create a macOS Conda environment (recommended on macOS):

```bash
./scripts/01_make_env_macos.sh
# then activate it, e.g. `conda activate toxic-comments` (name shown by script)
```

2b. Or create a plain Python venv (cross-platform):

```bash
./scripts/01_make_venv.sh
# then activate it, e.g. `source .venv/bin/activate`
```

3. Install Python dependencies:

```bash
pip install -r requirements.txt
```

4. (Optional) Install PyTorch for macOS-specific builds:

```bash
./scripts/02_install_torch_macos.sh
```

5. (Optional) Download dataset from Kaggle (requires `~/.kaggle/kaggle.json`):

```bash
./scripts/02_download_kaggle.sh
```

**Scripts in `scripts/`**
- `01_make_env_macos.sh`: Create a Conda environment tailored for macOS.
- `01_make_venv.sh`: Create a local Python virtual environment and install
	`pip` requirements into it.
- `02_download_kaggle.sh`: Use the Kaggle CLI to download the dataset(s).
	Ensure your Kaggle API credentials are available at `~/.kaggle/kaggle.json`.
- `02_install_torch_macos.sh`: Install a macOS-compatible PyTorch wheel (useful
	for Apple Silicon / macOS-specific installs).

**Usage**
- Notebooks are under `notebooks/` — open with Jupyter or VS Code Interactive
	window.
- Source code lives in `src/` — import modules or run scripts from there.

Example: open the main notebook

```bash
jupyter notebook notebooks/
```

**Kaggle credentials**
Place your `kaggle.json` (from https://www.kaggle.com/) in `~/.kaggle/` or set
the environment variables `KAGGLE_USERNAME` and `KAGGLE_KEY` before running the
download script.

**Troubleshooting**
- If a script fails with permissions, re-run `chmod +x scripts/*.sh`.
- For macOS-specific PyTorch issues, confirm Python version and follow PyTorch
	official install instructions: https://pytorch.org/get-started/locally/

**Next steps**
- Run the environment/script suited to your OS and follow the notebooks to
	preprocess data and run experiments.

---

