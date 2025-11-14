#!/usr/bin/env bash
set -euo pipefail

# Choose one of these strategies:

# --- Strategy 1: Conda/Miniforge (recommended on macOS, Apple Silicon) ---
if command -v conda >/dev/null 2>&1; then
  ENV_NAME="toxbench"
  conda create -y -n "$ENV_NAME" python=3.11
  conda activate "$ENV_NAME"
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  echo "✅ Conda env '$ENV_NAME' ready. Next: bash scripts/02_install_torch_macos.sh"
  exit 0
fi

# --- Strategy 2: pyenv (no conda) ---
if command -v pyenv >/dev/null 2>&1; then
  pyenv install -s 3.11.9
  pyenv local 3.11.9
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  echo "✅ venv with Python $(python -V) ready. Next: bash scripts/02_install_torch_macos.sh"
  exit 0
fi

# --- Strategy 3: system Python (only if it's 3.10 or 3.11) ---
PYVER=$(python3 -c 'import sys; print(".".join(map(str,sys.version_info[:2])))')
case "$PYVER" in
  3.10|3.11)
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    echo "✅ venv with Python $PYVER ready. Next: bash scripts/02_install_torch_macos.sh"
    ;;
  *)
    echo "❌ Your Python is $PYVER. Please install conda or pyenv and rerun this script."
    exit 1
    ;;
esac
