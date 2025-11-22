#!/usr/bin/env bash
set -euo pipefail

# Idempotent environment bootstrapper for macOS. Prefers conda/miniforge on
# Apple Silicon, then pyenv, then the system Python 3.10/3.11. Re-runs are safe;
# the script will reuse the existing environment and simply refresh dependencies.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REQ_FILE="$ROOT_DIR/requirements.txt"
ENV_NAME="toxbench"
PYENV_VERSION="3.11.9"

if [[ -n "${VIRTUAL_ENV:-}" && -n "${CONDA_PREFIX:-}" ]]; then
  echo "❌ Detected both a virtualenv ($VIRTUAL_ENV) and a conda env ($CONDA_PREFIX)."
  echo "Please 'deactivate' so only one environment remains active before running this script."
  exit 1
fi

if [[ ! -f "$REQ_FILE" ]]; then
  echo "requirements.txt not found next to scripts/. Run from project root."
  exit 1
fi

cd "$ROOT_DIR"

run_pip_sync() {
  "$PYTHON_BIN" -m pip install --upgrade pip
  "$PYTHON_BIN" -m pip install -r "$REQ_FILE"
}

# --- Strategy 1: Conda/Miniforge (recommended on macOS, Apple Silicon) ---
if command -v conda >/dev/null 2>&1; then
  __conda_setup="$(conda shell.bash hook 2>/dev/null)" || true
  eval "$__conda_setup"
  unset __conda_setup

  if conda env list | grep -Eq "^\s*$ENV_NAME\s"; then
    echo "Reusing existing conda env '$ENV_NAME'."
  else
    echo "Creating conda env '$ENV_NAME' with Python 3.11..."
    conda create -y -n "$ENV_NAME" python=3.11
  fi
  conda activate "$ENV_NAME"
  PYTHON_BIN="$CONDA_PREFIX/bin/python"
  run_pip_sync
  echo "✅ Conda env '$ENV_NAME' ready."
  echo "Next: bash scripts/02_install_torch_macos.sh"
  echo "Tip: run Python via scripts/run_python.sh to avoid shell aliases."
  exit 0
fi

# --- Strategy 2: pyenv (no conda) ---
if command -v pyenv >/dev/null 2>&1; then
  pyenv install -s "$PYENV_VERSION"
  pyenv local "$PYENV_VERSION"
  if [[ ! -d "$ROOT_DIR/.venv" ]]; then
    echo "Creating .venv with pyenv-provided Python..."
    python3 -m venv "$ROOT_DIR/.venv"
  else
    echo "Reusing existing .venv"
  fi
  # shellcheck source=/dev/null
  source "$ROOT_DIR/.venv/bin/activate"
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
  run_pip_sync
  echo "✅ venv with $(python -V) ready."
  echo "Next: bash scripts/02_install_torch_macos.sh"
  echo "Tip: run Python via scripts/run_python.sh to avoid shell aliases."
  exit 0
fi

# --- Strategy 3: system Python (only if it's 3.10 or 3.11) ---
PYVER=$(python3 -c 'import sys; print(".".join(map(str,sys.version_info[:2])))')
case "$PYVER" in
  3.10|3.11)
    if [[ ! -d "$ROOT_DIR/.venv" ]]; then
      echo "Creating .venv with system Python $PYVER..."
      python3 -m venv "$ROOT_DIR/.venv"
    else
      echo "Reusing existing .venv"
    fi
    # shellcheck source=/dev/null
    source "$ROOT_DIR/.venv/bin/activate"
    PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
    run_pip_sync
    echo "✅ venv with Python $PYVER ready."
    echo "Next: bash scripts/02_install_torch_macos.sh"
    echo "Tip: run Python via scripts/run_python.sh to avoid shell aliases."
    exit 0
    ;;
  *)
    echo "❌ Your Python is $PYVER. Please install conda (recommended) or pyenv and rerun this script."
    exit 1
    ;;
esac

