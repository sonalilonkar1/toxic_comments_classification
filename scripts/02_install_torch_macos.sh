#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="toxbench"

cd "$ROOT_DIR"

if [[ -n "${VIRTUAL_ENV:-}" && -n "${CONDA_PREFIX:-}" ]]; then
  echo "❌ Detected both a virtualenv ($VIRTUAL_ENV) and a conda env ($CONDA_PREFIX)." >&2
  echo "Please 'deactivate' so only one environment remains active before installing PyTorch." >&2
  exit 1
fi

activate_conda_env() {
  __conda_setup="$(conda shell.bash hook 2>/dev/null)" || true
  eval "$__conda_setup"
  unset __conda_setup
  conda activate "$ENV_NAME"
}

set_python_bin() {
  if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
    PYTHON_BIN="$CONDA_PREFIX/bin/python"
    return
  fi
  if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    PYTHON_BIN="$VIRTUAL_ENV/bin/python"
    return
  fi
  if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
    return
  fi
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
    return
  fi
  echo "❌ Unable to find a usable Python interpreter." >&2
  exit 1
}

activate_python() {
  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    echo "Using already-activated venv: $VIRTUAL_ENV"
    set_python_bin
    return
  fi

  if command -v conda >/dev/null 2>&1 && conda env list | grep -Eq "^\s*$ENV_NAME\s"; then
    echo "Activating conda env '$ENV_NAME'..."
    activate_conda_env
    set_python_bin
    return
  fi

  if [[ -d "$ROOT_DIR/.venv" ]]; then
    echo "Activating $ROOT_DIR/.venv ..."
    # shellcheck source=/dev/null
    source "$ROOT_DIR/.venv/bin/activate"
    set_python_bin
    return
  fi

  echo "❌ No active Python environment found. Run bash scripts/01_make_env_macos.sh first." >&2
  exit 1
}

activate_python

"$PYTHON_BIN" - <<'PY'
import sys, platform
print("Python:", sys.version)
print("Arch  :", platform.machine())
print("MacOS :", platform.system(), platform.mac_ver()[0])
PY

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  echo "Installing PyTorch via conda-forge..."
  conda install -y -c conda-forge pytorch torchvision torchaudio
  "$PYTHON_BIN" -c "import torch; print('Torch:', torch.__version__, 'MPS:', torch.backends.mps.is_available())"
  echo "✅ Installed PyTorch via conda-forge."
  echo "Use scripts/run_python.sh -c 'import torch' to verify from any shell."
  exit 0
fi

echo "Installing PyTorch via pip..."
"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install torch torchvision torchaudio
# "$PYTHON_BIN" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

"$PYTHON_BIN" - <<'PY'
import torch, platform
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("MPS available:", hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
PY

echo "✅ Installed PyTorch via pip."
