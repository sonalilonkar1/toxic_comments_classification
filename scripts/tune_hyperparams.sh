#!/usr/bin/env bash
set -euo pipefail

# Run hyperparameter tuning
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="toxbench"

if command -v conda >/dev/null 2>&1 && conda env list | grep -Eq "^\s*$ENV_NAME\s"; then
  cd "$ROOT_DIR"
  echo "Running with conda run"
  exec conda run -n "$ENV_NAME" python -m src.pipeline.tune "$@"
elif [[ -n "${CONDA_PREFIX:-}" ]]; then
  cd "$ROOT_DIR"
  echo "Running with CONDA_PREFIX"
  exec "$CONDA_PREFIX/bin/python" -m src.pipeline.tune "$@"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
  cd "$ROOT_DIR"
  echo "Running with VIRTUAL_ENV"
  exec "$VIRTUAL_ENV/bin/python" -m src.pipeline.tune "$@"
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  cd "$ROOT_DIR"
  echo "Running with .venv"
  exec "$ROOT_DIR/.venv/bin/python" -m src.pipeline.tune "$@"
elif command -v python3 >/dev/null 2>&1; then
  cd "$ROOT_DIR"
  echo "Running with python3"
  exec python3 -m src.pipeline.tune "$@"
elif command -v python >/dev/null 2>&1; then
  cd "$ROOT_DIR"
  echo "Running with python"
  exec python -m src.pipeline.tune "$@"
else
  echo "❌ Could not find a Python interpreter. Install Python 3.10+ or rerun scripts/01_make_env_macos.sh." >&2
  exit 1
fi