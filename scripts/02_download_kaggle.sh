#!/usr/bin/env bash
set -euo pipefail
# Run from project root: bash scripts/02_download_kaggle.sh
# Place kaggle.json in project root OR ~/.kaggle/kaggle.json (chmod 600)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_RUNNER="$ROOT_DIR/scripts/run_python.sh"

if [[ -n "${VIRTUAL_ENV:-}" && -n "${CONDA_PREFIX:-}" ]]; then
  echo "❌ Detected both a virtualenv ($VIRTUAL_ENV) and a conda env ($CONDA_PREFIX)." >&2
  echo "Please 'deactivate' so only one environment remains active before downloading data." >&2
  exit 1
fi

cd "$ROOT_DIR"

if "$PYTHON_RUNNER" - <<'PY'
import shutil
raise SystemExit(0 if shutil.which('kaggle') else 1)
PY
then
  echo "kaggle CLI found."
else
  echo "kaggle CLI not found. Installing into current Python environment..."
  "$PYTHON_RUNNER" -m pip install kaggle
fi

# Prefer project-root kaggle.json
if [[ -f "$ROOT_DIR/kaggle.json" ]]; then
  export KAGGLE_CONFIG_DIR="$ROOT_DIR"
  echo "Using project-root kaggle.json"
fi

"$PYTHON_RUNNER" -m src.cli.download_data
echo "✅ Download complete."
