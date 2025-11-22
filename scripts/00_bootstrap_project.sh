#!/usr/bin/env bash
set -euo pipefail

# One-shot setup helper that wires up the Python environment, installs PyTorch, and
# downloads Kaggle data. Pass --skip-* flags to bypass individual steps after the
# first successful run.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_SCRIPT="$ROOT_DIR/scripts/01_make_env_macos.sh"
TORCH_SCRIPT="$ROOT_DIR/scripts/02_install_torch_macos.sh"
DATA_SCRIPT="$ROOT_DIR/scripts/02_download_kaggle.sh"

if [[ -n "${VIRTUAL_ENV:-}" && -n "${CONDA_PREFIX:-}" ]]; then
  echo "❌ Detected both a virtualenv ($VIRTUAL_ENV) and a conda env ($CONDA_PREFIX)." >&2
  echo "Please 'deactivate' any virtualenv before running the bootstrap to avoid pip installing into the wrong interpreter." >&2
  exit 1
fi

usage() {
  cat <<'EOF'
Usage: bash scripts/00_bootstrap_project.sh [options]

Orchestrates the full first-time setup:
  1. Create/refresh the Python environment
  2. Install PyTorch bindings optimized for macOS
  3. Download the Kaggle dataset

Options:
  --skip-env      Skip the environment step (e.g., if already configured)
  --skip-torch    Skip the PyTorch install step
  --skip-data     Skip the Kaggle download step
  -h, --help      Show this help message
EOF
}

SKIP_ENV=0
SKIP_TORCH=0
SKIP_DATA=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-env) SKIP_ENV=1 ; shift ;;
    --skip-torch) SKIP_TORCH=1 ; shift ;;
    --skip-data) SKIP_DATA=1 ; shift ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

run_step() {
  local label="$1"
  local script_path="$2"
  shift 2
  printf "\n=== %s ===\n" "$label"
  bash "$script_path" "$@"
}

cd "$ROOT_DIR"

if [[ $SKIP_ENV -eq 0 ]]; then
  run_step "Environment setup" "$ENV_SCRIPT"
else
  echo "Skipping environment setup (per flag)."
fi

if [[ $SKIP_TORCH -eq 0 ]]; then
  run_step "PyTorch install" "$TORCH_SCRIPT"
else
  echo "Skipping PyTorch install (per flag)."
fi

if [[ $SKIP_DATA -eq 0 ]]; then
  run_step "Kaggle data download" "$DATA_SCRIPT"
else
  echo "Skipping Kaggle download (per flag)."
fi

printf "\n✅ Bootstrap complete. You're ready to work in %s\n" "$ROOT_DIR"
