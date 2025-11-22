#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="toxbench"

# Prevent using multiple environments at once.
if [[ -n "${VIRTUAL_ENV:-}" && -n "${CONDA_PREFIX:-}" ]]; then
  echo "❌ Detected both a virtualenv ($VIRTUAL_ENV) and a conda env ($CONDA_PREFIX)." >&2
  echo "Please 'deactivate' and/or 'conda deactivate' so only one environment remains active." >&2
  exit 1
fi

# Prefer toxbench conda env, then project .venv, else fall back to system python.
if command -v conda >/dev/null 2>&1 && conda env list | grep -Eq "^\s*$ENV_NAME\s"; then
  exec conda run -n "$ENV_NAME" python "$@"
elif [[ -n "${CONDA_PREFIX:-}" ]]; then
  exec "$CONDA_PREFIX/bin/python" "$@"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
  exec "$VIRTUAL_ENV/bin/python" "$@"
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  exec "$ROOT_DIR/.venv/bin/python" "$@"
elif command -v python3 >/dev/null 2>&1; then
  exec python3 "$@"
elif command -v python >/dev/null 2>&1; then
  exec python "$@"
else
  echo "❌ Could not find a Python interpreter. Install Python 3.10+ or rerun scripts/01_make_env_macos.sh." >&2
  exit 1
fi
