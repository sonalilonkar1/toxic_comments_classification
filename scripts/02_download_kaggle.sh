#!/usr/bin/env bash
set -euo pipefail
# Run from project root: bash scripts/02_download_kaggle.sh
# Place kaggle.json in project root OR ~/.kaggle/kaggle.json (chmod 600)

if command -v kaggle >/dev/null 2>&1; then
  echo "kaggle CLI found."
else
  echo "kaggle CLI not found. Installing..."
  pip install kaggle
fi

# Prefer project-root kaggle.json
if [[ -f "kaggle.json" ]]; then
  export KAGGLE_CONFIG_DIR="$(pwd)"
  echo "Using project-root kaggle.json"
fi

python -m src.cli.download_data
echo "✅ Download complete."
