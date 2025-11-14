#!/usr/bin/env bash
set -euo pipefail
# Run from project root: bash scripts/01_make_venv.sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo "✅ Venv ready. Activate with: source .venv/bin/activate"
