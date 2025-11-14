#!/usr/bin/env bash
set -euo pipefail

# Must be run inside the conda env or venv created in 01_make_env_macos.sh
python - <<'PY'
import sys, platform
print("Python:", sys.version)
print("Arch  :", platform.machine())
print("MacOS :", platform.system(), platform.mac_ver()[0])
PY

# A) If you used CONDA (recommended on macOS):
if command -v conda >/dev/null 2>&1; then
  # Prefer conda-forge for up-to-date macOS arm64 builds with MPS
  conda install -y -c conda-forge pytorch torchvision torchaudio
  echo "✅ Installed PyTorch via conda-forge (MPS supported on Apple Silicon)."
  python -c "import torch; print('Torch:', torch.__version__, 'MPS:', torch.backends.mps.is_available())"
  exit 0
fi

# B) If you used VENV (pip):
# Try native mac wheels first (should provide MPS on Apple Silicon in recent versions):
pip install --upgrade pip
pip install torch torchvision torchaudio

# If that fails, you can fall back to CPU-only wheels:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo "✅ Installed PyTorch via pip."
python - <<'PY'
import torch, platform
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("MPS available:", hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
PY
