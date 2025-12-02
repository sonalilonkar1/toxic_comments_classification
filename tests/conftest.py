"""Pytest configuration helpers."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repository root (which contains the `src` package) is importable when
# pytest is invoked directly without `PYTHONPATH` adjustments.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
