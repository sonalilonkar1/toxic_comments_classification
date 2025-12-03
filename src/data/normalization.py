"""Normalization configuration loading."""

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.data.preprocess import rich_normalize, toy_normalize

# Default path relative to project root (usually overridden by config)
DEFAULT_NORMALIZATION_CONFIG_PATH = Path("configs/normalization.yaml")


def load_normalization_config(path: Path) -> Dict[str, Any]:
    """Load normalization config from JSON or YAML."""
    if not path.exists():
        # Fallback default if file missing
        return {}
    
    if path.suffix == ".json":
        with open(path, "r") as f:
            return json.load(f)
    elif path.suffix in (".yaml", ".yml"):
        import yaml
        with open(path, "r") as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(f"Unknown config format: {path}")


def build_normalizer(config: Dict[str, Any]) -> Callable[[str], str]:
    """Build a normalization function from config dict.
    
    For now, maps 'method' string to function. Can be expanded to be a composed pipeline.
    """
    method = config.get("method", "toy")
    
    if method == "rich":
        return rich_normalize
    return toy_normalize


def compute_config_hash(config: Dict[str, Any]) -> str:
    """Compute deterministic hash of configuration for caching."""
    # Sort keys to ensure stable hash
    serialized = json.dumps(config, sort_keys=True)
    return hashlib.md5(serialized.encode("utf-8")).hexdigest()

