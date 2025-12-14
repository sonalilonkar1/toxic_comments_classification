"""Bucket configuration and utilities."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_BUCKET_CONFIG_PATH = Path("configs/buckets.yaml")


def load_bucket_config(path: Path) -> Dict[str, Any]:
    """Load bucket configuration."""
    if not path.exists():
        return {}
        
    if path.suffix == ".json":
        with open(path, "r") as f:
            return json.load(f)
    elif path.suffix in (".yaml", ".yml"):
        import yaml
        with open(path, "r") as f:
            return yaml.safe_load(f)
    return {}


def compute_bucket_hash(config: Dict[str, Any]) -> str:
    """Compute hash of bucket configuration."""
    serialized = json.dumps(config, sort_keys=True)
    return hashlib.md5(serialized.encode("utf-8")).hexdigest()




