"""Common data-handling utilities shared across all pipelines."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def compute_file_hash(path: Path, algorithm: str = "sha256") -> str:
    """Compute hex digest of a file in 64 KiB chunks."""
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def to_native(obj: Any) -> Any:
    """Recursively convert numpy/pydantic types to JSON-safe Python types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    return obj


def save_json(data: Any, path: Path) -> Path:
    """Write a JSON file with native-type conversion."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(to_native(data), f, indent=2)
    logger.info("Saved %s", path)
    return path


def load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)
