"""General-purpose utility functions.

This module provides common helper functions for logging, filesystem handling,
and basic validation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


def configure_logger(name: str = "cyber_threat_xai", level: int = logging.INFO) -> logging.Logger:
    """Create and configure a logger for the project."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float safely, returning a default when conversion fails."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
