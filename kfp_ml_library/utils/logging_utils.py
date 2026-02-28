"""
Logging utilities.

Provides a consistent logger setup for all pipeline components.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


def get_logger(
    name: str,
    level: int = logging.INFO,
    fmt: Optional[str] = None,
) -> logging.Logger:
    """
    Get (or create) a logger with a consistent format.

    Parameters
    ----------
    name : str
        Logger name, typically ``__name__``.
    level : int
        Logging level.
    fmt : str, optional
        Custom format string.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt or "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(level)
    return logger


def log_dict(logger: logging.Logger, data: dict, prefix: str = "") -> None:
    """Log each key-value pair of a dictionary."""
    for k, v in data.items():
        if isinstance(v, float):
            logger.info("%s%s: %.6f", prefix, k, v)
        else:
            logger.info("%s%s: %s", prefix, k, v)
