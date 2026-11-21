"""Structured logging utility using standard logging and rich formatting."""

import logging
import sys
from typing import Optional


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Configures and returns a structured logger with standardized formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(numeric_level)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(numeric_level)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger
