"""Logging configuration for neural_priors_gym."""

import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "neural_priors_gym",
    level: int = logging.INFO,
    fmt: Optional[str] = None,
) -> logging.Logger:
    """Set up a logger with consistent formatting."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(level)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        if fmt is None:
            fmt = "[%(levelname)s] %(name)s: %(message)s"
        handler.setFormatter(logging.Formatter(fmt))

        logger.addHandler(handler)
        logger.propagate = False

    return logger


def get_logger(name: str = "neural_priors_gym") -> logging.Logger:
    """Get or create a logger for a module."""
    logger = logging.getLogger(name)

    if "." in name and not logger.handlers:
        parent_name = name.split(".")[0]
        if logging.getLogger(parent_name).handlers:
            return logger

    if not logger.handlers:
        setup_logger(name)

    return logger


def set_log_level(level: int, name: str = "neural_priors_gym") -> None:
    """Change the logging level."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


_main_logger = setup_logger("neural_priors_gym", level=logging.INFO)
