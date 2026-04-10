"""Unified logging configuration for xarm_toolkit."""

import logging
import sys


def get_logger(name: str = "xarm_toolkit", level: int = logging.INFO) -> logging.Logger:
    """Get a configured logger instance.

    Args:
        name: Logger name (dot-separated hierarchy).
        level: Logging level (default: INFO).

    Returns:
        A configured logging.Logger.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)

    logger.setLevel(level)
    return logger
