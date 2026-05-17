"""Basic logger setup. Phase 1 is intentionally minimal."""

from __future__ import annotations

import logging
import os

_DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def setup_logging(level: str | int | None = None) -> None:
    """Configure the root logger once. Idempotent."""
    if level is None:
        level = os.environ.get("LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())
    logging.basicConfig(level=level, format=_DEFAULT_FORMAT)


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger."""
    return logging.getLogger(name)
