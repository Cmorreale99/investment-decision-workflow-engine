"""Repo and data path resolution.

All paths are derived from the package location so the project is portable
across machines and operating systems.
"""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Absolute path to the repository root."""
    return Path(__file__).resolve().parents[3]


def config_dir() -> Path:
    return project_root() / "config"


def data_dir() -> Path:
    """Honors the ``DATA_DIR`` env var; otherwise ``<root>/data``."""
    env_dir = os.environ.get("DATA_DIR")
    if env_dir:
        candidate = Path(env_dir)
        if not candidate.is_absolute():
            candidate = project_root() / candidate
        return candidate
    return project_root() / "data"


def raw_dir() -> Path:
    return data_dir() / "raw"


def interim_dir() -> Path:
    return data_dir() / "interim"


def processed_dir() -> Path:
    return data_dir() / "processed"


def embeddings_dir() -> Path:
    return data_dir() / "embeddings"
