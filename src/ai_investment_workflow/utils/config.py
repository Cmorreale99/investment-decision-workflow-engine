"""YAML config loaders.

Returns plain dicts/lists; typed config models can be layered on later
without changing this module's surface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .paths import config_dir


def load_yaml(path: str | Path) -> Any:
    """Load any YAML file by path."""
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_named(name: str) -> Any:
    return load_yaml(config_dir() / name)


def load_settings() -> dict[str, Any]:
    data = _load_named("settings.yaml")
    if not isinstance(data, dict) or "settings" not in data:
        raise ValueError("settings.yaml must contain a top-level 'settings' mapping")
    return data["settings"]


def load_universe() -> list[str]:
    data = _load_named("universe.yaml")
    if not isinstance(data, dict) or "universe" not in data:
        raise ValueError("universe.yaml must contain a top-level 'universe' list")
    tickers = data["universe"]
    if not isinstance(tickers, list) or not tickers:
        raise ValueError("universe.yaml 'universe' must be a non-empty list")
    return list(tickers)


def load_strategies() -> dict[str, Any]:
    data = _load_named("strategies.yaml")
    if not isinstance(data, dict) or "strategies" not in data:
        raise ValueError("strategies.yaml must contain a top-level 'strategies' mapping")
    return data


def load_risk_rules() -> dict[str, Any]:
    data = _load_named("risk_rules.yaml")
    if not isinstance(data, dict) or "risk_rules" not in data:
        raise ValueError("risk_rules.yaml must contain a top-level 'risk_rules' mapping")
    return data["risk_rules"]
