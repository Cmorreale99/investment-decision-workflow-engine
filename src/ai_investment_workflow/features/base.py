"""Feature builder Protocol and shared schema constants.

Output format
-------------
All builders emit **wide-format** DataFrames keyed by ``(date, asset_id)``,
with one column per feature. Rationale: wide format makes it cheap to
slice "all features at date T", to join with prices, and to feed a
ranking step in Phase 4. NaN cells are allowed when input history is too
short for the lookback window.

Required input columns
----------------------
``date``, ``asset_id``, ``close`` (other OHLCV columns are ignored). Input
must be the normalized price frame produced by Phase 2 ingestion.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

KEY_COLUMNS: list[str] = ["date", "asset_id"]
REQUIRED_PRICE_COLUMNS: set[str] = {"date", "asset_id", "close"}


@runtime_checkable
class FeatureBuilder(Protocol):
    """Stateless transform from a price frame to one or more features."""

    feature_names: tuple[str, ...]

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame: ...


def validate_price_frame(prices: pd.DataFrame) -> None:
    """Raise ``ValueError`` if the input is missing required columns."""
    missing = REQUIRED_PRICE_COLUMNS - set(prices.columns)
    if missing:
        raise ValueError(
            f"price frame missing required columns for feature building: {sorted(missing)}"
        )
