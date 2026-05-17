"""Provider Protocol and the canonical normalized price-frame schema.

A ``MarketDataProvider`` returns a single tidy DataFrame with one row per
(date, asset_id) pair. Concrete providers live under ``providers/``.
"""

from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable

import pandas as pd

PRICE_COLUMNS: list[str] = [
    "date",
    "asset_id",
    "open",
    "high",
    "low",
    "close",
    "volume",
]


@runtime_checkable
class MarketDataProvider(Protocol):
    """Sync OHLCV provider returning a normalized long-format DataFrame.

    The returned frame must contain exactly the columns in ``PRICE_COLUMNS``,
    have ``date`` rows sorted ascending per ``asset_id``, and have no NaN
    ``close`` values.
    """

    def fetch(
        self,
        tickers: list[str],
        start: date,
        end: date,
    ) -> pd.DataFrame: ...
