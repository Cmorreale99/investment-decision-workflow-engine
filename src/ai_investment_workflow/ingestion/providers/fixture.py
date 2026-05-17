"""Offline provider that reads a pre-saved parquet from ``tests/fixtures/``.

This is the default in test and offline use. The pipeline never talks to
the network when this provider is selected.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from ...utils.paths import project_root
from ..base import PRICE_COLUMNS


def default_fixture_path() -> Path:
    """Canonical location of the bundled price fixture."""
    return project_root() / "tests" / "fixtures" / "prices.parquet"


class FixtureProvider:
    """Read normalized OHLCV from a parquet file and filter by ticker/date."""

    def __init__(self, fixture_path: str | Path | None = None) -> None:
        self.fixture_path = (
            Path(fixture_path) if fixture_path else default_fixture_path()
        )

    def fetch(
        self,
        tickers: list[str],
        start: date,
        end: date,
    ) -> pd.DataFrame:
        if not self.fixture_path.exists():
            raise FileNotFoundError(
                f"fixture parquet not found: {self.fixture_path}"
            )

        df = pd.read_parquet(self.fixture_path)
        df["date"] = pd.to_datetime(df["date"]).dt.date

        mask = (
            df["asset_id"].isin(tickers)
            & (df["date"] >= start)
            & (df["date"] <= end)
        )
        result = df.loc[mask, PRICE_COLUMNS].copy()
        return result.sort_values(["asset_id", "date"]).reset_index(drop=True)
