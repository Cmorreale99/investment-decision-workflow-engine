"""End-to-end ingestion pipeline: fetch → validate → persist."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from .base import PRICE_COLUMNS, MarketDataProvider
from .providers.fixture import FixtureProvider
from .providers.yfinance_provider import YFinanceProvider
from .validation import validate_prices


def get_provider(name: str | None) -> MarketDataProvider:
    """Resolve a provider by config name. ``None`` defaults to ``fixture``."""
    resolved = (name or "fixture").lower()
    if resolved == "fixture":
        return FixtureProvider()
    if resolved == "yfinance":
        return YFinanceProvider()
    raise ValueError(
        f"unknown data_provider: {name!r} (expected 'fixture' or 'yfinance')"
    )


def run_ingestion(
    provider: MarketDataProvider,
    tickers: list[str],
    benchmark: str,
    start: date,
    end: date,
    raw_dir: Path,
    processed_dir: Path,
) -> pd.DataFrame:
    """Fetch the universe + benchmark, validate, and persist raw + processed."""
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    processed_path.mkdir(parents=True, exist_ok=True)

    symbols = sorted({*tickers, benchmark})
    df = provider.fetch(symbols, start, end)

    missing = [c for c in PRICE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"provider returned frame missing required columns: {missing}"
        )
    df = df[PRICE_COLUMNS].copy()
    validate_prices(df)

    for ticker, group in df.groupby("asset_id", sort=False):
        out = raw_path / f"{ticker}.parquet"
        group.reset_index(drop=True).to_parquet(out, index=False)

    df.to_parquet(processed_path / "prices.parquet", index=False)
    return df
