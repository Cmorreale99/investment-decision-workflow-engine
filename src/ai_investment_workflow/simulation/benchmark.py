"""Daily benchmark return series from the normalized price frame."""

from __future__ import annotations

import pandas as pd


def benchmark_returns(prices: pd.DataFrame, benchmark: str = "SPY") -> pd.Series:
    """Return daily total returns for ``benchmark`` indexed by trading date."""
    df = prices.loc[prices["asset_id"] == benchmark].copy()
    if df.empty:
        raise ValueError(f"benchmark {benchmark!r} not present in prices frame")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    prev = df["close"].shift(1)
    return (
        (df["close"] / prev - 1.0)
        .fillna(0.0)
        .rename("benchmark_return")
    )
