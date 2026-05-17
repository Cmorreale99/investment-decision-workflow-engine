"""Invariants every ingested price frame must satisfy."""

from __future__ import annotations

import pandas as pd

from .base import PRICE_COLUMNS


def validate_prices(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` if the frame breaks any normalization invariant."""
    missing = [c for c in PRICE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"price frame missing required columns: {missing}")

    if df.empty:
        raise ValueError("price frame is empty")

    if df["close"].isna().any():
        raise ValueError("close column contains NaN values")

    if df.duplicated(subset=["date", "asset_id"]).any():
        raise ValueError("duplicate (date, asset_id) rows detected")

    for ticker, group in df.groupby("asset_id", sort=False):
        if not group["date"].is_monotonic_increasing:
            raise ValueError(
                f"dates are not monotonically increasing for asset_id={ticker!r}"
            )
