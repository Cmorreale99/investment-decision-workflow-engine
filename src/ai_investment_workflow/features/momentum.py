"""Momentum features: total returns over 1m/3m/6m/12m trading-day windows."""

from __future__ import annotations

import pandas as pd

from .base import KEY_COLUMNS, validate_price_frame

WINDOWS: dict[str, int] = {
    "momentum_1m": 21,
    "momentum_3m": 63,
    "momentum_6m": 126,
    "momentum_12m": 252,
}


class MomentumBuilder:
    """``close[t] / close[t - n] - 1`` for n in {21, 63, 126, 252}.

    Insufficient history → NaN for that row.
    """

    feature_names: tuple[str, ...] = tuple(WINDOWS.keys())

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        validate_price_frame(prices)
        df = (
            prices[[*KEY_COLUMNS, "close"]]
            .sort_values(["asset_id", "date"])
            .reset_index(drop=True)
            .copy()
        )
        grouped = df.groupby("asset_id", sort=False)["close"]
        for name, window in WINDOWS.items():
            df[name] = grouped.pct_change(periods=window)
        return df[[*KEY_COLUMNS, *self.feature_names]].copy()
