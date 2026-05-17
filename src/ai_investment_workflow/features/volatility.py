"""Realized volatility: 21d / 63d annualized stdev of daily log returns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import KEY_COLUMNS, validate_price_frame

WINDOWS: dict[str, int] = {
    "volatility_21d": 21,
    "volatility_63d": 63,
}
TRADING_DAYS_PER_YEAR: int = 252


class VolatilityBuilder:
    """Annualized stdev of daily log returns over the trailing window.

    Insufficient history → NaN. Output is always non-negative when defined.
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
        prev_close = df.groupby("asset_id", sort=False)["close"].shift(1)
        df["_log_ret"] = np.log(df["close"] / prev_close)

        scale = float(np.sqrt(TRADING_DAYS_PER_YEAR))
        for name, window in WINDOWS.items():
            rolling = (
                df.groupby("asset_id", sort=False)["_log_ret"]
                .rolling(window)
                .std()
                .reset_index(level=0, drop=True)
            )
            df[name] = rolling * scale

        return df[[*KEY_COLUMNS, *self.feature_names]].copy()
