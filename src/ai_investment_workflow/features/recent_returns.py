"""Recent returns: 5d, 21d, 63d total returns.

Note: ``return_21d`` is mathematically identical to ``momentum_1m`` but
exposed separately because downstream signals reference it under this name.
"""

from __future__ import annotations

import pandas as pd

from .base import KEY_COLUMNS, validate_price_frame

WINDOWS: dict[str, int] = {
    "return_5d": 5,
    "return_21d": 21,
    "return_63d": 63,
}


class RecentReturnsBuilder:
    """``close[t] / close[t - n] - 1`` for n in {5, 21, 63}.

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
