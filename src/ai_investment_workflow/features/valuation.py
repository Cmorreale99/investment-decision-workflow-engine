"""Valuation proxies derived from close prices only.

Real valuation features (P/E, EV/EBITDA, P/B) require fundamentals that
the Phase 2 ingestion layer does not yet pull. This module is a
placeholder using 52-week high/low distance as proxies.

# TODO: fundamentals
#   Replace / augment with P/E, EV/EBITDA, P/B, dividend yield, etc. once
#   an external fundamentals source is wired (likely Phase 2.5 or via a
#   second MarketDataProvider variant). Until then, valuation features
#   should be treated as low-information by downstream strategies.
"""

from __future__ import annotations

import pandas as pd

from .base import KEY_COLUMNS, validate_price_frame

LOOKBACK_DAYS: int = 252  # ~ 52 weeks of trading days


class ValuationBuilder:
    """52-week price-position proxies.

    ``distance_from_52w_high`` = ``(close - high_52w) / high_52w`` (≤ 0).
    ``distance_from_52w_low``  = ``(close - low_52w)  / low_52w``  (≥ 0).

    Rolling windows use ``min_periods=1`` so early rows return a coarse
    high/low based on available history rather than NaN — sufficient for
    proxy use, but noisy until ~252 days of data accumulate.
    """

    feature_names: tuple[str, ...] = (
        "distance_from_52w_high",
        "distance_from_52w_low",
    )

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        validate_price_frame(prices)
        df = (
            prices[[*KEY_COLUMNS, "close"]]
            .sort_values(["asset_id", "date"])
            .reset_index(drop=True)
            .copy()
        )
        rolling_max = (
            df.groupby("asset_id", sort=False)["close"]
            .rolling(LOOKBACK_DAYS, min_periods=1)
            .max()
            .reset_index(level=0, drop=True)
        )
        rolling_min = (
            df.groupby("asset_id", sort=False)["close"]
            .rolling(LOOKBACK_DAYS, min_periods=1)
            .min()
            .reset_index(level=0, drop=True)
        )
        df["distance_from_52w_high"] = (df["close"] - rolling_max) / rolling_max
        df["distance_from_52w_low"] = (df["close"] - rolling_min) / rolling_min
        return df[[*KEY_COLUMNS, *self.feature_names]].copy()
