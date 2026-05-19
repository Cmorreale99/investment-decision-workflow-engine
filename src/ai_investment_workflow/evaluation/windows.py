"""Pure forward-window helpers for evaluation.

No schema emission. Used by ``asset_eval`` to compute per-decision
outcomes and by ``performance_by_strategy`` for attribution joins.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd

from ..schemas import OutcomeLabel

TRADING_DAYS_PER_YEAR: int = 252
CALENDAR_DAYS_PER_YEAR: int = 365


def calendar_to_trading_days(calendar_days: int) -> int:
    """Convert a calendar-day window to its nearest trading-day equivalent.

    30 calendar days → 21 trading days (≈ 30 × 252/365).
    """
    if calendar_days <= 0:
        return 0
    return max(1, round(calendar_days * TRADING_DAYS_PER_YEAR / CALENDAR_DAYS_PER_YEAR))


def _asset_series(prices: pd.DataFrame, asset_id: str) -> pd.DataFrame:
    return (
        prices.loc[prices["asset_id"] == asset_id, ["date", "close"]]
        .sort_values("date")
        .reset_index(drop=True)
    )


def forward_window_return(
    prices: pd.DataFrame,
    asset_id: str,
    start: date,
    trading_days: int,
) -> float | None:
    """Total return from the close at ``start`` to ``trading_days`` ahead.

    Returns ``None`` when ``start`` is not a trading day for ``asset_id``
    or when the window extends past available history.
    """
    asset = _asset_series(prices, asset_id)
    if asset.empty:
        return None
    matches = asset.index[asset["date"] == start]
    if len(matches) == 0:
        return None
    start_idx = int(matches[0])
    end_idx = start_idx + int(trading_days)
    if end_idx >= len(asset):
        return None
    start_close = float(asset.iloc[start_idx]["close"])
    end_close = float(asset.iloc[end_idx]["close"])
    if start_close <= 0 or not math.isfinite(start_close) or not math.isfinite(end_close):
        return None
    return end_close / start_close - 1.0


def forward_window_max_drawdown(
    prices: pd.DataFrame,
    asset_id: str,
    start: date,
    trading_days: int,
) -> float | None:
    """Most-negative peak-to-trough drawdown across the forward window.

    Returns ``None`` when ``start`` is not a trading day; returns ``0.0``
    when the window contains a single observation (no opportunity to draw
    down). Truncates to available history rather than returning ``None``.
    """
    asset = _asset_series(prices, asset_id)
    if asset.empty:
        return None
    matches = asset.index[asset["date"] == start]
    if len(matches) == 0:
        return None
    start_idx = int(matches[0])
    end_idx = min(start_idx + int(trading_days), len(asset) - 1)
    if end_idx <= start_idx:
        return 0.0
    closes = asset.iloc[start_idx : end_idx + 1]["close"].to_numpy(dtype=float)
    if (closes <= 0).any() or not np.isfinite(closes).all():
        return None
    cumulative = closes / closes[0]
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    return float(drawdown.min())


def outcome_label(
    excess_return: float | None,
    neutral_band: float = 0.001,
) -> OutcomeLabel:
    """Map an excess return to an ``OutcomeLabel``.

    - ``None`` or NaN → ``PENDING``
    - ``excess_return > +neutral_band`` → ``OUTPERFORMED``
    - ``excess_return < -neutral_band`` → ``UNDERPERFORMED``
    - otherwise → ``NEUTRAL``
    """
    if excess_return is None:
        return OutcomeLabel.PENDING
    if isinstance(excess_return, float) and math.isnan(excess_return):
        return OutcomeLabel.PENDING
    if excess_return > neutral_band:
        return OutcomeLabel.OUTPERFORMED
    if excess_return < -neutral_band:
        return OutcomeLabel.UNDERPERFORMED
    return OutcomeLabel.NEUTRAL
