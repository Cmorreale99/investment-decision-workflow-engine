"""Pure return-series metrics. No I/O, no schema emission."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR: int = 252


def total_return(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    return float((1.0 + returns).prod() - 1.0)


def annualized_return(
    returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    n = len(returns)
    if n == 0:
        return 0.0
    cumulative = float((1.0 + returns).prod())
    if cumulative <= 0.0:
        return -1.0
    return float(cumulative ** (periods_per_year / n) - 1.0)


def volatility(
    returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    if len(returns) < 2:
        return 0.0
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    """Most-negative peak-to-trough return on the cumulative curve.

    Returns 0.0 when the series is empty or has no drawdown.
    """
    if returns.empty:
        return 0.0
    cumulative = (1.0 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return float(drawdown.min())


def hit_rate(returns: pd.Series) -> float:
    """Fraction of return observations strictly greater than zero."""
    if returns.empty:
        return 0.0
    return float((returns > 0).mean())
