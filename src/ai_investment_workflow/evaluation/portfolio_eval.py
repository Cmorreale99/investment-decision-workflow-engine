"""Portfolio-level diagnostics built on top of Phase 5 simulation output."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..simulation import PortfolioState

ROLLING_HIT_RATE_WINDOW: int = 21


def _longest_run(mask: pd.Series) -> int:
    """Longest run of consecutive ``True`` values."""
    longest = 0
    current = 0
    for v in mask.to_numpy():
        if bool(v):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def evaluate_portfolio(
    state: PortfolioState,
    benchmark_returns: pd.Series,
    excess_returns: pd.Series,
    rolling_window: int = ROLLING_HIT_RATE_WINDOW,
) -> dict[str, Any]:
    """Return a **flat scalar** diagnostics map suitable for single-row parquet."""
    returns = state.returns

    if returns.empty:
        return {
            "n_trading_days": 0,
            "n_winning_days": 0,
            "n_losing_days": 0,
            "rolling_hit_rate_mean": 0.0,
            "rolling_hit_rate_min": 0.0,
            "rolling_hit_rate_max": 0.0,
            "longest_losing_streak_days": 0,
            "longest_winning_streak_days": 0,
            "max_drawdown_duration_days": 0,
            "monthly_hit_rate_min": 0.0,
            "monthly_hit_rate_max": 0.0,
            "monthly_hit_rate_mean": 0.0,
            "total_excess_return": 0.0,
        }

    rolling_hit = (returns > 0).rolling(rolling_window).mean().dropna()

    losing_streak = _longest_run(returns < 0)
    winning_streak = _longest_run(returns > 0)

    cumulative = (1.0 + returns).cumprod()
    peak = cumulative.cummax()
    in_drawdown = cumulative < peak
    dd_duration = _longest_run(in_drawdown)

    monthly_rates: dict[str, float] = {}
    if isinstance(returns.index, pd.DatetimeIndex):
        for month, group in returns.groupby(returns.index.to_period("M")):
            monthly_rates[str(month)] = float((group > 0).mean())

    monthly_values = list(monthly_rates.values())

    total_excess = (
        float((1.0 + excess_returns).prod() - 1.0)
        if not excess_returns.empty
        else 0.0
    )

    return {
        "n_trading_days": int(len(returns)),
        "n_winning_days": int((returns > 0).sum()),
        "n_losing_days": int((returns < 0).sum()),
        "rolling_hit_rate_mean": float(rolling_hit.mean()) if not rolling_hit.empty else 0.0,
        "rolling_hit_rate_min": float(rolling_hit.min()) if not rolling_hit.empty else 0.0,
        "rolling_hit_rate_max": float(rolling_hit.max()) if not rolling_hit.empty else 0.0,
        "longest_losing_streak_days": int(losing_streak),
        "longest_winning_streak_days": int(winning_streak),
        "max_drawdown_duration_days": int(dd_duration),
        "monthly_hit_rate_min": float(min(monthly_values)) if monthly_values else 0.0,
        "monthly_hit_rate_max": float(max(monthly_values)) if monthly_values else 0.0,
        "monthly_hit_rate_mean": (
            float(sum(monthly_values) / len(monthly_values)) if monthly_values else 0.0
        ),
        "total_excess_return": total_excess,
    }
