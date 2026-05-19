"""Per-decision asset evaluation → list[PerformanceRecord]."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from ..schemas import OutcomeLabel, PerformanceRecord
from .windows import (
    calendar_to_trading_days,
    forward_window_max_drawdown,
    forward_window_return,
    outcome_label,
)


def _decision_id(rebal_date: date, asset_id: str) -> str:
    """Phase 1 architecture-doc convention: ``dec_{YYYYMMDD}_{ASSET}``."""
    return f"dec_{rebal_date.strftime('%Y%m%d')}_{asset_id}"


def _rebalance_dates(
    rankings_history: pd.DataFrame, rebalance_day: str
) -> list[date]:
    df = rankings_history.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["_week"] = df["date"].dt.to_period(rebalance_day)
    return (
        df.groupby("_week")["date"].max().sort_values().dt.date.tolist()
    )


def evaluate_asset_decisions(
    rankings_history: pd.DataFrame,
    prices: pd.DataFrame,
    evaluation_window_days: int,
    benchmark: str = "SPY",
    top_n: int = 10,
    rebalance_day: str = "W-FRI",
    neutral_band: float = 0.001,
) -> list[PerformanceRecord]:
    """One ``PerformanceRecord`` per (rebalance_date, top-N pick).

    ``evaluation_window_days`` is a **calendar** window per
    ``settings.yaml``; it is converted to the nearest trading-day window
    for the price-based math.

    Decisions whose forward window extends past available history are
    emitted as ``OutcomeLabel.PENDING`` with zeroed numeric fields, so
    they remain schema-valid (``max_drawdown <= 0`` and
    ``evaluation_end >= evaluation_start``).
    """
    if rankings_history.empty:
        return []

    trading_days = calendar_to_trading_days(int(evaluation_window_days))
    rebal_dates = _rebalance_dates(rankings_history, rebalance_day)

    records: list[PerformanceRecord] = []
    for rebal_date in rebal_dates:
        slice_ = rankings_history.loc[rankings_history["date"] == rebal_date]
        picks = slice_.sort_values(["rank", "asset_id"]).head(top_n)
        for row in picks.itertuples(index=False):
            asset_id = row.asset_id

            asset_ret = forward_window_return(
                prices, asset_id, rebal_date, trading_days
            )
            bench_ret = forward_window_return(
                prices, benchmark, rebal_date, trading_days
            )

            if asset_ret is None or bench_ret is None:
                asset_ret_val = 0.0
                bench_ret_val = 0.0
                excess_val = 0.0
                dd_val = 0.0
                label = OutcomeLabel.PENDING
            else:
                excess = asset_ret - bench_ret
                dd_raw = forward_window_max_drawdown(
                    prices, asset_id, rebal_date, trading_days
                )
                asset_ret_val = float(asset_ret)
                bench_ret_val = float(bench_ret)
                excess_val = float(excess)
                # Schema requires max_drawdown <= 0; clip any +epsilon noise.
                dd_val = float(min(0.0, dd_raw if dd_raw is not None else 0.0))
                label = outcome_label(excess, neutral_band=neutral_band)

            end_date = rebal_date + timedelta(days=int(evaluation_window_days))
            records.append(
                PerformanceRecord(
                    decision_id=_decision_id(rebal_date, asset_id),
                    asset_id=asset_id,
                    evaluation_start=rebal_date,
                    evaluation_end=end_date,
                    asset_return=asset_ret_val,
                    benchmark_return=bench_ret_val,
                    excess_return=excess_val,
                    max_drawdown=dd_val,
                    outcome_label=label,
                )
            )
    return records
