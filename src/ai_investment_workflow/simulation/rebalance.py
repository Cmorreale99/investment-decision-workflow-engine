"""Weekly rebalance schedule generation from a rankings history frame."""

from __future__ import annotations

import pandas as pd

from .base import Portfolio, equal_weight_topn


def rebalance_weekly(
    rankings_history: pd.DataFrame,
    top_n: int,
    rebalance_day: str = "W-FRI",
    exclude_conflicts: bool = False,
) -> list[Portfolio]:
    """One ``Portfolio`` per week, formed at the last ranking date in that week.

    Parameters
    ----------
    rankings_history
        Long format: ``date, asset_id, composite_score, rank, strategy_conflict``.
    top_n
        Number of equal-weighted positions.
    rebalance_day
        Pandas period alias (e.g. ``"W-FRI"`` for weeks ending Friday).
    exclude_conflicts
        If True, drop rows where ``strategy_conflict`` is true before
        picking the top N.
    """
    if rankings_history.empty:
        return []

    df = rankings_history.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["_week"] = df["date"].dt.to_period(rebalance_day)

    rebal_dates = (
        df.groupby("_week")["date"]
        .max()
        .sort_values()
        .dt.date
        .tolist()
    )

    portfolios: list[Portfolio] = []
    for rebal_date in rebal_dates:
        slice_ = rankings_history.loc[rankings_history["date"] == rebal_date]
        weights = equal_weight_topn(
            slice_, top_n=top_n, exclude_conflicts=exclude_conflicts
        )
        if not weights:
            continue
        cash_weight = max(0.0, 1.0 - sum(weights.values()))
        portfolios.append(
            Portfolio(as_of=rebal_date, weights=weights, cash=cash_weight)
        )
    return portfolios
