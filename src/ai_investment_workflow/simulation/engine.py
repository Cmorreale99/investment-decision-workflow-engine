"""Paper portfolio engine: step forward daily and rebalance on schedule."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .base import Portfolio, PortfolioState, apply_rebalance


def _pivot_close(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"])
    pivot = df.pivot_table(
        index="date", columns="asset_id", values="close", aggfunc="last"
    ).sort_index()
    return pivot


def run_paper_portfolio(
    portfolios: Sequence[Portfolio],
    prices: pd.DataFrame,
    starting_cash: float = 1_000_000.0,
) -> PortfolioState:
    """Step a paper portfolio forward over the prices' date range.

    Assumptions
    -----------
    - No slippage, no commissions, end-of-day fills at the close.
    - Fractional shares allowed.
    - Rebalance dates are the ``Portfolio.as_of`` dates; any rebalance
      date not present in the price index is skipped.
    - Days before the first rebalance hold 100% cash at ``starting_cash``.
    """
    if not portfolios:
        empty_idx = pd.DatetimeIndex([], name="date")
        return PortfolioState(
            history=pd.DataFrame(columns=["date", "asset_id", "weight", "value"]),
            cash_series=pd.Series(dtype=float, index=empty_idx, name="cash"),
            total_value=pd.Series(dtype=float, index=empty_idx, name="total_value"),
            returns=pd.Series(dtype=float, index=empty_idx, name="portfolio_return"),
        )

    price_pivot = _pivot_close(prices)
    if price_pivot.empty:
        raise ValueError("prices frame has no rows")

    first_rebalance = pd.Timestamp(min(p.as_of for p in portfolios))
    # Trade only from the first rebalance onward — pre-rebalance days are
    # all-cash and add no information.
    price_pivot = price_pivot.loc[price_pivot.index >= first_rebalance]
    if price_pivot.empty:
        raise ValueError(
            "no price dates on or after the first rebalance date"
        )

    schedule = {
        pd.Timestamp(p.as_of): p
        for p in sorted(portfolios, key=lambda x: x.as_of)
    }

    shares: dict[str, float] = {}
    cash: float = float(starting_cash)

    cash_history: dict[pd.Timestamp, float] = {}
    total_history: dict[pd.Timestamp, float] = {}
    history_rows: list[dict] = []

    for ts in price_pivot.index:
        row = price_pivot.loc[ts]
        prices_today = {
            asset: float(price)
            for asset, price in row.items()
            if pd.notna(price) and price > 0
        }

        if ts in schedule:
            target = schedule[ts]
            shares, cash = apply_rebalance(
                prev_shares=shares,
                cash=cash,
                target_weights=target.weights,
                prices_at_date=prices_today,
            )

        position_value = 0.0
        per_asset_value: dict[str, float] = {}
        for asset, s in shares.items():
            price = prices_today.get(asset)
            if price is None:
                continue
            v = s * price
            per_asset_value[asset] = v
            position_value += v

        total = cash + position_value
        cash_history[ts] = cash
        total_history[ts] = total

        for asset, v in per_asset_value.items():
            history_rows.append(
                {
                    "date": ts.date(),
                    "asset_id": asset,
                    "value": v,
                    "_total": total,
                }
            )

    history_df = pd.DataFrame(history_rows)
    if not history_df.empty:
        history_df["weight"] = np.where(
            history_df["_total"] > 0,
            history_df["value"] / history_df["_total"],
            0.0,
        )
        history_df = history_df[["date", "asset_id", "weight", "value"]]

    cash_s = pd.Series(cash_history, name="cash").sort_index()
    total_s = pd.Series(total_history, name="total_value").sort_index()
    # Explicit shift-based pct_change avoids the pandas FutureWarning around
    # default ``fill_method`` behavior.
    prev = total_s.shift(1)
    returns = (total_s / prev - 1.0).fillna(0.0).rename("portfolio_return")

    return PortfolioState(
        history=history_df,
        cash_series=cash_s,
        total_value=total_s,
        returns=returns,
    )
