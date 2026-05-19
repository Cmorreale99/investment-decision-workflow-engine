"""Portfolio data structures and pure rebalance/clip helpers.

Conventions
-----------
- ``Portfolio.weights`` are **fractions of NAV** at the rebalance date,
  one per held asset. They are expected to sum to ≤ 1; the residual is
  cash and is captured in ``Portfolio.cash`` for documentation.
- ``PortfolioState`` is the **realized** time series produced by the
  engine. Its ``history`` frame carries dollar values, and weights in
  that frame are computed from dollar value / NAV at each date.

Phase 5 assumptions (deterministic baseline):
- No slippage, no commissions, no taxes.
- End-of-day fills at the trading day's close.
- Equal-weight top-N unless overridden.
- Risk clipping is applied at rebalance time; the residual goes to cash.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Mapping

import pandas as pd


@dataclass(frozen=True)
class Portfolio:
    """Target portfolio at a single rebalance date."""

    as_of: date
    weights: dict[str, float] = field(default_factory=dict)
    cash: float = 0.0


@dataclass(frozen=True)
class PortfolioState:
    """Daily history produced by ``engine.run_paper_portfolio``."""

    history: pd.DataFrame  # date, asset_id, weight, value
    cash_series: pd.Series  # date → cash dollars
    total_value: pd.Series  # date → total NAV in dollars
    returns: pd.Series  # date → daily portfolio return


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def equal_weight_topn(
    rankings_at_date: pd.DataFrame,
    top_n: int,
    exclude_conflicts: bool = False,
) -> dict[str, float]:
    """Return ``{asset_id: 1/N}`` for the top ``top_n`` by rank.

    Ties on ``rank`` are broken by ``asset_id`` ascending so the output is
    deterministic. When ``exclude_conflicts=True``, rows with
    ``strategy_conflict == True`` are dropped before selection.
    """
    if top_n <= 0 or rankings_at_date.empty:
        return {}
    df = rankings_at_date
    if exclude_conflicts and "strategy_conflict" in df.columns:
        df = df.loc[~df["strategy_conflict"].astype(bool)]
    if df.empty:
        return {}
    df = df.sort_values(["rank", "asset_id"]).head(top_n)
    weight = 1.0 / len(df)
    return {row.asset_id: weight for row in df.itertuples(index=False)}


def clip_position_weights(
    weights: Mapping[str, float],
    max_weight: float | None,
) -> dict[str, float]:
    """Cap each weight at ``max_weight``. Residual is left to cash.

    No redistribution: by design, a binding cap reduces gross exposure
    rather than concentrating capital in already-uncapped names.
    """
    if max_weight is None or max_weight >= 1.0 or not weights:
        return dict(weights)
    return {k: min(float(v), float(max_weight)) for k, v in weights.items()}


def clip_sector_weights(
    weights: Mapping[str, float],
    sectors: Mapping[str, str] | None,
    max_sector_weight: float | None,
) -> dict[str, float]:
    """Scale positions in over-cap sectors down to the cap proportionally.

    No-op when ``sectors`` is missing or the cap is ≥ 1.0. Assets with no
    sector mapping are treated as sector ``"Unknown"``.
    """
    if not sectors or max_sector_weight is None or max_sector_weight >= 1.0:
        return dict(weights)
    out = {k: float(v) for k, v in weights.items()}
    totals: dict[str, float] = {}
    for asset, w in out.items():
        sec = sectors.get(asset, "Unknown")
        totals[sec] = totals.get(sec, 0.0) + w
    for sec, total in totals.items():
        if total > max_sector_weight:
            scale = max_sector_weight / total
            for asset in list(out.keys()):
                if sectors.get(asset, "Unknown") == sec:
                    out[asset] *= scale
    return out


def apply_rebalance(
    prev_shares: Mapping[str, float],
    cash: float,
    target_weights: Mapping[str, float],
    prices_at_date: Mapping[str, float],
) -> tuple[dict[str, float], float]:
    """Compute new shares + new cash after rebalancing to target weights.

    Pure function. Skips assets with missing or non-positive prices.
    Assumes no slippage, no commissions, end-of-day fills.
    """
    nav = float(cash)
    for asset, shares in prev_shares.items():
        price = prices_at_date.get(asset)
        if price is None or price <= 0:
            continue
        nav += float(shares) * float(price)

    new_shares: dict[str, float] = {}
    for asset, weight in target_weights.items():
        if weight <= 0:
            continue
        price = prices_at_date.get(asset)
        if price is None or price <= 0:
            continue
        dollars = nav * float(weight)
        new_shares[asset] = dollars / float(price)

    invested = sum(s * prices_at_date[a] for a, s in new_shares.items())
    return new_shares, nav - invested
