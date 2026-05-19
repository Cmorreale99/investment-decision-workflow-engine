"""End-to-end simulation orchestration."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import pandas as pd

from ..utils.config import load_risk_rules as _load_risk_rules_yaml
from .base import (
    Portfolio,
    PortfolioState,
    clip_position_weights,
    clip_sector_weights,
)
from .benchmark import benchmark_returns
from .engine import run_paper_portfolio
from .metrics import (
    annualized_return,
    hit_rate,
    max_drawdown,
    total_return,
    volatility,
)
from .rebalance import rebalance_weekly


def _apply_risk_clipping(
    portfolios: list[Portfolio],
    sectors: Mapping[str, str] | None,
    max_single: float | None,
    max_sector: float | None,
) -> list[Portfolio]:
    out: list[Portfolio] = []
    for p in portfolios:
        w = clip_position_weights(p.weights, max_single)
        w = clip_sector_weights(w, sectors, max_sector)
        cash = max(0.0, 1.0 - sum(w.values()))
        out.append(replace(p, weights=w, cash=cash))
    return out


def run_simulation(
    rankings_history: pd.DataFrame,
    prices: pd.DataFrame,
    settings: Mapping[str, Any],
    *,
    sectors: Mapping[str, str] | None = None,
    exclude_conflicts: bool = False,
    risk_clipping: bool = True,
) -> dict[str, Any]:
    """Return ``{portfolio_state, benchmark_returns, excess_returns, metrics}``.

    Parameters
    ----------
    rankings_history
        Long format with ``date, asset_id, composite_score, rank,
        strategy_conflict``. See ``ranking_history.build_rankings_history``.
    prices
        Phase 2 normalized price frame.
    settings
        Parsed ``config/settings.yaml`` (``top_n_candidates``, ``benchmark``,
        optional ``starting_cash``, ``rebalance_frequency``).
    sectors
        Optional ``{asset_id: sector_name}`` map enabling sector clipping.
    exclude_conflicts
        Drop conflict-flagged rows before forming each week's top-N.
    risk_clipping
        When True (default), apply ``max_single_position_weight`` and
        ``max_sector_weight`` clipping from ``config/risk_rules.yaml``.
    """
    top_n = int(settings.get("top_n_candidates", 10))
    benchmark = str(settings.get("benchmark", "SPY"))
    starting_cash = float(settings.get("starting_cash", 1_000_000.0))
    rebalance_day = _rebalance_period_alias(
        settings.get("rebalance_frequency", "weekly")
    )

    portfolios = rebalance_weekly(
        rankings_history,
        top_n=top_n,
        rebalance_day=rebalance_day,
        exclude_conflicts=exclude_conflicts,
    )

    if risk_clipping:
        risk_cfg = _load_risk_rules_yaml()
        portfolios = _apply_risk_clipping(
            portfolios,
            sectors=sectors,
            max_single=float(risk_cfg.get("max_single_position_weight", 1.0)),
            max_sector=float(risk_cfg.get("max_sector_weight", 1.0)),
        )

    state = run_paper_portfolio(portfolios, prices, starting_cash=starting_cash)
    bench = benchmark_returns(prices, benchmark=benchmark)
    bench_aligned = bench.reindex(state.returns.index).fillna(0.0)
    excess = (state.returns - bench_aligned).rename("excess_return")

    metrics = {
        "total_return": total_return(state.returns),
        "annualized_return": annualized_return(state.returns),
        "volatility": volatility(state.returns),
        "max_drawdown": max_drawdown(state.returns),
        "hit_rate": hit_rate(state.returns),
        "benchmark_total_return": total_return(bench_aligned),
        "benchmark_annualized_return": annualized_return(bench_aligned),
        "benchmark_volatility": volatility(bench_aligned),
        "benchmark_max_drawdown": max_drawdown(bench_aligned),
        "excess_total_return": (
            total_return(state.returns) - total_return(bench_aligned)
        ),
        "n_rebalances": len(portfolios),
        "n_trading_days": int(len(state.returns)),
    }

    return {
        "portfolio_state": state,
        "benchmark_returns": bench_aligned,
        "excess_returns": excess,
        "metrics": metrics,
    }


def _rebalance_period_alias(frequency: str) -> str:
    """Map settings ``rebalance_frequency`` to a pandas period alias.

    Only ``weekly`` is supported in Phase 5; anything else falls back to
    ``W-FRI`` with a soft assumption (documented).
    """
    if frequency == "weekly":
        return "W-FRI"
    return "W-FRI"


def state_to_returns_frame(
    state: PortfolioState,
    benchmark: pd.Series,
    excess: pd.Series,
) -> pd.DataFrame:
    """Format the engine output as ``data/processed/portfolio_returns.parquet``."""
    idx = state.returns.index
    df = pd.DataFrame(
        {
            "date": [ts.date() if hasattr(ts, "date") else ts for ts in idx],
            "portfolio_return": state.returns.to_numpy(),
            "benchmark_return": benchmark.reindex(idx).fillna(0.0).to_numpy(),
            "excess_return": excess.reindex(idx).fillna(0.0).to_numpy(),
        }
    )
    return df
