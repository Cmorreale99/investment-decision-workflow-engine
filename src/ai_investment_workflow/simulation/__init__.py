"""Simulation layer: paper portfolio + benchmark + metrics.

Phase 5. Pure pandas/numpy. No LLM, no network, no execution. Does not
produce buy/sell recommendations; only deterministic simulated histories.
"""

from .base import (
    Portfolio,
    PortfolioState,
    apply_rebalance,
    clip_position_weights,
    clip_sector_weights,
    equal_weight_topn,
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
from .pipeline import run_simulation, state_to_returns_frame
from .ranking_history import build_rankings_history
from .rebalance import rebalance_weekly

__all__ = [
    "Portfolio",
    "PortfolioState",
    "annualized_return",
    "apply_rebalance",
    "benchmark_returns",
    "build_rankings_history",
    "clip_position_weights",
    "clip_sector_weights",
    "equal_weight_topn",
    "hit_rate",
    "max_drawdown",
    "rebalance_weekly",
    "run_paper_portfolio",
    "run_simulation",
    "state_to_returns_frame",
    "total_return",
    "volatility",
]
