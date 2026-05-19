"""Evaluation layer: outcome diagnostics and benchmark-relative reporting.

Phase 6. Final layer of the deterministic baseline. Pure pandas/numpy.
No LLM, no agents, no human review, no UI, no execution.
"""

from .asset_eval import evaluate_asset_decisions
from .by_strategy import build_signals_history, performance_by_strategy
from .pipeline import performance_records_to_frame, run_evaluation
from .portfolio_eval import evaluate_portfolio
from .windows import (
    calendar_to_trading_days,
    forward_window_max_drawdown,
    forward_window_return,
    outcome_label,
)

__all__ = [
    "build_signals_history",
    "calendar_to_trading_days",
    "evaluate_asset_decisions",
    "evaluate_portfolio",
    "forward_window_max_drawdown",
    "forward_window_return",
    "outcome_label",
    "performance_by_strategy",
    "performance_records_to_frame",
    "run_evaluation",
]
