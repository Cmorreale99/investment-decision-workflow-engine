"""End-to-end evaluation orchestration."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from .asset_eval import evaluate_asset_decisions
from .by_strategy import build_signals_history, performance_by_strategy
from .portfolio_eval import evaluate_portfolio


def run_evaluation(
    simulation_result: Mapping[str, Any],
    rankings_history: pd.DataFrame,
    prices: pd.DataFrame,
    settings: Mapping[str, Any],
    *,
    features: pd.DataFrame | None = None,
    signals_history: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Return ``{performance_records, portfolio_diagnostics, by_strategy_table}``.

    ``signals_history`` and ``features`` together govern strategy attribution:
    pass ``signals_history`` directly, or pass ``features`` and let this
    function recompute it. If both are missing, every record is attributed
    to ``"composite"``.
    """
    evaluation_window_days = int(settings.get("evaluation_window_days", 30))
    benchmark = str(settings.get("benchmark", "SPY"))
    top_n = int(settings.get("top_n_candidates", 10))

    records = evaluate_asset_decisions(
        rankings_history=rankings_history,
        prices=prices,
        evaluation_window_days=evaluation_window_days,
        benchmark=benchmark,
        top_n=top_n,
    )

    state = simulation_result["portfolio_state"]
    bench_returns = simulation_result["benchmark_returns"]
    excess_returns = simulation_result["excess_returns"]

    diagnostics = evaluate_portfolio(state, bench_returns, excess_returns)
    diagnostics.update(simulation_result.get("metrics", {}))

    if signals_history is None and features is not None:
        signals_history = build_signals_history(features)

    by_strategy = performance_by_strategy(
        rankings_history=rankings_history,
        performance_records=records,
        signals_history=signals_history,
    )

    return {
        "performance_records": records,
        "portfolio_diagnostics": diagnostics,
        "by_strategy_table": by_strategy,
    }


def performance_records_to_frame(records) -> pd.DataFrame:
    """Long-format frame mirroring ``PerformanceRecord`` fields for parquet."""
    if not records:
        return pd.DataFrame(
            columns=[
                "decision_id",
                "asset_id",
                "evaluation_start",
                "evaluation_end",
                "asset_return",
                "benchmark_return",
                "excess_return",
                "max_drawdown",
                "outcome_label",
            ]
        )
    rows: list[dict] = []
    for r in records:
        rows.append(
            {
                "decision_id": r.decision_id,
                "asset_id": r.asset_id,
                "evaluation_start": r.evaluation_start,
                "evaluation_end": r.evaluation_end,
                "asset_return": float(r.asset_return),
                "benchmark_return": float(r.benchmark_return),
                "excess_return": float(r.excess_return),
                "max_drawdown": float(r.max_drawdown),
                "outcome_label": r.outcome_label.value,
            }
        )
    return pd.DataFrame(rows)
