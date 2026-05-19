"""Phase 5 entry point: paper-portfolio simulation + benchmark.

Reads ``data/processed/features.parquet`` (Phase 3) and
``data/processed/prices.parquet`` (Phase 2). Phase 4's ``rankings.parquet``
only carries the most recent ``as_of``, so this script computes a full
**historical** rankings frame inline by calling the Phase 4 scoring
primitives over every available date — see
``simulation.ranking_history.build_rankings_history``.

Writes:
- ``data/processed/portfolio_history.parquet``
- ``data/processed/portfolio_returns.parquet``
- ``data/processed/portfolio_summary.parquet``

No LLM, no network, no execution. Deterministic for deterministic input.
"""

from __future__ import annotations

import sys

import pandas as pd

from ai_investment_workflow.simulation import (
    build_rankings_history,
    run_simulation,
    state_to_returns_frame,
)
from ai_investment_workflow.utils import (
    get_logger,
    load_settings,
    processed_dir,
    setup_logging,
)


def main() -> int:
    setup_logging()
    log = get_logger(__name__)

    features_path = processed_dir() / "features.parquet"
    prices_path = processed_dir() / "prices.parquet"
    if not features_path.is_file():
        log.error(
            "features parquet not found: %s (run scripts/build_features.py first)",
            features_path,
        )
        return 1
    if not prices_path.is_file():
        log.error(
            "prices parquet not found: %s (run scripts/run_ingestion.py first)",
            prices_path,
        )
        return 1

    settings = load_settings()
    prices = pd.read_parquet(prices_path)
    prices["date"] = pd.to_datetime(prices["date"]).dt.date
    features = pd.read_parquet(features_path)
    features["date"] = pd.to_datetime(features["date"]).dt.date

    log.info(
        "Computing historical rankings across %d dates from %d feature rows",
        features["date"].nunique(),
        len(features),
    )
    rankings_history = build_rankings_history(features)
    if rankings_history.empty:
        log.warning(
            "no rankings produced — check that enabled strategies have valid features"
        )
        return 0

    result = run_simulation(rankings_history, prices, settings)
    state = result["portfolio_state"]

    history_out = processed_dir() / "portfolio_history.parquet"
    returns_out = processed_dir() / "portfolio_returns.parquet"
    summary_out = processed_dir() / "portfolio_summary.parquet"

    state.history.to_parquet(history_out, index=False)
    state_to_returns_frame(
        state, result["benchmark_returns"], result["excess_returns"]
    ).to_parquet(returns_out, index=False)
    pd.DataFrame([result["metrics"]]).to_parquet(summary_out, index=False)

    log.info("Wrote %s, %s, %s", history_out, returns_out, summary_out)
    metrics = result["metrics"]
    log.info(
        "Total return: %.4f | Benchmark: %.4f | Excess: %.4f | Max DD: %.4f | Vol: %.4f | Hit rate: %.4f",
        metrics["total_return"],
        metrics["benchmark_total_return"],
        metrics["excess_total_return"],
        metrics["max_drawdown"],
        metrics["volatility"],
        metrics["hit_rate"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
