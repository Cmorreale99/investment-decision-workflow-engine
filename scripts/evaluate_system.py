"""Phase 6 entry point: evaluation diagnostics.

Pipeline
--------
1. Read ``data/processed/{prices,features}.parquet`` (Phases 2 & 3).
2. Use ``rankings_history.parquet`` if present; otherwise recompute via
   ``simulation.build_rankings_history``.
3. Run the Phase 5 simulation inline so we have a fresh ``PortfolioState``
   for portfolio-level diagnostics.
4. Run ``evaluation.run_evaluation`` and persist:
     - ``data/processed/performance_records.parquet``
     - ``data/processed/portfolio_diagnostics.parquet``
     - ``data/processed/performance_by_strategy.parquet``

No LLM, no network, no recommendations.
"""

from __future__ import annotations

import sys

import pandas as pd

from ai_investment_workflow.evaluation import (
    performance_records_to_frame,
    run_evaluation,
)
from ai_investment_workflow.simulation import (
    build_rankings_history,
    run_simulation,
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

    rh_path = processed_dir() / "rankings_history.parquet"
    if rh_path.is_file():
        rankings_history = pd.read_parquet(rh_path)
        rankings_history["date"] = pd.to_datetime(rankings_history["date"]).dt.date
        log.info("loaded rankings_history.parquet (%d rows)", len(rankings_history))
    else:
        log.info("rankings_history.parquet absent; recomputing inline")
        rankings_history = build_rankings_history(features)

    if rankings_history.empty:
        log.warning("no rankings produced; skipping evaluation")
        return 0

    log.info("re-running Phase 5 simulation to obtain a fresh PortfolioState")
    simulation_result = run_simulation(rankings_history, prices, settings)

    log.info("running Phase 6 evaluation")
    result = run_evaluation(
        simulation_result=simulation_result,
        rankings_history=rankings_history,
        prices=prices,
        settings=settings,
        features=features,
    )

    records = result["performance_records"]
    records_df = performance_records_to_frame(records)
    records_out = processed_dir() / "performance_records.parquet"
    records_df.to_parquet(records_out, index=False)

    diagnostics_out = processed_dir() / "portfolio_diagnostics.parquet"
    pd.DataFrame([result["portfolio_diagnostics"]]).to_parquet(
        diagnostics_out, index=False
    )

    by_strategy_out = processed_dir() / "performance_by_strategy.parquet"
    result["by_strategy_table"].to_parquet(by_strategy_out, index=False)

    diag = result["portfolio_diagnostics"]
    log.info(
        "Records: %d | Total return: %.4f | Excess: %.4f | Hit rate: %.4f | Max DD: %.4f",
        len(records),
        float(diag.get("total_return", 0.0)),
        float(diag.get("excess_total_return", 0.0)),
        float(diag.get("hit_rate", 0.0)),
        float(diag.get("max_drawdown", 0.0)),
    )
    log.info("Wrote %s, %s, %s", records_out, diagnostics_out, by_strategy_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
