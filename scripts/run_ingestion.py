"""Phase 2 entry point: pull and normalize market data.

Reads provider, benchmark, and lookback from ``config/settings.yaml``.
Universe comes from ``config/universe.yaml``. Writes per-ticker parquet
to ``data/raw/`` and the merged frame to ``data/processed/prices.parquet``.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta

from ai_investment_workflow.ingestion import get_provider, run_ingestion
from ai_investment_workflow.utils import (
    get_logger,
    load_settings,
    load_universe,
    processed_dir,
    raw_dir,
    setup_logging,
)


def main() -> int:
    setup_logging()
    log = get_logger(__name__)

    settings = load_settings()
    universe = load_universe()
    provider_name = settings.get("data_provider", "fixture")
    benchmark = settings.get("benchmark", "SPY")
    lookback_days = int(settings.get("lookback_days", 365))

    end = date.today()
    start = end - timedelta(days=lookback_days)

    log.info(
        "Ingesting %d tickers + benchmark %s via %s provider (%s → %s)",
        len(universe),
        benchmark,
        provider_name,
        start,
        end,
    )

    provider = get_provider(provider_name)
    df = run_ingestion(
        provider=provider,
        tickers=universe,
        benchmark=benchmark,
        start=start,
        end=end,
        raw_dir=raw_dir(),
        processed_dir=processed_dir(),
    )
    log.info(
        "Ingested %d rows across %d assets", len(df), df["asset_id"].nunique()
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
