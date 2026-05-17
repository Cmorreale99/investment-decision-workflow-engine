"""Phase 3 entry point: compute features from the Phase 2 price frame.

Reads ``data/processed/prices.parquet``, runs every default builder, and
writes the wide-format result to ``data/processed/features.parquet``.
"""

from __future__ import annotations

import sys

import pandas as pd

from ai_investment_workflow.features import compute_feature_frame
from ai_investment_workflow.utils import get_logger, processed_dir, setup_logging


def main() -> int:
    setup_logging()
    log = get_logger(__name__)

    prices_path = processed_dir() / "prices.parquet"
    if not prices_path.is_file():
        log.error("prices parquet not found: %s (run scripts/run_ingestion.py first)", prices_path)
        return 1

    prices = pd.read_parquet(prices_path)
    prices["date"] = pd.to_datetime(prices["date"]).dt.date
    log.info("Loaded %d price rows for %d assets", len(prices), prices["asset_id"].nunique())

    frame = compute_feature_frame(prices)
    out_path = processed_dir() / "features.parquet"
    frame.to_parquet(out_path, index=False)

    feature_cols = [c for c in frame.columns if c not in ("date", "asset_id")]
    log.info(
        "Wrote %d feature rows with %d feature columns to %s",
        len(frame),
        len(feature_cols),
        out_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
