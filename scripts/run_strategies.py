"""Phase 4 entry point: score, rank, and risk-flag the universe.

Reads ``data/processed/features.parquet`` (and ``prices.parquet`` for
the per-asset ``CompanySnapshot`` used by risk evaluation), runs the
strategy pipeline, and writes:

- ``data/processed/signals.parquet`` (long format)
- ``data/processed/rankings.parquet``
- ``data/processed/risk_flags.parquet`` (long format)

Top-N candidates per ``settings.top_n_candidates`` are logged.
"""

from __future__ import annotations

import sys

import pandas as pd

from ai_investment_workflow.features import build_company_snapshots
from ai_investment_workflow.risk import RiskConfig, evaluate_risk
from ai_investment_workflow.strategies import build_strategy_signals
from ai_investment_workflow.utils import (
    get_logger,
    load_settings,
    processed_dir,
    setup_logging,
)


def _row_to_feature_dict(row, feature_cols: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for col in feature_cols:
        value = getattr(row, col)
        if pd.notna(value):
            out[col] = float(value)
    return out


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
    top_n = int(settings.get("top_n_candidates", 10))

    prices = pd.read_parquet(prices_path)
    prices["date"] = pd.to_datetime(prices["date"]).dt.date
    features = pd.read_parquet(features_path)
    features["date"] = pd.to_datetime(features["date"]).dt.date

    as_of = features["date"].max()
    log.info("Building strategy signals for %s (top_n=%d)", as_of, top_n)

    signals = build_strategy_signals(features, as_of=as_of)
    if not signals:
        log.warning(
            "no strategy signals produced; check enabled strategies and feature coverage"
        )
        return 0

    # --- signals.parquet (long format) -----------------------------------
    signal_rows: list[dict] = []
    for asset_id, sig in signals.items():
        for strat_name, score in sig.strategy_scores.items():
            signal_rows.append(
                {
                    "date": as_of,
                    "asset_id": asset_id,
                    "strategy": strat_name,
                    "score": float(score),
                }
            )
    signals_df = pd.DataFrame(signal_rows)
    signals_out = processed_dir() / "signals.parquet"
    signals_df.to_parquet(signals_out, index=False)
    log.info("Wrote %d strategy score rows to %s", len(signals_df), signals_out)

    # --- rankings.parquet -------------------------------------------------
    ranking_rows = [
        {
            "date": as_of,
            "asset_id": sig.asset_id,
            "composite_score": float(sig.composite_score),
            "rank": int(sig.rank),
            "strategy_conflict": bool(sig.strategy_conflict),
        }
        for sig in signals.values()
    ]
    rankings_df = (
        pd.DataFrame(ranking_rows).sort_values("rank").reset_index(drop=True)
    )
    rankings_out = processed_dir() / "rankings.parquet"
    rankings_df.to_parquet(rankings_out, index=False)
    log.info("Wrote rankings to %s", rankings_out)

    # --- risk_flags.parquet ----------------------------------------------
    snapshots = build_company_snapshots(prices, as_of=as_of)
    risk_cfg = RiskConfig.from_yaml()
    feature_cols = [c for c in features.columns if c not in ("date", "asset_id")]
    feature_slice = features.loc[features["date"] == as_of]

    risk_rows: list[dict] = []
    for row in feature_slice.itertuples(index=False):
        asset_id = row.asset_id
        snap = snapshots.get(asset_id)
        if snap is None:
            continue
        feats = _row_to_feature_dict(row, feature_cols)
        flags = evaluate_risk(snap, feats, config=risk_cfg)
        for flag_name, value in flags.items():
            risk_rows.append(
                {
                    "date": as_of,
                    "asset_id": asset_id,
                    "risk_flag": flag_name,
                    "value": bool(value),
                }
            )

    risk_out = processed_dir() / "risk_flags.parquet"
    if risk_rows:
        pd.DataFrame(risk_rows).to_parquet(risk_out, index=False)
        log.info("Wrote %d risk flag rows to %s", len(risk_rows), risk_out)
    else:
        log.warning("no risk flags produced; skipping risk_flags.parquet")

    # --- top-N log -------------------------------------------------------
    head = rankings_df.head(top_n)
    log.info("Top %d candidates by composite score:", top_n)
    for row in head.itertuples(index=False):
        log.info(
            "  rank=%d %s composite=%.3f conflict=%s",
            row.rank,
            row.asset_id,
            row.composite_score,
            row.strategy_conflict,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
