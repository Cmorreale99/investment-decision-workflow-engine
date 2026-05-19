"""Compute a historical rankings frame from a feature time series.

Phase 4's ``run_strategies.py`` persists rankings only at the most recent
``as_of``. For simulation, we need a ranking row per (date, asset_id)
across the full history. This module recomputes that inline using the
Phase 4 strategy pipeline — pure pandas/numpy, deterministic, no LLM.
"""

from __future__ import annotations

import pandas as pd

from ..strategies import (
    composite_score,
    detect_conflicts,
    load_conflict_config,
    load_enabled_strategies,
    load_strategy_weights,
)


def build_rankings_history(features: pd.DataFrame) -> pd.DataFrame:
    """Long-format rankings: ``date, asset_id, composite_score, rank, strategy_conflict``."""
    if features.empty:
        return pd.DataFrame(
            columns=["date", "asset_id", "composite_score", "rank", "strategy_conflict"]
        )

    strategies = load_enabled_strategies()
    weights = load_strategy_weights()
    conflict_cfg = load_conflict_config()

    score_frames = {s.name: s.score(features) for s in strategies}

    rows: list[pd.DataFrame] = []
    dates = sorted(features["date"].unique())
    for d in dates:
        scores: dict[str, dict[str, float]] = {}
        for name, frame in score_frames.items():
            slice_ = frame.loc[frame["date"] == d]
            per_asset: dict[str, float] = {}
            for row in slice_.itertuples(index=False):
                if pd.notna(row.score):
                    per_asset[row.asset_id] = float(row.score)
            scores[name] = per_asset
        if not any(scores.values()):
            continue

        composite = composite_score(scores, weights, as_of=d)
        if composite.empty:
            continue

        ranking = _rank_for_date(composite, as_of=d)
        conflicts = detect_conflicts(
            scores,
            threshold=conflict_cfg["threshold"],
            require_sign_disagreement=conflict_cfg["require_sign_disagreement"],
        )
        ranking["strategy_conflict"] = (
            ranking["asset_id"].map(conflicts).fillna(False).astype(bool)
        )
        rows.append(ranking)

    if not rows:
        return pd.DataFrame(
            columns=["date", "asset_id", "composite_score", "rank", "strategy_conflict"]
        )
    return pd.concat(rows, ignore_index=True)


def _rank_for_date(composite: pd.Series, as_of) -> pd.DataFrame:
    df = composite.rename("composite_score").to_frame()
    df.index.name = "asset_id"
    df = df.reset_index()
    df["date"] = as_of
    df = df.sort_values(
        ["composite_score", "asset_id"], ascending=[False, True]
    ).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df[["date", "asset_id", "composite_score", "rank"]]
