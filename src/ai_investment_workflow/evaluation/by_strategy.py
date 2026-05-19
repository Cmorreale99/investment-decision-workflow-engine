"""Strategy attribution: which strategy drove each decision, on aggregate.

Attribution rule
----------------
Each ``PerformanceRecord`` is attributed to the strategy with the highest
score for that ``(date, asset_id)`` at decision time. When no
``signals_history`` is supplied, all decisions are bucketed under
``"composite"`` so the table is still emitted.

``PENDING`` outcomes are excluded from the aggregation: their excess
returns are zero placeholders, not real measurements.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from ..schemas import OutcomeLabel, PerformanceRecord
from ..strategies import load_enabled_strategies

_COLUMNS = [
    "strategy",
    "n_decisions",
    "hit_rate",
    "mean_excess_return",
    "median_excess_return",
]


def build_signals_history(features: pd.DataFrame) -> pd.DataFrame:
    """Long-format ``(date, asset_id, strategy, score)`` across all dates."""
    if features.empty:
        return pd.DataFrame(columns=["date", "asset_id", "strategy", "score"])
    frames: list[pd.DataFrame] = []
    for s in load_enabled_strategies():
        df = s.score(features).copy()
        df["strategy"] = s.name
        frames.append(df[["date", "asset_id", "strategy", "score"]])
    if not frames:
        return pd.DataFrame(columns=["date", "asset_id", "strategy", "score"])
    out = pd.concat(frames, ignore_index=True)
    return out.dropna(subset=["score"]).reset_index(drop=True)


def _attribution_map(signals_history: pd.DataFrame) -> dict[tuple, str]:
    """``{(date, asset_id) → strategy}`` using max score at that point."""
    if signals_history.empty:
        return {}
    sigs = signals_history.copy()
    sigs["date"] = pd.to_datetime(sigs["date"]).dt.date
    idx = sigs.groupby(["date", "asset_id"])["score"].idxmax()
    selected = sigs.loc[idx]
    out: dict[tuple, str] = {}
    for row in selected.itertuples(index=False):
        out[(row.date, row.asset_id)] = str(row.strategy)
    return out


def performance_by_strategy(
    rankings_history: pd.DataFrame,
    performance_records: Sequence[PerformanceRecord],
    signals_history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Aggregate per-decision excess returns by attributed strategy."""
    del rankings_history  # accepted for forward-compat; current rule uses signals.

    if not performance_records:
        return pd.DataFrame(columns=_COLUMNS)

    attribution = (
        _attribution_map(signals_history)
        if signals_history is not None
        else {}
    )

    rows: list[dict] = []
    for r in performance_records:
        if r.outcome_label == OutcomeLabel.PENDING:
            continue
        strategy = attribution.get(
            (r.evaluation_start, r.asset_id), "composite"
        )
        rows.append(
            {"strategy": strategy, "excess_return": float(r.excess_return)}
        )

    if not rows:
        return pd.DataFrame(columns=_COLUMNS)

    df = pd.DataFrame(rows)
    agg = (
        df.groupby("strategy")
        .agg(
            n_decisions=("excess_return", "count"),
            hit_rate=("excess_return", lambda s: float((s > 0).mean())),
            mean_excess_return=("excess_return", "mean"),
            median_excess_return=("excess_return", "median"),
        )
        .reset_index()
    )
    return agg.sort_values("strategy").reset_index(drop=True)
