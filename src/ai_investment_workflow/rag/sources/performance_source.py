"""Performance-history source backed by Phase 6 ``PerformanceRecord``s.

Returns the most recent ``max_records`` **completed** records for an
asset (default 5). Point-in-time discipline: a record only counts once
its evaluation window has closed (``evaluation_end <= as_of``) and its
outcome is no longer ``pending`` — outcomes that were unknowable at
``as_of`` are never surfaced.
"""

from __future__ import annotations

from datetime import date
from typing import Sequence

import pandas as pd

from ...schemas import OutcomeLabel, PerformanceRecord

#: Default number of prior records surfaced per asset.
DEFAULT_MAX_RECORDS: int = 5

_COLUMNS = [
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


def _to_frame(
    records: pd.DataFrame | Sequence[PerformanceRecord] | None,
) -> pd.DataFrame:
    """Normalize input to a typed frame mirroring ``PerformanceRecord``."""
    if records is None:
        return pd.DataFrame(columns=_COLUMNS)
    if isinstance(records, pd.DataFrame):
        frame = records.copy()
        if frame.empty:
            return pd.DataFrame(columns=_COLUMNS)
        for col in ("evaluation_start", "evaluation_end"):
            frame[col] = pd.to_datetime(frame[col]).dt.date
        return frame[_COLUMNS]
    rows = [
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
        for r in records
    ]
    return pd.DataFrame(rows, columns=_COLUMNS)


class PerformanceSource:
    """Surfaces completed outcome history as ContextPacket fields.

    - ``prior_decisions``: decision ids, most recent first.
    - ``recent_performance``: float summary of the surfaced records
      (count, mean/last excess return, hit rate, mean max drawdown).
      Empty history yields ``{}`` / ``[]`` — never fabricated values.
    """

    name = "performance_source"

    def __init__(
        self,
        records: pd.DataFrame | Sequence[PerformanceRecord] | None = None,
        max_records: int = DEFAULT_MAX_RECORDS,
    ) -> None:
        if max_records <= 0:
            raise ValueError(f"max_records must be positive, got {max_records}")
        self._records = _to_frame(records)
        self._max_records = max_records

    def fetch(self, asset_id: str, as_of: date) -> dict:
        frame = self._records
        if frame.empty:
            return {"recent_performance": {}, "prior_decisions": []}

        mask = (
            (frame["asset_id"] == asset_id)
            & (frame["outcome_label"] != OutcomeLabel.PENDING.value)
            & (frame["evaluation_end"] <= as_of)
        )
        completed = (
            frame.loc[mask]
            .sort_values(
                ["evaluation_end", "decision_id"], ascending=[False, False]
            )
            .head(self._max_records)
        )
        if completed.empty:
            return {"recent_performance": {}, "prior_decisions": []}

        outperformed = (
            completed["outcome_label"] == OutcomeLabel.OUTPERFORMED.value
        ).sum()
        summary = {
            "n_records": float(len(completed)),
            "mean_excess_return": float(completed["excess_return"].mean()),
            "last_excess_return": float(completed["excess_return"].iloc[0]),
            "hit_rate": float(outperformed / len(completed)),
            "mean_max_drawdown": float(completed["max_drawdown"].mean()),
        }
        return {
            "recent_performance": summary,
            "prior_decisions": completed["decision_id"].tolist(),
        }
