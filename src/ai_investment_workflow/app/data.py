"""Read-only loaders for the dashboard's input artifacts.

Every loader reads an artifact a prior phase already wrote under
``data/processed/`` and degrades gracefully to an empty state when the
file is absent — the dashboard shows nothing rather than fabricating
anything. Nothing here writes or mutates an artifact.

This module imports no UI library; it is pure and offline-testable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..human_review import (
    DECISION_LOG_FILENAME,
    RECOMMENDATIONS_FILENAME,
    load_decision_log,
    load_recommendations,
)
from ..rag import PACKETS_FILENAME
from ..schemas import ContextPacket, DecisionRecord, Recommendation

#: Phase 10 / Phase 6 report artifacts under ``data/processed/``.
DECISION_DIAGNOSTICS_FILENAME: str = "decision_diagnostics.json"
PORTFOLIO_DIAGNOSTICS_FILENAME: str = "portfolio_diagnostics.parquet"


def load_context_packets(path: str | Path) -> dict[str, ContextPacket]:
    """Load Phase 7 packets into a ``{asset_id: ContextPacket}`` map."""
    path = Path(path)
    if not path.is_file():
        return {}
    packets: dict[str, ContextPacket] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            packet = ContextPacket.model_validate(json.loads(line))
            packets[packet.asset_id] = packet
    return packets


def load_decision_diagnostics(path: str | Path) -> dict | None:
    """Load the Phase 10 diagnostics report; ``None`` if absent."""
    path = Path(path)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_portfolio_diagnostics(path: str | Path) -> dict | None:
    """Load the single-row Phase 6 portfolio diagnostics; ``None`` if absent."""
    path = Path(path)
    if not path.is_file():
        return None
    import pandas as pd

    frame = pd.read_parquet(path)
    if frame.empty:
        return None
    return frame.iloc[0].to_dict()


def latest_decision_by_recommendation(
    decisions: list[DecisionRecord],
) -> dict[str, DecisionRecord]:
    """Map each ``recommendation_id`` to its most recent decision (last wins)."""
    by_recommendation: dict[str, DecisionRecord] = {}
    for decision in decisions:
        by_recommendation[decision.recommendation_id] = decision
    return by_recommendation


@dataclass(frozen=True)
class DashboardData:
    """All read-only artifacts the dashboard renders, loaded once."""

    recommendations: list[Recommendation] = field(default_factory=list)
    packets: dict[str, ContextPacket] = field(default_factory=dict)
    decisions: list[DecisionRecord] = field(default_factory=list)
    decision_diagnostics: dict | None = None
    portfolio_diagnostics: dict | None = None

    @classmethod
    def load(cls, processed_dir: str | Path) -> "DashboardData":
        base = Path(processed_dir)
        recs_path = base / RECOMMENDATIONS_FILENAME
        recommendations = (
            load_recommendations(recs_path) if recs_path.is_file() else []
        )
        return cls(
            recommendations=recommendations,
            packets=load_context_packets(base / PACKETS_FILENAME),
            decisions=load_decision_log(base / DECISION_LOG_FILENAME),
            decision_diagnostics=load_decision_diagnostics(
                base / DECISION_DIAGNOSTICS_FILENAME
            ),
            portfolio_diagnostics=load_portfolio_diagnostics(
                base / PORTFOLIO_DIAGNOSTICS_FILENAME
            ),
        )
