"""The dashboard's sole write path: recording a human decision.

Action buttons do not touch the decision log directly — they build a
``HumanDecision`` and route it through the Phase 9 human review layer
(``apply_decision`` + ``append_decisions``), inheriting its guarantees:
the decision is schema-validated, the log is append-only and deduped, and
nothing else (raw data, recommendations, packets, diagnostics) is mutated.
Recording a decision is an explicit human action — it never auto-approves
and never executes a trade.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ..human_review import (
    DECISION_LOG_FILENAME,
    apply_decision,
    append_decisions,
)
from ..schemas import DecisionRecord, HumanAction, HumanDecision, Recommendation
from ..utils.paths import processed_dir


def record_decision(
    recommendation: Recommendation,
    human_action: HumanAction,
    *,
    override: bool = False,
    notes: str | None = None,
    reviewed_at: datetime | None = None,
    log_path: str | Path | None = None,
) -> DecisionRecord:
    """Record one human action against a recommendation and persist it.

    Returns the resulting ``DecisionRecord``. Appending is idempotent —
    re-recording the same recommendation replaces its prior record rather
    than duplicating it.
    """
    decision = HumanDecision(
        asset_id=recommendation.asset_id,
        recommendation_id=recommendation.recommendation_id,
        human_action=human_action,
        override=override,
        human_notes=notes,
        reviewed_at=reviewed_at or datetime.now(),
    )
    record = apply_decision(recommendation, decision)
    target = Path(log_path) if log_path else processed_dir() / DECISION_LOG_FILENAME
    append_decisions([record], target)
    return record
