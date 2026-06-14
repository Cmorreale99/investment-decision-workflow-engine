"""Human review layer: approve / reject / watchlist / override actions.

Phase 9. Every Phase 8 recommendation must pass through an explicit human
decision before it becomes a final ``DecisionRecord`` — the system action
is never auto-applied, and this layer never mutates raw data or executes
trades. The default ``ScriptedDecisionSource`` is deterministic and fully
offline; an optional ``InteractiveDecisionSource`` prompts on stdin (see
:func:`build_decision_source`).
"""

from __future__ import annotations

from .base import (
    FINAL_STATUS,
    DecisionSource,
    apply_decision,
    decision_id_for,
    final_status_for,
)
from .decision_log import (
    DECISION_LOG_FILENAME,
    RECOMMENDATIONS_FILENAME,
    append_decisions,
    decided_recommendation_ids,
    decision_log_to_jsonl,
    load_decision_log,
    load_recommendations,
    write_decision_log,
)
from .pipeline import review_recommendations
from .queue import ReviewQueue
from .sources import (
    HUMAN_DECISIONS_FILENAME,
    InteractiveDecisionSource,
    ScriptedDecisionSource,
    default_decisions_path,
    load_human_decisions,
)

#: Values of ``settings`` / CLI selection that pick the offline scripted source.
_OFFLINE_NAMES: frozenset[str] = frozenset({"", "scripted", "offline", "none"})


def build_decision_source(name: str | None = None, **kwargs) -> DecisionSource:
    """Select a decision source by name.

    ``None`` / ``"scripted"`` / ``"offline"`` (the default) returns the
    deterministic offline ``ScriptedDecisionSource``; ``"interactive"``
    returns the stdin-driven source.
    """
    key = (name or "").strip().lower()
    if key in _OFFLINE_NAMES:
        return ScriptedDecisionSource(**kwargs)
    if key == "interactive":
        return InteractiveDecisionSource()
    raise ValueError(
        f"unknown decision source {name!r} (expected one of: scripted, interactive)"
    )


__all__ = [
    "DECISION_LOG_FILENAME",
    "FINAL_STATUS",
    "HUMAN_DECISIONS_FILENAME",
    "InteractiveDecisionSource",
    "RECOMMENDATIONS_FILENAME",
    "DecisionSource",
    "ReviewQueue",
    "ScriptedDecisionSource",
    "append_decisions",
    "apply_decision",
    "build_decision_source",
    "decided_recommendation_ids",
    "decision_id_for",
    "decision_log_to_jsonl",
    "default_decisions_path",
    "final_status_for",
    "load_decision_log",
    "load_human_decisions",
    "load_recommendations",
    "review_recommendations",
    "write_decision_log",
]
