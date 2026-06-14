"""Human review layer: decision seam and recommendation→record merge.

Phase 9. Every Phase 8 ``Recommendation`` must pass through a human
decision before it becomes a final ``DecisionRecord`` — the system action
is never auto-applied. That discipline is enforced here:

- a ``DecisionSource`` supplies a ``HumanDecision`` for a recommendation
  (or ``None`` to leave it pending — nothing is recorded without an
  explicit human action);
- ``apply_decision`` merges the system recommendation with the human
  decision into one schema-validated ``DecisionRecord``, after checking
  the two refer to the same recommendation/asset (audit integrity);
- ``final_status`` is derived from the human action.

This layer reads recommendations and writes decision records; it never
mutates raw data, recommendations, or any upstream artifact, and it never
executes a trade.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..schemas import (
    DecisionRecord,
    HumanAction,
    HumanDecision,
    Recommendation,
)

#: Final status recorded for each human action.
FINAL_STATUS: dict[HumanAction, str] = {
    HumanAction.APPROVE: "approved",
    HumanAction.REJECT: "rejected",
    HumanAction.WATCHLIST: "watchlisted",
    HumanAction.OVERRIDE: "overridden",
    HumanAction.NEEDS_REVIEW: "needs_review",
}


@runtime_checkable
class DecisionSource(Protocol):
    """Supplies a human decision for a recommendation, or ``None``.

    Returning ``None`` means "not yet reviewed" — the recommendation stays
    pending and no record is written. Implementations must not mutate the
    recommendation or execute trades.
    """

    name: str

    def decide(self, recommendation: Recommendation) -> HumanDecision | None:
        ...


def final_status_for(action: HumanAction) -> str:
    """Derive the persisted ``final_status`` from a human action."""
    try:
        return FINAL_STATUS[action]
    except KeyError:  # pragma: no cover - enum is exhaustive
        raise ValueError(f"unknown human action: {action!r}") from None


def decision_id_for(recommendation: Recommendation) -> str:
    """``dec_{YYYY_MM_DD}_{asset_id}`` — mirrors the recommendation id."""
    return f"dec_{recommendation.timestamp:%Y_%m_%d}_{recommendation.asset_id}"


def apply_decision(
    recommendation: Recommendation, human_decision: HumanDecision
) -> DecisionRecord:
    """Merge a recommendation and its human decision into a DecisionRecord.

    Raises ``ValueError`` if the decision does not refer to the same
    recommendation and asset — a decision is never applied to the wrong
    candidate. Inputs are read-only (the schema objects are frozen).
    """
    if human_decision.recommendation_id != recommendation.recommendation_id:
        raise ValueError(
            f"decision targets recommendation "
            f"{human_decision.recommendation_id!r}, expected "
            f"{recommendation.recommendation_id!r}"
        )
    if human_decision.asset_id != recommendation.asset_id:
        raise ValueError(
            f"decision is for asset {human_decision.asset_id!r}, "
            f"expected {recommendation.asset_id!r}"
        )

    return DecisionRecord(
        decision_id=decision_id_for(recommendation),
        asset_id=recommendation.asset_id,
        timestamp=recommendation.timestamp,
        recommendation_id=recommendation.recommendation_id,
        system_action=recommendation.action,
        system_conviction=recommendation.conviction,
        human_action=human_decision.human_action,
        override=human_decision.override,
        human_notes=human_decision.human_notes,
        final_status=final_status_for(human_decision.human_action),
    )
