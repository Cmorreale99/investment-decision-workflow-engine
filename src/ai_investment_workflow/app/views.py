"""Pure presentation view models assembled from loaded artifacts.

Joins the per-candidate recommendation (Phase 8) with its grounded
context packet (Phase 7, for scores/sector/conflict) and any recorded
human decision (Phase 9), and summarizes the evaluation diagnostics
(Phases 6 and 10) into headline numbers. No I/O, no UI, deterministic —
the Streamlit shell only renders what these functions return.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .data import DashboardData, latest_decision_by_recommendation


class CandidateCard(BaseModel):
    """Everything the dashboard shows for one candidate (README review card)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    asset_id: str
    recommendation_id: str
    sector: str | None = None
    composite_score: float | None = None
    rank: int | None = None
    strategy_scores: dict[str, float] = Field(default_factory=dict)
    strategy_conflict: bool | None = None
    action: str = ""
    conviction: float = 0.0
    reasons: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    human_action: str | None = None
    final_status: str | None = None
    human_notes: str | None = None


class EvaluationSummary(BaseModel):
    """Headline metrics pulled from the Phase 6 / Phase 10 diagnostics."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total_decisions: int | None = None
    approval_rate: float | None = None
    override_rate: float | None = None
    decision_hit_rate: float | None = None
    approved_minus_rejected_excess: float | None = None
    portfolio_total_return: float | None = None
    portfolio_excess_return: float | None = None
    portfolio_hit_rate: float | None = None
    portfolio_max_drawdown: float | None = None


def build_candidate_cards(data: DashboardData) -> list[CandidateCard]:
    """One card per recommendation, in recommendation (rank) order.

    Scores/sector/conflict come from the matching context packet when one
    exists; reasons/risks from the recommendation; the human columns from
    the most recent decision on that recommendation. Missing joins degrade
    to ``None``/empty rather than failing.
    """
    by_recommendation = latest_decision_by_recommendation(data.decisions)
    cards: list[CandidateCard] = []
    for rec in data.recommendations:
        packet = data.packets.get(rec.asset_id)
        signal = packet.strategy_signal if packet else None
        decision = by_recommendation.get(rec.recommendation_id)
        cards.append(
            CandidateCard(
                asset_id=rec.asset_id,
                recommendation_id=rec.recommendation_id,
                sector=packet.company_snapshot.sector if packet else None,
                composite_score=signal.composite_score if signal else None,
                rank=signal.rank if signal else None,
                strategy_scores=dict(signal.strategy_scores) if signal else {},
                strategy_conflict=signal.strategy_conflict if signal else None,
                action=rec.action.value,
                conviction=rec.conviction,
                reasons=list(rec.rationale),
                risks=list(rec.risks),
                human_action=decision.human_action.value if decision else None,
                final_status=decision.final_status if decision else None,
                human_notes=decision.human_notes if decision else None,
            )
        )
    return cards


def build_evaluation_summary(data: DashboardData) -> EvaluationSummary:
    """Extract headline metrics; every field is ``None`` when unavailable."""
    diag = data.decision_diagnostics or {}
    overall = diag.get("overall_outcome") or {}
    port = data.portfolio_diagnostics or {}

    def _num(value) -> float | None:
        return float(value) if value is not None else None

    return EvaluationSummary(
        total_decisions=diag.get("total_decisions"),
        approval_rate=_num(diag.get("approval_rate")),
        override_rate=_num(diag.get("override_rate")),
        decision_hit_rate=_num(overall.get("hit_rate")),
        approved_minus_rejected_excess=_num(diag.get("approved_minus_rejected_excess")),
        portfolio_total_return=_num(port.get("total_return")),
        portfolio_excess_return=_num(port.get("excess_total_return")),
        portfolio_hit_rate=_num(port.get("hit_rate")),
        portfolio_max_drawdown=_num(port.get("max_drawdown")),
    )
