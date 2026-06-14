"""Decision & override diagnostics over the human review layer.

Phase 10. The Phase 6 evaluation diagnoses the deterministic simulation;
this module diagnoses the *human decisions* by joining the Phase 9
decision log (``DecisionRecord``) to the Phase 6 performance records
(``PerformanceRecord``) strictly on ``decision_id``.

It answers the README's first-class evaluation questions: did approved
recommendations outperform rejected ones, did higher-conviction tiers do
better, and did human overrides help or hurt. It generates no
recommendations and mutates nothing — inputs are read-only.

Discipline carried over from the rest of the pipeline:

- a decision contributes to *outcome* metrics only once a matching
  performance record exists; until then it is **pending** and is counted
  in coverage/rate metrics but never given a fabricated outcome;
- the join is strictly on ``decision_id`` — no link is invented;
- results are deterministic: fixed tier order, rounded floats, and the
  JSON form is emitted with sorted keys.

A "hit" is an ``OutcomeLabel.OUTPERFORMED`` (the neutral band already
applied when the performance record was built); excess uses
``excess_return``.
"""

from __future__ import annotations

from typing import Callable, Iterable

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from ..schemas import DecisionRecord, HumanAction, OutcomeLabel, PerformanceRecord

#: Conviction tier boundaries (lower-inclusive), in fixed reporting order.
CONVICTION_TIERS: tuple[str, ...] = ("low", "medium", "high")


def conviction_tier(value: float) -> str:
    """Bucket a system conviction in [0, 1] into low/medium/high.

    ``low`` = [0, 0.34), ``medium`` = [0.34, 0.67), ``high`` = [0.67, 1].
    """
    if value < 0.34:
        return "low"
    if value < 0.67:
        return "medium"
    return "high"


class OutcomeStats(BaseModel):
    """Outcome summary for a group of decisions that have performance records."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    count: int = Field(..., ge=0)
    hit_rate: float | None = None
    mean_excess_return: float | None = None


class DecisionDiagnostics(BaseModel):
    """Schema-validated diagnostics over the human decision log."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total_decisions: int = Field(..., ge=0)
    decisions_with_outcomes: int = Field(..., ge=0)
    pending_outcomes: int = Field(..., ge=0)

    action_counts: dict[str, int] = Field(default_factory=dict)
    approval_rate: float = 0.0
    reject_rate: float = 0.0
    watchlist_rate: float = 0.0
    needs_review_rate: float = 0.0
    override_rate: float = 0.0

    overall_outcome: OutcomeStats
    by_action: dict[str, OutcomeStats] = Field(default_factory=dict)
    by_final_status: dict[str, OutcomeStats] = Field(default_factory=dict)
    by_conviction_tier: dict[str, OutcomeStats] = Field(default_factory=dict)

    approved_mean_excess: float | None = None
    rejected_mean_excess: float | None = None
    approved_minus_rejected_excess: float | None = None

    override_impact: dict[str, OutcomeStats] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

#: (excess_return, outcome_label) extracted from a performance record.
_Pair = tuple[float, OutcomeLabel]


def _stats(pairs: list[_Pair]) -> OutcomeStats:
    n = len(pairs)
    if n == 0:
        return OutcomeStats(count=0, hit_rate=None, mean_excess_return=None)
    hits = sum(1 for _, label in pairs if label is OutcomeLabel.OUTPERFORMED)
    mean = sum(excess for excess, _ in pairs) / n
    return OutcomeStats(
        count=n,
        hit_rate=round(hits / n, 4),
        mean_excess_return=round(mean, 6),
    )


def _rate(count: int, total: int) -> float:
    return round(count / total, 4) if total else 0.0


def diagnose_decisions(
    decision_records: Iterable[DecisionRecord],
    performance_records: Iterable[PerformanceRecord],
) -> DecisionDiagnostics:
    """Join decisions to outcomes on ``decision_id`` and summarize.

    Decisions without a matching performance record are pending: they
    count toward totals and rate metrics but are excluded from every
    outcome metric (never fabricated).
    """
    decisions = list(decision_records)
    perf = {record.decision_id: record for record in performance_records}
    total = len(decisions)

    action_counts: dict[str, int] = {}
    for decision in decisions:
        key = decision.human_action.value
        action_counts[key] = action_counts.get(key, 0) + 1
    override_count = sum(1 for d in decisions if d.override)

    matched = [(d, perf[d.decision_id]) for d in decisions if d.decision_id in perf]

    def pairs_where(predicate: Callable[[DecisionRecord], bool]) -> list[_Pair]:
        return [
            (record.excess_return, record.outcome_label)
            for decision, record in matched
            if predicate(decision)
        ]

    overall = _stats([(r.excess_return, r.outcome_label) for _, r in matched])

    by_action = {
        action: _stats(pairs_where(lambda d, a=action: d.human_action.value == a))
        for action in sorted(action_counts)
    }
    by_final_status = {
        status: _stats(pairs_where(lambda d, s=status: d.final_status == s))
        for status in sorted({d.final_status for d in decisions})
    }
    present_tiers = {conviction_tier(d.system_conviction) for d in decisions}
    by_conviction_tier = {
        tier: _stats(pairs_where(lambda d, t=tier: conviction_tier(d.system_conviction) == t))
        for tier in CONVICTION_TIERS
        if tier in present_tiers
    }

    approved = _stats(pairs_where(lambda d: d.human_action is HumanAction.APPROVE))
    rejected = _stats(pairs_where(lambda d: d.human_action is HumanAction.REJECT))
    approved_minus_rejected = (
        round(approved.mean_excess_return - rejected.mean_excess_return, 6)
        if approved.mean_excess_return is not None
        and rejected.mean_excess_return is not None
        else None
    )

    override_impact = {
        "overridden": _stats(pairs_where(lambda d: d.override)),
        "not_overridden": _stats(pairs_where(lambda d: not d.override)),
    }

    return DecisionDiagnostics(
        total_decisions=total,
        decisions_with_outcomes=len(matched),
        pending_outcomes=total - len(matched),
        action_counts=action_counts,
        approval_rate=_rate(action_counts.get(HumanAction.APPROVE.value, 0), total),
        reject_rate=_rate(action_counts.get(HumanAction.REJECT.value, 0), total),
        watchlist_rate=_rate(action_counts.get(HumanAction.WATCHLIST.value, 0), total),
        needs_review_rate=_rate(
            action_counts.get(HumanAction.NEEDS_REVIEW.value, 0), total
        ),
        override_rate=_rate(override_count, total),
        overall_outcome=overall,
        by_action=by_action,
        by_final_status=by_final_status,
        by_conviction_tier=by_conviction_tier,
        approved_mean_excess=approved.mean_excess_return,
        rejected_mean_excess=rejected.mean_excess_return,
        approved_minus_rejected_excess=approved_minus_rejected,
        override_impact=override_impact,
    )


# ---------------------------------------------------------------------------
# I/O helpers (read-only inputs, deterministic output)
# ---------------------------------------------------------------------------


def performance_records_from_frame(frame: pd.DataFrame) -> list[PerformanceRecord]:
    """Reconstruct ``PerformanceRecord`` objects from a parquet-shaped frame.

    Inverse of ``evaluation.performance_records_to_frame``. Dates may arrive
    as pandas timestamps; they are coerced back to ``date``.
    """
    records: list[PerformanceRecord] = []
    for row in frame.to_dict(orient="records"):
        records.append(
            PerformanceRecord(
                decision_id=str(row["decision_id"]),
                asset_id=str(row["asset_id"]),
                evaluation_start=pd.Timestamp(row["evaluation_start"]).date(),
                evaluation_end=pd.Timestamp(row["evaluation_end"]).date(),
                asset_return=float(row["asset_return"]),
                benchmark_return=float(row["benchmark_return"]),
                excess_return=float(row["excess_return"]),
                max_drawdown=float(row["max_drawdown"]),
                outcome_label=OutcomeLabel(row["outcome_label"]),
            )
        )
    return records


def diagnostics_to_json(diagnostics: DecisionDiagnostics) -> str:
    """Deterministic JSON (sorted keys, 2-space indent) for the report file."""
    import json

    return json.dumps(diagnostics.model_dump(mode="json"), sort_keys=True, indent=2)
