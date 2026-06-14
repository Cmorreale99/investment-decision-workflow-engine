"""Phase 10 decision-diagnostics tests. Offline and deterministic."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from ai_investment_workflow.evaluation import (
    DecisionDiagnostics,
    conviction_tier,
    diagnose_decisions,
    diagnostics_to_json,
    performance_records_from_frame,
    performance_records_to_frame,
)
from ai_investment_workflow.human_review import final_status_for
from ai_investment_workflow.schemas import (
    DecisionRecord,
    HumanAction,
    OutcomeLabel,
    PerformanceRecord,
    SystemAction,
)

TS = date(2026, 4, 1)


def _decision(
    asset: str,
    action: HumanAction,
    *,
    conviction: float = 0.6,
    override: bool = False,
) -> DecisionRecord:
    return DecisionRecord(
        decision_id=f"dec_{TS:%Y_%m_%d}_{asset}",
        asset_id=asset,
        timestamp=TS,
        recommendation_id=f"rec_{TS:%Y_%m_%d}_{asset}",
        system_action=SystemAction.WATCHLIST,
        system_conviction=conviction,
        human_action=action,
        override=override,
        human_notes=None,
        final_status=final_status_for(action),
    )


def _perf(asset: str, excess: float, label: OutcomeLabel) -> PerformanceRecord:
    return PerformanceRecord(
        decision_id=f"dec_{TS:%Y_%m_%d}_{asset}",
        asset_id=asset,
        evaluation_start=TS,
        evaluation_end=TS + timedelta(days=30),
        asset_return=excess + 0.02,
        benchmark_return=0.02,
        excess_return=excess,
        max_drawdown=-0.03,
        outcome_label=label,
    )


# ---------------------------------------------------------------------------
# Conviction tiers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,tier",
    [
        (0.0, "low"),
        (0.33, "low"),
        (0.34, "medium"),
        (0.66, "medium"),
        (0.67, "high"),
        (1.0, "high"),
    ],
)
def test_conviction_tier_boundaries(value, tier) -> None:
    assert conviction_tier(value) == tier


# ---------------------------------------------------------------------------
# Empty / rate-only
# ---------------------------------------------------------------------------


def test_empty_inputs_degrade_gracefully() -> None:
    diag = diagnose_decisions([], [])
    assert isinstance(diag, DecisionDiagnostics)
    assert diag.total_decisions == 0
    assert diag.decisions_with_outcomes == 0
    assert diag.pending_outcomes == 0
    assert diag.overall_outcome.count == 0
    assert diag.overall_outcome.hit_rate is None
    assert diag.approval_rate == 0.0
    assert diag.approved_minus_rejected_excess is None


def test_rate_metrics_without_outcomes() -> None:
    decisions = [
        _decision("AAPL", HumanAction.APPROVE),
        _decision("MSFT", HumanAction.REJECT),
        _decision("NVDA", HumanAction.WATCHLIST, override=True),
        _decision("AMZN", HumanAction.APPROVE, override=True),
    ]
    diag = diagnose_decisions(decisions, [])  # no performance records yet

    assert diag.total_decisions == 4
    assert diag.pending_outcomes == 4
    assert diag.decisions_with_outcomes == 0
    assert diag.action_counts == {"APPROVE": 2, "REJECT": 1, "WATCHLIST": 1}
    assert diag.approval_rate == 0.5
    assert diag.reject_rate == 0.25
    assert diag.watchlist_rate == 0.25
    assert diag.override_rate == 0.5
    # Outcome metrics are empty (pending), never fabricated.
    assert diag.overall_outcome.count == 0
    assert diag.by_action["APPROVE"].count == 0
    assert diag.approved_minus_rejected_excess is None


# ---------------------------------------------------------------------------
# Outcome metrics
# ---------------------------------------------------------------------------


def test_approved_vs_rejected_counterfactual() -> None:
    decisions = [
        _decision("AAPL", HumanAction.APPROVE),
        _decision("MSFT", HumanAction.REJECT),
    ]
    perf = [
        _perf("AAPL", 0.05, OutcomeLabel.OUTPERFORMED),
        _perf("MSFT", -0.03, OutcomeLabel.UNDERPERFORMED),
    ]
    diag = diagnose_decisions(decisions, perf)

    assert diag.decisions_with_outcomes == 2
    assert diag.pending_outcomes == 0
    assert diag.approved_mean_excess == pytest.approx(0.05)
    assert diag.rejected_mean_excess == pytest.approx(-0.03)
    assert diag.approved_minus_rejected_excess == pytest.approx(0.08)
    assert diag.by_action["APPROVE"].hit_rate == pytest.approx(1.0)
    assert diag.by_action["REJECT"].hit_rate == pytest.approx(0.0)
    assert diag.overall_outcome.count == 2
    assert diag.overall_outcome.hit_rate == pytest.approx(0.5)


def test_pending_decisions_excluded_from_outcomes() -> None:
    decisions = [
        _decision("AAPL", HumanAction.APPROVE),
        _decision("MSFT", HumanAction.APPROVE),  # no performance record
    ]
    perf = [_perf("AAPL", 0.04, OutcomeLabel.OUTPERFORMED)]
    diag = diagnose_decisions(decisions, perf)

    assert diag.total_decisions == 2
    assert diag.decisions_with_outcomes == 1
    assert diag.pending_outcomes == 1
    assert diag.overall_outcome.count == 1
    assert diag.overall_outcome.mean_excess_return == pytest.approx(0.04)


def test_hit_uses_outperformed_label_not_sign() -> None:
    # Positive excess but labeled NEUTRAL must not count as a hit.
    decisions = [_decision("AAPL", HumanAction.APPROVE)]
    perf = [_perf("AAPL", 0.0005, OutcomeLabel.NEUTRAL)]
    diag = diagnose_decisions(decisions, perf)
    assert diag.overall_outcome.hit_rate == pytest.approx(0.0)
    assert diag.overall_outcome.mean_excess_return == pytest.approx(0.0005)


def test_override_impact_split() -> None:
    decisions = [
        _decision("AAPL", HumanAction.OVERRIDE, override=True),
        _decision("MSFT", HumanAction.APPROVE, override=False),
    ]
    perf = [
        _perf("AAPL", -0.02, OutcomeLabel.UNDERPERFORMED),
        _perf("MSFT", 0.06, OutcomeLabel.OUTPERFORMED),
    ]
    diag = diagnose_decisions(decisions, perf)

    assert diag.override_impact["overridden"].count == 1
    assert diag.override_impact["overridden"].mean_excess_return == pytest.approx(-0.02)
    assert diag.override_impact["not_overridden"].mean_excess_return == pytest.approx(0.06)


def test_by_conviction_tier_groups() -> None:
    decisions = [
        _decision("AAPL", HumanAction.APPROVE, conviction=0.2),  # low
        _decision("MSFT", HumanAction.APPROVE, conviction=0.5),  # medium
        _decision("NVDA", HumanAction.APPROVE, conviction=0.9),  # high
    ]
    perf = [
        _perf("AAPL", -0.01, OutcomeLabel.UNDERPERFORMED),
        _perf("MSFT", 0.02, OutcomeLabel.OUTPERFORMED),
        _perf("NVDA", 0.07, OutcomeLabel.OUTPERFORMED),
    ]
    diag = diagnose_decisions(decisions, perf)

    assert set(diag.by_conviction_tier) == {"low", "medium", "high"}
    assert diag.by_conviction_tier["low"].mean_excess_return == pytest.approx(-0.01)
    assert diag.by_conviction_tier["high"].mean_excess_return == pytest.approx(0.07)


def test_join_is_strictly_by_decision_id() -> None:
    decisions = [_decision("AAPL", HumanAction.APPROVE)]
    # Performance record for an asset with no decision must be ignored.
    perf = [_perf("ZZZZ", 0.99, OutcomeLabel.OUTPERFORMED)]
    diag = diagnose_decisions(decisions, perf)
    assert diag.decisions_with_outcomes == 0
    assert diag.pending_outcomes == 1
    assert diag.overall_outcome.count == 0


# ---------------------------------------------------------------------------
# Determinism & I/O
# ---------------------------------------------------------------------------


def test_diagnose_is_deterministic() -> None:
    decisions = [
        _decision("AAPL", HumanAction.APPROVE),
        _decision("MSFT", HumanAction.REJECT),
    ]
    perf = [
        _perf("AAPL", 0.05, OutcomeLabel.OUTPERFORMED),
        _perf("MSFT", -0.03, OutcomeLabel.UNDERPERFORMED),
    ]
    assert diagnose_decisions(decisions, perf) == diagnose_decisions(decisions, perf)


def test_diagnostics_json_is_deterministic_and_sorted() -> None:
    diag = diagnose_decisions(
        [_decision("AAPL", HumanAction.APPROVE)],
        [_perf("AAPL", 0.05, OutcomeLabel.OUTPERFORMED)],
    )
    first = diagnostics_to_json(diag)
    assert first == diagnostics_to_json(diag)
    # sorted keys: action_counts precedes total_decisions in the serialized body
    assert first.index("approval_rate") < first.index("total_decisions")


def test_performance_records_frame_roundtrip() -> None:
    records = [
        _perf("AAPL", 0.05, OutcomeLabel.OUTPERFORMED),
        _perf("MSFT", -0.03, OutcomeLabel.UNDERPERFORMED),
    ]
    frame = performance_records_to_frame(records)
    assert performance_records_from_frame(frame) == records
