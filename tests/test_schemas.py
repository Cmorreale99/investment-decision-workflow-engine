"""Validate every schema: round-trip + negative cases."""

from __future__ import annotations

from datetime import date, datetime

import pytest
from pydantic import ValidationError

from ai_investment_workflow.schemas import (
    CompanySnapshot,
    ContextPacket,
    DecisionRecord,
    FeatureSet,
    HumanAction,
    HumanDecision,
    OutcomeLabel,
    PerformanceRecord,
    Recommendation,
    StrategySignal,
    SystemAction,
)


# ---------------------------------------------------------------------------
# CompanySnapshot
# ---------------------------------------------------------------------------


def _company_snapshot() -> CompanySnapshot:
    return CompanySnapshot(
        asset_id="AAPL",
        timestamp=date(2026, 4, 1),
        sector="Technology",
        price=182.34,
    )


def test_company_snapshot_roundtrip() -> None:
    snap = _company_snapshot()
    restored = CompanySnapshot.model_validate_json(snap.model_dump_json())
    assert restored == snap


def test_company_snapshot_rejects_nonpositive_price() -> None:
    with pytest.raises(ValidationError):
        CompanySnapshot(
            asset_id="AAPL", timestamp=date(2026, 4, 1), sector="Tech", price=0.0
        )


def test_company_snapshot_rejects_missing_required() -> None:
    with pytest.raises(ValidationError):
        CompanySnapshot(asset_id="AAPL", timestamp=date(2026, 4, 1), sector="Tech")  # type: ignore[call-arg]


def test_company_snapshot_is_frozen() -> None:
    snap = _company_snapshot()
    with pytest.raises(ValidationError):
        snap.price = 200.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# FeatureSet
# ---------------------------------------------------------------------------


def test_feature_set_roundtrip() -> None:
    fs = FeatureSet(
        asset_id="AAPL",
        timestamp=date(2026, 4, 1),
        features={"momentum_3m": 0.12, "volatility_30d": 0.21},
    )
    restored = FeatureSet.model_validate_json(fs.model_dump_json())
    assert restored == fs


def test_feature_set_rejects_extra_field() -> None:
    with pytest.raises(ValidationError):
        FeatureSet(
            asset_id="AAPL",
            timestamp=date(2026, 4, 1),
            features={},
            unexpected=True,  # type: ignore[call-arg]
        )


# ---------------------------------------------------------------------------
# StrategySignal
# ---------------------------------------------------------------------------


def _strategy_signal() -> StrategySignal:
    return StrategySignal(
        asset_id="AAPL",
        timestamp=date(2026, 4, 1),
        strategy_scores={"value": -0.4, "momentum": 0.8},
        composite_score=0.35,
        rank=7,
        strategy_conflict=True,
    )


def test_strategy_signal_roundtrip() -> None:
    sig = _strategy_signal()
    restored = StrategySignal.model_validate_json(sig.model_dump_json())
    assert restored == sig


def test_strategy_signal_rejects_out_of_range_composite() -> None:
    with pytest.raises(ValidationError):
        StrategySignal(
            asset_id="AAPL",
            timestamp=date(2026, 4, 1),
            strategy_scores={"value": 0.1},
            composite_score=1.5,
            rank=1,
        )


def test_strategy_signal_rejects_out_of_range_subscore() -> None:
    with pytest.raises(ValidationError):
        StrategySignal(
            asset_id="AAPL",
            timestamp=date(2026, 4, 1),
            strategy_scores={"momentum": 2.0},
            composite_score=0.1,
            rank=1,
        )


def test_strategy_signal_rejects_invalid_rank() -> None:
    with pytest.raises(ValidationError):
        StrategySignal(
            asset_id="AAPL",
            timestamp=date(2026, 4, 1),
            strategy_scores={"value": 0.1},
            composite_score=0.1,
            rank=0,
        )


# ---------------------------------------------------------------------------
# ContextPacket
# ---------------------------------------------------------------------------


def test_context_packet_roundtrip() -> None:
    pkt = ContextPacket(
        asset_id="AAPL",
        timestamp=date(2026, 4, 1),
        company_snapshot=_company_snapshot(),
        strategy_signal=_strategy_signal(),
        recent_performance={"return_3m": 0.12},
        risk_flags={"high_volatility": False},
        prior_decisions=[],
    )
    restored = ContextPacket.model_validate_json(pkt.model_dump_json())
    assert restored == pkt


def test_context_packet_rejects_missing_required() -> None:
    with pytest.raises(ValidationError):
        ContextPacket(  # type: ignore[call-arg]
            asset_id="AAPL",
            timestamp=date(2026, 4, 1),
            company_snapshot=_company_snapshot(),
        )


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------


def _recommendation() -> Recommendation:
    return Recommendation(
        recommendation_id="rec_20260401_AAPL",
        asset_id="AAPL",
        timestamp=date(2026, 4, 1),
        action=SystemAction.WATCHLIST,
        conviction=0.68,
        rationale=["Strong momentum signal", "Positive recent relative performance"],
        risks=["Elevated valuation"],
        suggested_next_step="Review manually before approval",
    )


def test_recommendation_roundtrip() -> None:
    rec = _recommendation()
    restored = Recommendation.model_validate_json(rec.model_dump_json())
    assert restored == rec


def test_recommendation_rejects_out_of_range_conviction() -> None:
    with pytest.raises(ValidationError):
        Recommendation(
            recommendation_id="rec_x",
            asset_id="AAPL",
            timestamp=date(2026, 4, 1),
            action=SystemAction.BUY,
            conviction=1.2,
        )


def test_recommendation_rejects_invalid_action() -> None:
    with pytest.raises(ValidationError):
        Recommendation(
            recommendation_id="rec_x",
            asset_id="AAPL",
            timestamp=date(2026, 4, 1),
            action="NOPE",  # type: ignore[arg-type]
            conviction=0.5,
        )


# ---------------------------------------------------------------------------
# HumanDecision
# ---------------------------------------------------------------------------


def test_human_decision_roundtrip() -> None:
    hd = HumanDecision(
        asset_id="AAPL",
        recommendation_id="rec_20260401_AAPL",
        human_action=HumanAction.WATCHLIST,
        override=True,
        human_notes="Momentum strong, valuation risky before earnings.",
        review_status="completed",
        reviewed_at=datetime(2026, 4, 1, 15, 30, 0),
    )
    restored = HumanDecision.model_validate_json(hd.model_dump_json())
    assert restored == hd


def test_human_decision_rejects_invalid_action() -> None:
    with pytest.raises(ValidationError):
        HumanDecision(
            asset_id="AAPL",
            recommendation_id="rec_x",
            human_action="MAYBE",  # type: ignore[arg-type]
            reviewed_at=datetime(2026, 4, 1, 15, 30, 0),
        )


def test_human_decision_rejects_missing_required() -> None:
    with pytest.raises(ValidationError):
        HumanDecision(  # type: ignore[call-arg]
            asset_id="AAPL",
            recommendation_id="rec_x",
            human_action=HumanAction.APPROVE,
        )


# ---------------------------------------------------------------------------
# DecisionRecord
# ---------------------------------------------------------------------------


def test_decision_record_roundtrip() -> None:
    rec = DecisionRecord(
        decision_id="dec_20260401_AAPL",
        asset_id="AAPL",
        timestamp=date(2026, 4, 1),
        recommendation_id="rec_20260401_AAPL",
        system_action=SystemAction.BUY,
        system_conviction=0.72,
        human_action=HumanAction.WATCHLIST,
        override=True,
        human_notes="Strong momentum, but valuation risk too high.",
        final_status="watchlisted",
    )
    restored = DecisionRecord.model_validate_json(rec.model_dump_json())
    assert restored == rec


def test_decision_record_rejects_out_of_range_conviction() -> None:
    with pytest.raises(ValidationError):
        DecisionRecord(
            decision_id="dec_x",
            asset_id="AAPL",
            timestamp=date(2026, 4, 1),
            recommendation_id="rec_x",
            system_action=SystemAction.BUY,
            system_conviction=-0.1,
            human_action=HumanAction.APPROVE,
            final_status="approved",
        )


# ---------------------------------------------------------------------------
# PerformanceRecord
# ---------------------------------------------------------------------------


def test_performance_record_roundtrip() -> None:
    perf = PerformanceRecord(
        decision_id="dec_20260401_AAPL",
        asset_id="AAPL",
        evaluation_start=date(2026, 4, 1),
        evaluation_end=date(2026, 5, 1),
        asset_return=0.042,
        benchmark_return=0.018,
        excess_return=0.024,
        max_drawdown=-0.031,
        outcome_label=OutcomeLabel.OUTPERFORMED,
    )
    restored = PerformanceRecord.model_validate_json(perf.model_dump_json())
    assert restored == perf


def test_performance_record_rejects_positive_drawdown() -> None:
    with pytest.raises(ValidationError):
        PerformanceRecord(
            decision_id="dec_x",
            asset_id="AAPL",
            evaluation_start=date(2026, 4, 1),
            evaluation_end=date(2026, 5, 1),
            asset_return=0.0,
            benchmark_return=0.0,
            excess_return=0.0,
            max_drawdown=0.05,
            outcome_label=OutcomeLabel.NEUTRAL,
        )


def test_performance_record_rejects_reversed_window() -> None:
    with pytest.raises(ValidationError):
        PerformanceRecord(
            decision_id="dec_x",
            asset_id="AAPL",
            evaluation_start=date(2026, 5, 1),
            evaluation_end=date(2026, 4, 1),
            asset_return=0.0,
            benchmark_return=0.0,
            excess_return=0.0,
            max_drawdown=-0.01,
            outcome_label=OutcomeLabel.NEUTRAL,
        )


def test_performance_record_rejects_invalid_outcome() -> None:
    with pytest.raises(ValidationError):
        PerformanceRecord(
            decision_id="dec_x",
            asset_id="AAPL",
            evaluation_start=date(2026, 4, 1),
            evaluation_end=date(2026, 5, 1),
            asset_return=0.0,
            benchmark_return=0.0,
            excess_return=0.0,
            max_drawdown=-0.01,
            outcome_label="great",  # type: ignore[arg-type]
        )
