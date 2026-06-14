"""Phase 11 dashboard tests. Offline; Streamlit is never imported."""

from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

from ai_investment_workflow.app import (
    CandidateCard,
    DashboardData,
    build_candidate_cards,
    build_evaluation_summary,
    load_context_packets,
    load_decision_diagnostics,
    record_decision,
)
from ai_investment_workflow.human_review import (
    DECISION_LOG_FILENAME,
    RECOMMENDATIONS_FILENAME,
    load_decision_log,
)
from ai_investment_workflow.rag import PACKETS_FILENAME
from ai_investment_workflow.schemas import (
    CompanySnapshot,
    ContextPacket,
    DecisionRecord,
    HumanAction,
    Recommendation,
    StrategySignal,
    SystemAction,
)

TS = date(2026, 4, 1)
REVIEWED_AT = datetime(2026, 4, 1, 15, 30)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _rec(asset: str, *, action: SystemAction = SystemAction.WATCHLIST, conviction: float = 0.6) -> Recommendation:
    return Recommendation(
        recommendation_id=f"rec_{TS:%Y_%m_%d}_{asset}",
        asset_id=asset,
        timestamp=TS,
        action=action,
        conviction=conviction,
        rationale=["Strong momentum [cf. strategy_signal]"],
        risks=["Elevated valuation [cf. company_snapshot]"],
        suggested_next_step="Review manually before approval",
    )


def _packet(asset: str, *, sector: str = "Technology", conflict: bool = True) -> ContextPacket:
    return ContextPacket(
        asset_id=asset,
        timestamp=TS,
        company_snapshot=CompanySnapshot(
            asset_id=asset, timestamp=TS, sector=sector, price=182.34
        ),
        strategy_signal=StrategySignal(
            asset_id=asset,
            timestamp=TS,
            strategy_scores={"value": -0.4, "momentum": 0.8},
            composite_score=0.2,
            rank=1,
            strategy_conflict=conflict,
        ),
    )


def _record(rec: Recommendation, action: HumanAction, *, final_status: str) -> DecisionRecord:
    return DecisionRecord(
        decision_id=f"dec_{TS:%Y_%m_%d}_{rec.asset_id}",
        asset_id=rec.asset_id,
        timestamp=TS,
        recommendation_id=rec.recommendation_id,
        system_action=rec.action,
        system_conviction=rec.conviction,
        human_action=action,
        override=False,
        human_notes="looks good",
        final_status=final_status,
    )


def _seed(base: Path, *, recs=(), packets=(), decisions=(), diagnostics=None, portfolio=None) -> None:
    if recs:
        (base / RECOMMENDATIONS_FILENAME).write_text(
            "".join(r.model_dump_json() + "\n" for r in recs), encoding="utf-8"
        )
    if packets:
        (base / PACKETS_FILENAME).write_text(
            "".join(p.model_dump_json() + "\n" for p in packets), encoding="utf-8"
        )
    if decisions:
        (base / DECISION_LOG_FILENAME).write_text(
            "".join(d.model_dump_json() + "\n" for d in decisions), encoding="utf-8"
        )
    if diagnostics is not None:
        (base / "decision_diagnostics.json").write_text(
            json.dumps(diagnostics), encoding="utf-8"
        )
    if portfolio is not None:
        pd.DataFrame([portfolio]).to_parquet(base / "portfolio_diagnostics.parquet")


# ---------------------------------------------------------------------------
# Loaders / graceful empty states
# ---------------------------------------------------------------------------


def test_load_empty_directory_degrades(tmp_path) -> None:
    data = DashboardData.load(tmp_path)
    assert data.recommendations == []
    assert data.packets == {}
    assert data.decisions == []
    assert data.decision_diagnostics is None
    assert data.portfolio_diagnostics is None


def test_load_context_packets_present_and_absent(tmp_path) -> None:
    assert load_context_packets(tmp_path / "missing.jsonl") == {}
    _seed(tmp_path, packets=[_packet("AAPL")])
    packets = load_context_packets(tmp_path / PACKETS_FILENAME)
    assert set(packets) == {"AAPL"}
    assert isinstance(packets["AAPL"], ContextPacket)


def test_load_decision_diagnostics_absent_is_none(tmp_path) -> None:
    assert load_decision_diagnostics(tmp_path / "decision_diagnostics.json") is None


def test_dashboard_data_load_populated(tmp_path) -> None:
    rec = _rec("AAPL")
    _seed(
        tmp_path,
        recs=[rec],
        packets=[_packet("AAPL")],
        decisions=[_record(rec, HumanAction.APPROVE, final_status="approved")],
        diagnostics={"total_decisions": 1},
        portfolio={"total_return": 0.05},
    )
    data = DashboardData.load(tmp_path)
    assert [r.asset_id for r in data.recommendations] == ["AAPL"]
    assert "AAPL" in data.packets
    assert len(data.decisions) == 1
    assert data.decision_diagnostics == {"total_decisions": 1}
    assert data.portfolio_diagnostics["total_return"] == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# View models
# ---------------------------------------------------------------------------


def test_build_candidate_cards_joins_packet_and_decision() -> None:
    rec = _rec("AAPL", action=SystemAction.WATCHLIST, conviction=0.68)
    data = DashboardData(
        recommendations=[rec],
        packets={"AAPL": _packet("AAPL", sector="Technology", conflict=True)},
        decisions=[_record(rec, HumanAction.WATCHLIST, final_status="watchlisted")],
    )
    [card] = build_candidate_cards(data)
    assert isinstance(card, CandidateCard)
    assert card.asset_id == "AAPL"
    assert card.sector == "Technology"
    assert card.strategy_scores == {"value": -0.4, "momentum": 0.8}
    assert card.strategy_conflict is True
    assert card.action == "WATCHLIST"
    assert card.conviction == pytest.approx(0.68)
    assert card.reasons == ["Strong momentum [cf. strategy_signal]"]
    assert card.risks == ["Elevated valuation [cf. company_snapshot]"]
    assert card.human_action == "WATCHLIST"
    assert card.final_status == "watchlisted"


def test_build_candidate_cards_degrades_without_packet_or_decision() -> None:
    rec = _rec("MSFT")
    data = DashboardData(recommendations=[rec])  # no packet, no decision
    [card] = build_candidate_cards(data)
    assert card.sector is None
    assert card.composite_score is None
    assert card.strategy_scores == {}
    assert card.strategy_conflict is None
    assert card.human_action is None
    assert card.final_status is None
    assert card.reasons == rec.rationale  # recommendation fields still shown


def test_build_evaluation_summary_extracts_headline_metrics() -> None:
    data = DashboardData(
        decision_diagnostics={
            "total_decisions": 4,
            "approval_rate": 0.5,
            "override_rate": 0.25,
            "approved_minus_rejected_excess": 0.09,
            "overall_outcome": {"hit_rate": 0.75},
        },
        portfolio_diagnostics={
            "total_return": 0.05,
            "excess_total_return": 0.02,
            "hit_rate": 0.6,
            "max_drawdown": -0.03,
        },
    )
    summary = build_evaluation_summary(data)
    assert summary.total_decisions == 4
    assert summary.approval_rate == pytest.approx(0.5)
    assert summary.decision_hit_rate == pytest.approx(0.75)
    assert summary.approved_minus_rejected_excess == pytest.approx(0.09)
    assert summary.portfolio_total_return == pytest.approx(0.05)
    assert summary.portfolio_max_drawdown == pytest.approx(-0.03)


def test_build_evaluation_summary_all_none_when_absent() -> None:
    summary = build_evaluation_summary(DashboardData())
    assert summary.total_decisions is None
    assert summary.approval_rate is None
    assert summary.portfolio_total_return is None


# ---------------------------------------------------------------------------
# Action path (sole write, through human_review)
# ---------------------------------------------------------------------------


def test_record_decision_appends_via_human_review(tmp_path) -> None:
    log = tmp_path / DECISION_LOG_FILENAME
    rec = _rec("AAPL", action=SystemAction.BUY, conviction=0.72)

    record = record_decision(
        rec,
        HumanAction.WATCHLIST,
        override=True,
        notes="valuation risk",
        reviewed_at=REVIEWED_AT,
        log_path=log,
    )
    assert isinstance(record, DecisionRecord)
    assert record.final_status == "watchlisted"
    assert record.system_action is SystemAction.BUY
    assert record.override is True

    [persisted] = load_decision_log(log)
    assert persisted == record


def test_record_decision_is_idempotent(tmp_path) -> None:
    log = tmp_path / DECISION_LOG_FILENAME
    rec = _rec("AAPL")
    record_decision(rec, HumanAction.APPROVE, reviewed_at=REVIEWED_AT, log_path=log)
    first = log.read_bytes()
    # Re-recording the same recommendation replaces, not duplicates.
    record_decision(rec, HumanAction.APPROVE, reviewed_at=REVIEWED_AT, log_path=log)
    assert log.read_bytes() == first
    assert len(load_decision_log(log)) == 1


# ---------------------------------------------------------------------------
# Guardrail: importing the app package must not load Streamlit
# ---------------------------------------------------------------------------


def test_importing_app_package_does_not_load_streamlit() -> None:
    import ai_investment_workflow.app  # noqa: F401

    assert "streamlit" not in sys.modules
