"""Phase 8 AI-reasoning-layer tests. All offline, on the stub provider."""

from __future__ import annotations

import ast
import sys
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from ai_investment_workflow.agents import (
    CITABLE_FIELDS,
    NEXT_STEP,
    AgentOutput,
    AgentRole,
    AnalystAgent,
    CitedClaim,
    Orchestrator,
    RiskAgent,
    StrategyAgent,
    StubProvider,
    assert_grounded,
    build_provider,
)
from ai_investment_workflow.features import compute_feature_frame
from ai_investment_workflow.ingestion import FixtureProvider
from ai_investment_workflow.rag import (
    NotesSource,
    PerformanceSource,
    SignalSource,
    SnapshotSource,
    build_context_packet,
)
from ai_investment_workflow.schemas import (
    CompanySnapshot,
    ContextPacket,
    Recommendation,
    StrategySignal,
    SystemAction,
)
from ai_investment_workflow.strategies import build_strategy_signals

TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "SPY"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def grounded_packet() -> ContextPacket:
    """A fully grounded packet built from the bundled fixture provider."""
    provider = FixtureProvider()
    prices = provider.fetch(TICKERS, date(2025, 1, 1), date(2025, 6, 1))
    features = compute_feature_frame(prices)
    as_of = features["date"].max()
    sources = [
        SnapshotSource(prices),
        SignalSource(features, prices=prices),
        PerformanceSource(None),
        NotesSource(Path("nonexistent_notes.jsonl")),
    ]
    packet = build_context_packet("AAPL", as_of, sources)
    assert packet is not None
    return packet


def _packet(
    *,
    composite: float,
    conflict: bool = False,
    risk_flags: dict[str, bool] | None = None,
    scores: dict[str, float] | None = None,
) -> ContextPacket:
    """Construct a synthetic, schema-valid packet for action-rule tests."""
    as_of = date(2026, 4, 1)
    return ContextPacket(
        asset_id="AAPL",
        timestamp=as_of,
        company_snapshot=CompanySnapshot(
            asset_id="AAPL", timestamp=as_of, sector="Technology", price=182.34
        ),
        strategy_signal=StrategySignal(
            asset_id="AAPL",
            timestamp=as_of,
            strategy_scores=scores or {"value": composite, "momentum": composite},
            composite_score=composite,
            rank=1,
            strategy_conflict=conflict,
        ),
        risk_flags=risk_flags or {},
    )


# ---------------------------------------------------------------------------
# CitedClaim / grounding contract
# ---------------------------------------------------------------------------


def test_citable_fields_match_schema() -> None:
    expected = set(ContextPacket.model_fields) - {"asset_id", "timestamp"}
    assert set(CITABLE_FIELDS) == expected


def test_cited_claim_rejects_non_citable_field() -> None:
    with pytest.raises(ValidationError):
        CitedClaim(claim="not a real fact field", field="asset_id")
    with pytest.raises(ValidationError):
        CitedClaim(claim="made up", field="totally_invented")


def test_assert_grounded_rejects_unpopulated_field() -> None:
    packet = _packet(composite=0.5)  # human_notes is empty
    bad = AgentOutput(
        role=AgentRole.ANALYST,
        summary="x",
        rationale=[CitedClaim(claim="cites empty notes", field="human_notes")],
        conviction=0.5,
        suggested_action=SystemAction.HOLD,
    )
    with pytest.raises(ValueError, match="human_notes"):
        assert_grounded(bad, packet)


def test_assert_grounded_accepts_required_fields() -> None:
    packet = _packet(composite=0.5)
    ok = AgentOutput(
        role=AgentRole.ANALYST,
        summary="x",
        rationale=[CitedClaim(claim="snapshot fact", field="company_snapshot")],
        risks=[CitedClaim(claim="signal fact", field="strategy_signal")],
        conviction=0.5,
        suggested_action=SystemAction.HOLD,
    )
    assert_grounded(ok, packet)  # does not raise


# ---------------------------------------------------------------------------
# StubProvider
# ---------------------------------------------------------------------------


def test_stub_provider_is_deterministic(grounded_packet) -> None:
    provider = StubProvider()
    for role in AgentRole:
        assert provider.analyze(role, grounded_packet) == provider.analyze(
            role, grounded_packet
        )


@pytest.mark.parametrize("role", list(AgentRole))
def test_stub_outputs_are_grounded(grounded_packet, role) -> None:
    output = StubProvider().analyze(role, grounded_packet)
    assert output.role is role
    assert 0.0 <= output.conviction <= 1.0
    assert output.summary
    # Every claim must cite a populated packet field.
    assert_grounded(output, grounded_packet)
    for claim in [*output.rationale, *output.risks]:
        assert claim.field in CITABLE_FIELDS


def test_stub_analyst_flags_negative_composite() -> None:
    out = StubProvider().analyze(AgentRole.ANALYST, _packet(composite=-0.6))
    # A negative composite shows up as a risk, not a rationale strength.
    assert any("composite" in c.claim.lower() for c in out.risks)


def test_stub_risk_reports_tripped_flags() -> None:
    packet = _packet(composite=0.5, risk_flags={"high_volatility": True, "illiquid": False})
    out = StubProvider().analyze(AgentRole.RISK, packet)
    risk_text = " ".join(c.claim for c in out.risks)
    assert "high_volatility" in risk_text
    assert all(c.field == "risk_flags" for c in out.risks)
    assert out.conviction == pytest.approx(0.5)  # 1 of 2 flags tripped


# ---------------------------------------------------------------------------
# Role agents
# ---------------------------------------------------------------------------


def test_role_agents_run_and_enforce_role(grounded_packet) -> None:
    provider = StubProvider()
    assert AnalystAgent(provider).run(grounded_packet).role is AgentRole.ANALYST
    assert StrategyAgent(provider).run(grounded_packet).role is AgentRole.STRATEGY
    assert RiskAgent(provider).run(grounded_packet).role is AgentRole.RISK


def test_agent_rejects_provider_role_mismatch(grounded_packet) -> None:
    class WrongRoleProvider:
        name = "wrong"

        def analyze(self, role, packet):
            return StubProvider().analyze(AgentRole.RISK, packet)

    with pytest.raises(ValueError, match="role"):
        AnalystAgent(WrongRoleProvider()).run(grounded_packet)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def test_orchestrator_emits_valid_recommendation(grounded_packet) -> None:
    rec = Orchestrator(StubProvider()).recommend(grounded_packet)

    assert isinstance(rec, Recommendation)
    assert rec.asset_id == grounded_packet.asset_id
    assert rec.timestamp == grounded_packet.timestamp
    assert rec.action in set(SystemAction)
    assert 0.0 <= rec.conviction <= 1.0
    assert rec.recommendation_id == (
        f"rec_{grounded_packet.timestamp:%Y_%m_%d}_{grounded_packet.asset_id}"
    )
    # Human review is mandatory: this is never an approval.
    assert rec.suggested_next_step == NEXT_STEP
    # Citations survive into the recommendation strings.
    for line in [*rec.rationale, *rec.risks]:
        assert "[cf. " in line


def test_orchestrator_is_deterministic(grounded_packet) -> None:
    a = Orchestrator(StubProvider()).recommend(grounded_packet)
    b = Orchestrator(StubProvider()).recommend(grounded_packet)
    assert a == b


def test_risk_flag_caps_action_at_watchlist() -> None:
    # Strong positive composite would otherwise BUY, but a tripped flag caps it.
    packet = _packet(composite=0.9, risk_flags={"high_volatility": True})
    rec = Orchestrator(StubProvider()).recommend(packet)
    assert rec.action is SystemAction.WATCHLIST


def test_strong_positive_clean_packet_buys() -> None:
    packet = _packet(composite=0.9, risk_flags={"high_volatility": False})
    rec = Orchestrator(StubProvider()).recommend(packet)
    assert rec.action is SystemAction.BUY


def test_strong_negative_packet_sells() -> None:
    packet = _packet(composite=-0.8, risk_flags={"high_volatility": False})
    rec = Orchestrator(StubProvider()).recommend(packet)
    assert rec.action is SystemAction.SELL


def test_conflict_without_flags_watchlists() -> None:
    packet = _packet(composite=0.9, conflict=True, risk_flags={"high_volatility": False})
    rec = Orchestrator(StubProvider()).recommend(packet)
    assert rec.action is SystemAction.WATCHLIST


# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", [None, "", "stub", "offline", "STUB"])
def test_build_provider_defaults_to_stub(name) -> None:
    assert isinstance(build_provider(name), StubProvider)


def test_build_provider_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown reasoning_provider"):
        build_provider("gpt")


# ---------------------------------------------------------------------------
# Guardrail: the LLM SDK is never imported eagerly
# ---------------------------------------------------------------------------


def _top_level_import_roots(path: Path) -> set[str]:
    """Module-scope import roots only (not nested-in-function imports)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def test_no_eager_llm_sdk_imports_in_agents() -> None:
    agents_dir = Path(__file__).resolve().parents[1] / "src" / "ai_investment_workflow" / "agents"
    targets = sorted(agents_dir.glob("*.py"))
    assert targets
    for path in targets:
        assert "anthropic" not in _top_level_import_roots(path), (
            f"{path} imports the anthropic SDK at module scope; it must be lazy"
        )


def test_importing_agents_does_not_load_anthropic() -> None:
    # Importing the package (already done at module top) must not pull the SDK.
    import ai_investment_workflow.agents  # noqa: F401
    import ai_investment_workflow.agents.anthropic_provider  # noqa: F401

    assert "anthropic" not in sys.modules
