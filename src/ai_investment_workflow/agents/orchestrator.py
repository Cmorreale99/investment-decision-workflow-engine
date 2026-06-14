"""Orchestrator: merge the three agent outputs into one Recommendation.

Runs the Analyst, Strategy, and Risk agents over a single ``ContextPacket``
and folds their grounded findings into exactly one schema-validated
``Recommendation``. The merge is deterministic and rule-based:

- rationale = Analyst + Strategy rationale claims (each rendered with its
  cited packet field, so the audit trail survives into the recommendation);
- risks = risk claims from all three agents;
- conviction = a fixed weighted blend of the three agents' convictions;
- action = a deterministic rule over the composite score, strategy
  conflict, and any tripped risk flag — risk flags cap the action at
  ``WATCHLIST`` so a flagged candidate is never auto-promoted to ``BUY``.

The orchestrator never executes a trade and never marks anything approved:
``suggested_next_step`` always routes the candidate to human review.
"""

from __future__ import annotations

from ..schemas import ContextPacket, Recommendation, SystemAction
from .base import AgentOutput, AgentRole, CitedClaim
from .roles import AnalystAgent, Agent, RiskAgent, StrategyAgent

#: Conviction blend weights. Risk is a damper, not the primary driver.
_WEIGHTS: dict[AgentRole, float] = {
    AgentRole.ANALYST: 0.4,
    AgentRole.STRATEGY: 0.4,
    AgentRole.RISK: 0.2,
}

#: Human review is mandatory; this is never an approval.
NEXT_STEP: str = "Review manually before approval"


def _render(claim: CitedClaim) -> str:
    """Render a claim with its provenance so the citation is auditable."""
    return f"{claim.claim} [cf. {claim.field}]"


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


class Orchestrator:
    """Coordinates the three role agents and emits one Recommendation."""

    def __init__(self, provider) -> None:
        self._agents: dict[AgentRole, Agent] = {
            AgentRole.ANALYST: AnalystAgent(provider),
            AgentRole.STRATEGY: StrategyAgent(provider),
            AgentRole.RISK: RiskAgent(provider),
        }

    def analyze(self, packet: ContextPacket) -> dict[AgentRole, AgentOutput]:
        """Run every agent; outputs are grounding-checked inside each agent."""
        return {role: agent.run(packet) for role, agent in self._agents.items()}

    def recommend(self, packet: ContextPacket) -> Recommendation:
        outputs = self.analyze(packet)
        analyst = outputs[AgentRole.ANALYST]
        strategy = outputs[AgentRole.STRATEGY]
        risk = outputs[AgentRole.RISK]

        rationale = _dedupe(
            [_render(c) for c in analyst.rationale]
            + [_render(c) for c in strategy.rationale]
        )
        risks = _dedupe(
            [_render(c) for c in analyst.risks]
            + [_render(c) for c in strategy.risks]
            + [_render(c) for c in risk.risks]
        )

        conviction = round(
            sum(_WEIGHTS[role] * out.conviction for role, out in outputs.items()), 4
        )
        conviction = max(0.0, min(1.0, conviction))

        action = self._decide_action(packet, conviction, bool(risk.risks))

        return Recommendation(
            recommendation_id=self._recommendation_id(packet),
            asset_id=packet.asset_id,
            timestamp=packet.timestamp,
            action=action,
            conviction=conviction,
            rationale=rationale,
            risks=risks,
            suggested_next_step=NEXT_STEP,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _recommendation_id(packet: ContextPacket) -> str:
        return f"rec_{packet.timestamp:%Y_%m_%d}_{packet.asset_id}"

    @staticmethod
    def _decide_action(
        packet: ContextPacket, conviction: float, has_risk: bool
    ) -> SystemAction:
        signal = packet.strategy_signal
        flag_tripped = any(packet.risk_flags.values())

        if flag_tripped:
            # A hard risk flag never auto-promotes past human review.
            return SystemAction.WATCHLIST
        if signal.composite_score <= -0.4:
            return SystemAction.SELL
        if (
            signal.composite_score >= 0.4
            and conviction >= 0.6
            and not signal.strategy_conflict
        ):
            return SystemAction.BUY
        if signal.strategy_conflict or has_risk or conviction >= 0.5:
            return SystemAction.WATCHLIST
        return SystemAction.HOLD
