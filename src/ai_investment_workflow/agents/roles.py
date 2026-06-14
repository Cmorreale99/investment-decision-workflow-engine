"""Role-specific agents: Analyst, Strategy, and Risk.

Each agent binds one :class:`AgentRole` to a ``ReasoningProvider`` and is
responsible for two things: asking the provider to reason over a
``ContextPacket``, and re-asserting that the result is grounded (every
claim cites a populated packet field). The reasoning itself lives in the
provider, so swapping ``StubProvider`` for the Anthropic provider changes
how the findings are produced without changing this grounding contract.
"""

from __future__ import annotations

from ..schemas import ContextPacket
from .base import AgentOutput, AgentRole, ReasoningProvider, assert_grounded


class Agent:
    """Base role agent: reason via the provider, then enforce grounding."""

    role: AgentRole

    def __init__(self, provider: ReasoningProvider) -> None:
        self._provider = provider

    def run(self, packet: ContextPacket) -> AgentOutput:
        output = self._provider.analyze(self.role, packet)
        if output.role is not self.role:
            raise ValueError(
                f"provider returned role {output.role!r}, expected {self.role!r}"
            )
        assert_grounded(output, packet)
        return output


class AnalystAgent(Agent):
    """Summarizes strengths, weaknesses, and recent changes."""

    role = AgentRole.ANALYST


class StrategyAgent(Agent):
    """Evaluates strategy fit and signal conflict."""

    role = AgentRole.STRATEGY


class RiskAgent(Agent):
    """Reviews volatility, liquidity, exposure, and constraints."""

    role = AgentRole.RISK
