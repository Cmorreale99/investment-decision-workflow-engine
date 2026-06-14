"""AI reasoning layer: role-specific agents behind a provider seam.

Phase 8. Agents reason only over grounded ``ContextPacket`` objects,
produce schema-validated structured outputs whose every claim cites a
packet field, and never execute trades or bypass human review. The
default ``StubProvider`` is deterministic and fully offline; the optional
Anthropic provider sits behind the same ``ReasoningProvider`` protocol and
imports its SDK lazily (see :func:`build_provider`).
"""

from __future__ import annotations

from .base import (
    CITABLE_FIELDS,
    AgentOutput,
    AgentRole,
    CitedClaim,
    ReasoningProvider,
    assert_grounded,
    field_is_populated,
)
from .orchestrator import NEXT_STEP, Orchestrator
from .roles import Agent, AnalystAgent, RiskAgent, StrategyAgent
from .stub_provider import StubProvider

#: Values of ``settings.reasoning_provider`` that select the offline stub.
_OFFLINE_NAMES: frozenset[str] = frozenset({"", "stub", "offline", "none"})


def build_provider(name: str | None = None, **kwargs) -> ReasoningProvider:
    """Select a reasoning provider from ``settings.reasoning_provider``.

    ``None`` / ``"stub"`` / ``"offline"`` (the default) returns the
    deterministic offline ``StubProvider``. ``"anthropic"`` constructs the
    Anthropic-backed provider, importing the SDK lazily at that point — so
    the import only happens when a caller explicitly opts in.
    """
    key = (name or "").strip().lower()
    if key in _OFFLINE_NAMES:
        return StubProvider()
    if key == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(**kwargs)
    raise ValueError(
        f"unknown reasoning_provider {name!r} (expected one of: stub, anthropic)"
    )


__all__ = [
    "Agent",
    "AgentOutput",
    "AgentRole",
    "AnalystAgent",
    "CITABLE_FIELDS",
    "CitedClaim",
    "NEXT_STEP",
    "Orchestrator",
    "ReasoningProvider",
    "RiskAgent",
    "StrategyAgent",
    "StubProvider",
    "assert_grounded",
    "build_provider",
    "field_is_populated",
]
