"""AI reasoning layer: provider seam, role outputs, and grounding rules.

Phase 8. The agents reason *only* over a ``ContextPacket`` and may not
introduce facts that aren't in it. That discipline is enforced
mechanically here, the same way Phase 7 enforces packet provenance:

- a reasoning result is a structured ``AgentOutput`` (schema-validated);
- every rationale and risk item is a ``CitedClaim`` that names the
  ``ContextPacket`` field it draws from (``CITABLE_FIELDS``);
- ``assert_grounded`` rejects any claim whose cited field is absent or
  empty in the packet, so a hallucinated fact cannot survive review.

The ``ReasoningProvider`` protocol is the only seam an LLM may plug into.
The default ``StubProvider`` is fully offline and deterministic; the
optional Anthropic provider lives behind the same protocol and imports
its SDK lazily (see ``anthropic_provider``). No agent executes trades or
bypasses human review.
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..schemas import ContextPacket, SystemAction

#: ``ContextPacket`` identity fields a claim may never cite as a "fact".
_IDENTITY_FIELDS: frozenset[str] = frozenset({"asset_id", "timestamp"})

#: Fact-bearing packet fields a claim is allowed to cite. Derived from the
#: schema so it stays in sync if ``ContextPacket`` gains/loses a field.
CITABLE_FIELDS: frozenset[str] = frozenset(ContextPacket.model_fields) - _IDENTITY_FIELDS


class AgentRole(str, Enum):
    """The three role-specific reviewers in the reasoning layer."""

    ANALYST = "analyst"
    STRATEGY = "strategy"
    RISK = "risk"


class CitedClaim(BaseModel):
    """One reasoning statement plus the packet field it is grounded in.

    ``field`` must be a real, citable ``ContextPacket`` field name; whether
    that field is actually *populated* for a given packet is checked at
    grounding time by :func:`assert_grounded`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    claim: str = Field(..., min_length=1)
    field: str = Field(..., min_length=1)

    @field_validator("field")
    @classmethod
    def _field_is_citable(cls, value: str) -> str:
        if value not in CITABLE_FIELDS:
            raise ValueError(
                f"claim cites {value!r}, which is not a ContextPacket fact field "
                f"(allowed: {sorted(CITABLE_FIELDS)})"
            )
        return value


class AgentOutput(BaseModel):
    """Structured, schema-validated output of a single role agent."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: AgentRole
    summary: str = Field(..., min_length=1)
    rationale: list[CitedClaim] = Field(default_factory=list)
    risks: list[CitedClaim] = Field(default_factory=list)
    conviction: float = Field(..., ge=0.0, le=1.0)
    suggested_action: SystemAction


@runtime_checkable
class ReasoningProvider(Protocol):
    """Swappable reasoning engine for a single ``(role, packet)`` pair.

    Implementations must return an ``AgentOutput`` whose every claim cites
    a packet field. They must not execute trades, mutate the packet, or
    reach for facts outside it.
    """

    name: str

    def analyze(self, role: AgentRole, packet: ContextPacket) -> AgentOutput:
        ...


def field_is_populated(packet: ContextPacket, field: str) -> bool:
    """Whether ``field`` carries a real value for this packet.

    Required schema objects (``company_snapshot``, ``strategy_signal``)
    are always populated; the optional dict/list fields count as populated
    only when non-empty.
    """
    value = getattr(packet, field)
    if isinstance(value, (dict, list, str)):
        return bool(value)
    return value is not None


def assert_grounded(output: AgentOutput, packet: ContextPacket) -> None:
    """Reject any claim whose cited field is absent or empty in the packet.

    This is the Phase 8 analogue of the retrieval layer's grounding
    guard: a rationale or risk that cannot be traced to a populated packet
    field is treated as fabricated and raises ``ValueError``.
    """
    for kind in ("rationale", "risks"):
        for claim in getattr(output, kind):
            if not field_is_populated(packet, claim.field):
                raise ValueError(
                    f"{output.role.value} agent {kind} claim cites {claim.field!r}, "
                    f"which is not populated for {packet.asset_id} at {packet.timestamp}"
                )
