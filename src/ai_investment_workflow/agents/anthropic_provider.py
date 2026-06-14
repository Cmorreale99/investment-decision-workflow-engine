"""Optional Anthropic-backed reasoning provider (gated, lazy import).

This provider is only constructed when ``settings.reasoning_provider`` is
set to ``"anthropic"``. The ``anthropic`` SDK is imported **lazily inside
the constructor** — importing this module (or the ``agents`` package) does
not pull in the SDK, so the offline baseline and the test suite never need
it installed.

It plugs into the same ``ReasoningProvider`` protocol as ``StubProvider``
and returns the same schema-validated ``AgentOutput``. Structured outputs
constrain the response to the ``AgentOutput`` schema; the orchestrator
then runs ``assert_grounded`` exactly as it does for the stub, so a model
that cites a field absent from the packet is rejected rather than trusted.
The model is told to cite only packet fields and never to recommend trade
execution or bypass human review.
"""

from __future__ import annotations

from ..schemas import ContextPacket
from .base import CITABLE_FIELDS, AgentOutput, AgentRole

#: Default model. Opus 4.8 is Anthropic's most capable Opus-tier model.
DEFAULT_MODEL: str = "claude-opus-4-8"

_ROLE_BRIEF: dict[AgentRole, str] = {
    AgentRole.ANALYST: (
        "You are the Analyst. Summarize strengths, weaknesses, and recent "
        "changes for the candidate."
    ),
    AgentRole.STRATEGY: (
        "You are the Strategy reviewer. Evaluate strategy fit and whether the "
        "strategy signals conflict."
    ),
    AgentRole.RISK: (
        "You are the Risk reviewer. Assess volatility, liquidity, exposure, and "
        "any tripped risk rules."
    ),
}


def _system_prompt(role: AgentRole) -> str:
    return (
        f"{_ROLE_BRIEF[role]}\n\n"
        "You reason ONLY over the ContextPacket JSON provided in the user "
        "message. Do not introduce any fact that is not present in the packet.\n"
        "Every rationale and risk item is a claim plus the packet field it is "
        "drawn from. The 'field' value MUST be exactly one of: "
        f"{sorted(CITABLE_FIELDS)}. Only cite a field that is present and "
        "non-empty in the packet.\n"
        "Set 'conviction' in [0, 1] and 'suggested_action' to one of BUY, SELL, "
        "HOLD, or WATCHLIST.\n"
        "You do not execute trades and you do not approve decisions; a human "
        "reviewer makes the final call."
    )


class AnthropicProvider:
    """``ReasoningProvider`` backed by the Anthropic Messages API."""

    name = "anthropic"

    def __init__(self, model: str = DEFAULT_MODEL, *, api_key: str | None = None) -> None:
        # Lazy import: keeps the SDK off the import path of the offline baseline.
        import anthropic

        self._model = model
        self._client = (
            anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        )

    def analyze(self, role: AgentRole, packet: ContextPacket) -> AgentOutput:
        response = self._client.messages.parse(
            model=self._model,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            system=_system_prompt(role),
            messages=[
                {
                    "role": "user",
                    "content": (
                        "ContextPacket (the only facts you may use):\n"
                        f"{packet.model_dump_json()}"
                    ),
                }
            ],
            output_format=AgentOutput,
        )
        parsed = response.parsed_output
        if parsed is None:  # refusal / non-conforming output
            raise ValueError(
                f"Anthropic provider returned no structured output for "
                f"{packet.asset_id} ({role.value}); stop_reason={response.stop_reason}"
            )
        # Pin the role to the one we requested regardless of what the model set.
        return parsed.model_copy(update={"role": role})
