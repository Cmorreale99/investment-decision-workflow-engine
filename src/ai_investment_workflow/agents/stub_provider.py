"""Deterministic, fully offline reasoning provider (the default).

``StubProvider`` is the zero-dependency reasoning engine the test suite
and the offline pipeline run on. It produces structured ``AgentOutput``
deterministically from ``ContextPacket`` facts — same packet in, same
findings out — and every claim it emits cites the packet field it was
derived from, so it satisfies the grounding contract by construction.

It performs no I/O and imports no LLM SDK. The role logic is intentionally
simple, readable rules over the packet; it is a baseline, not an attempt
to mimic a language model.
"""

from __future__ import annotations

from ..schemas import ContextPacket, SystemAction
from .base import AgentOutput, AgentRole, CitedClaim


def _conviction_from_composite(composite: float) -> float:
    """Map composite score in [-1, 1] onto a [0, 1] conviction."""
    return round(max(0.0, min(1.0, (composite + 1.0) / 2.0)), 4)


def _active_risk_flags(packet: ContextPacket) -> list[str]:
    return sorted(name for name, tripped in packet.risk_flags.items() if tripped)


class StubProvider:
    """Deterministic offline reasoning over a single ``(role, packet)``."""

    name = "stub"

    def analyze(self, role: AgentRole, packet: ContextPacket) -> AgentOutput:
        if role is AgentRole.ANALYST:
            return self._analyst(packet)
        if role is AgentRole.STRATEGY:
            return self._strategy(packet)
        if role is AgentRole.RISK:
            return self._risk(packet)
        raise ValueError(f"unknown agent role: {role!r}")  # pragma: no cover

    # ------------------------------------------------------------------
    # Analyst: strengths, weaknesses, recent changes
    # ------------------------------------------------------------------
    def _analyst(self, packet: ContextPacket) -> AgentOutput:
        signal = packet.strategy_signal
        snapshot = packet.company_snapshot
        rationale: list[CitedClaim] = []
        risks: list[CitedClaim] = []

        if signal.composite_score >= 0:
            rationale.append(
                CitedClaim(
                    claim=(
                        f"Composite score {signal.composite_score:+.2f} ranks the "
                        f"candidate favorably (rank {signal.rank})."
                    ),
                    field="strategy_signal",
                )
            )
        else:
            risks.append(
                CitedClaim(
                    claim=(
                        f"Composite score {signal.composite_score:+.2f} is negative "
                        f"(rank {signal.rank})."
                    ),
                    field="strategy_signal",
                )
            )

        rationale.append(
            CitedClaim(
                claim=f"Operates in the {snapshot.sector} sector at {snapshot.price:.2f}.",
                field="company_snapshot",
            )
        )

        ret_3m = packet.recent_performance.get("return_3m")
        last_excess = packet.recent_performance.get("last_excess_return")
        if ret_3m is not None:
            target = rationale if ret_3m >= 0 else risks
            target.append(
                CitedClaim(
                    claim=f"Trailing 3-month return is {ret_3m:+.2%}.",
                    field="recent_performance",
                )
            )
        elif last_excess is not None:
            target = rationale if last_excess >= 0 else risks
            target.append(
                CitedClaim(
                    claim=f"Most recent excess return was {last_excess:+.2%}.",
                    field="recent_performance",
                )
            )

        if packet.prior_decisions:
            rationale.append(
                CitedClaim(
                    claim=f"{len(packet.prior_decisions)} prior decision(s) on record.",
                    field="prior_decisions",
                )
            )
        if packet.human_notes:
            rationale.append(
                CitedClaim(
                    claim="Prior human notes are available for context.",
                    field="human_notes",
                )
            )

        conviction = _conviction_from_composite(signal.composite_score)
        action = (
            SystemAction.BUY
            if signal.composite_score >= 0.4
            else SystemAction.SELL
            if signal.composite_score <= -0.4
            else SystemAction.HOLD
        )
        summary = (
            f"{packet.asset_id}: composite {signal.composite_score:+.2f}, "
            f"rank {signal.rank} in {snapshot.sector}."
        )
        return AgentOutput(
            role=AgentRole.ANALYST,
            summary=summary,
            rationale=rationale,
            risks=risks,
            conviction=conviction,
            suggested_action=action,
        )

    # ------------------------------------------------------------------
    # Strategy: strategy fit and signal conflict
    # ------------------------------------------------------------------
    def _strategy(self, packet: ContextPacket) -> AgentOutput:
        signal = packet.strategy_signal
        rationale: list[CitedClaim] = []
        risks: list[CitedClaim] = []

        for name in sorted(signal.strategy_scores):
            score = signal.strategy_scores[name]
            target = rationale if score >= 0 else risks
            target.append(
                CitedClaim(
                    claim=f"{name.capitalize()} strategy score is {score:+.2f}.",
                    field="strategy_signal",
                )
            )

        if signal.strategy_conflict:
            risks.append(
                CitedClaim(
                    claim="Strategies disagree (strategy_conflict flag is set).",
                    field="strategy_signal",
                )
            )
        else:
            rationale.append(
                CitedClaim(
                    claim="Strategy signals are aligned (no conflict flag).",
                    field="strategy_signal",
                )
            )

        conviction = _conviction_from_composite(signal.composite_score)
        if signal.strategy_conflict:
            conviction = round(conviction * 0.8, 4)

        if signal.strategy_conflict:
            action = SystemAction.WATCHLIST
        elif signal.composite_score >= 0.4:
            action = SystemAction.BUY
        elif signal.composite_score <= -0.4:
            action = SystemAction.SELL
        else:
            action = SystemAction.HOLD

        summary = (
            f"{packet.asset_id}: composite {signal.composite_score:+.2f}; "
            f"conflict={signal.strategy_conflict}."
        )
        return AgentOutput(
            role=AgentRole.STRATEGY,
            summary=summary,
            rationale=rationale,
            risks=risks,
            conviction=conviction,
            suggested_action=action,
        )

    # ------------------------------------------------------------------
    # Risk: volatility, liquidity, exposure, constraints
    # ------------------------------------------------------------------
    def _risk(self, packet: ContextPacket) -> AgentOutput:
        snapshot = packet.company_snapshot
        rationale: list[CitedClaim] = []
        risks: list[CitedClaim] = []

        active = _active_risk_flags(packet)
        for flag in active:
            risks.append(
                CitedClaim(
                    claim=f"Risk flag {flag!r} is tripped.",
                    field="risk_flags",
                )
            )
        if packet.risk_flags and not active:
            rationale.append(
                CitedClaim(
                    claim="No hard risk rule is violated.",
                    field="risk_flags",
                )
            )

        # Sector exposure is always citable from the snapshot.
        rationale.append(
            CitedClaim(
                claim=f"Sector exposure to {snapshot.sector} is noted for sizing.",
                field="company_snapshot",
            )
        )

        # Conviction here reflects risk comfort: high when nothing tripped.
        n_flags = len(packet.risk_flags)
        if n_flags:
            conviction = round(1.0 - len(active) / n_flags, 4)
        else:
            conviction = 0.5  # no risk model available -> neutral
        action = SystemAction.WATCHLIST if active else SystemAction.HOLD

        summary = (
            f"{packet.asset_id}: {len(active)} of {n_flags} risk flag(s) tripped."
            if n_flags
            else f"{packet.asset_id}: no risk flags evaluated."
        )
        return AgentOutput(
            role=AgentRole.RISK,
            summary=summary,
            rationale=rationale,
            risks=risks,
            conviction=conviction,
            suggested_action=action,
        )
