"""Grounded ``ContextPacket`` construction.

Every populated packet field must trace back to exactly one source's
``fetch(...)`` return (enforced by ``merge_source_payloads`` and
re-asserted post-construction by ``_assert_grounded``). Fields no
source provided stay at their schema defaults — they are never filled
in here.
"""

from __future__ import annotations

from datetime import date
from typing import Sequence

from ..schemas import ContextPacket
from .base import SOURCE_FIELDS, ContextSource, merge_source_payloads

#: Fields without which a packet cannot be grounded at all.
REQUIRED_FIELDS: tuple[str, ...] = ("company_snapshot", "strategy_signal")

#: Schema defaults for the optional source-populated fields.
_OPTIONAL_DEFAULTS: dict[str, object] = {
    "recent_performance": {},
    "risk_flags": {},
    "prior_decisions": [],
    "human_notes": [],
    "research_notes": [],
}


def _check_identity(merged: dict[str, object], asset_id: str, as_of: date) -> None:
    """Embedded schema objects must match the requested identity."""
    for field in REQUIRED_FIELDS:
        obj = merged[field]
        if obj.asset_id != asset_id:  # type: ignore[union-attr]
            raise ValueError(
                f"{field} is for asset {obj.asset_id!r}, expected {asset_id!r}"
            )
        if obj.timestamp != as_of:  # type: ignore[union-attr]
            raise ValueError(
                f"{field} is dated {obj.timestamp}, expected {as_of}"
            )


def _assert_grounded(packet: ContextPacket, merged: dict[str, object]) -> None:
    """Every populated field equals a source value; the rest are defaults."""
    for field in SOURCE_FIELDS:
        value = getattr(packet, field)
        if field in merged:
            assert value == merged[field], (
                f"packet field {field!r} diverged from its source value"
            )
        else:
            assert value == _OPTIONAL_DEFAULTS[field], (
                f"packet field {field!r} was populated without a source"
            )


def build_context_packet(
    asset_id: str,
    as_of: date,
    sources: Sequence[ContextSource],
) -> ContextPacket | None:
    """Assemble one grounded packet, or ``None`` if it cannot be grounded.

    Returns ``None`` (rather than raising) when no source can supply the
    required ``company_snapshot`` / ``strategy_signal`` — the pipeline
    skips such assets instead of fabricating data. Identity mismatches
    and duplicate/unknown source fields raise ``ValueError``.
    """
    merged, _provenance = merge_source_payloads(sources, asset_id, as_of)
    if any(merged.get(field) is None for field in REQUIRED_FIELDS):
        return None
    _check_identity(merged, asset_id, as_of)

    packet = ContextPacket(asset_id=asset_id, timestamp=as_of, **merged)
    _assert_grounded(packet, merged)
    return packet
