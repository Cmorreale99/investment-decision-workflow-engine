"""Retrieval-layer building blocks: source protocol and grounding helpers.

A ``ContextSource`` is a named provider of ContextPacket fields for one
``(asset_id, as_of)`` pair. Sources only *fetch* facts that already exist
in upstream artifacts (Phase 2 prices, Phase 4 signals, Phase 6
performance records, human notes) — they never invent values.

The merge helpers here enforce that discipline mechanically:

- a source may only contribute keys that are real ``ContextPacket``
  fields (``SOURCE_FIELDS``);
- two sources may not contribute the same field, so every populated
  field has exactly one provenance.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Protocol, Sequence, runtime_checkable

from pydantic import BaseModel

#: ContextPacket fields a source may populate. ``asset_id`` and
#: ``timestamp`` are identity fields supplied by the caller, never by a
#: source.
SOURCE_FIELDS: frozenset[str] = frozenset(
    {
        "company_snapshot",
        "strategy_signal",
        "recent_performance",
        "risk_flags",
        "prior_decisions",
        "human_notes",
        "research_notes",
    }
)


@runtime_checkable
class ContextSource(Protocol):
    """Named provider of ContextPacket fields for ``(asset_id, as_of)``.

    ``fetch`` returns a possibly-empty dict whose keys are a subset of
    ``SOURCE_FIELDS``. Missing inputs degrade to an empty dict (or empty
    field values) — never an exception, never fabricated data.
    """

    name: str

    def fetch(self, asset_id: str, as_of: date) -> dict:
        ...


def canonical_json(model: BaseModel) -> str:
    """Deterministic JSON for a schema object: sorted keys, no whitespace.

    Used for byte-identical JSONL persistence and for hashing schema
    objects into embedding seeds.
    """
    return json.dumps(
        model.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def merge_source_payloads(
    sources: Sequence[ContextSource],
    asset_id: str,
    as_of: date,
) -> tuple[dict[str, object], dict[str, str]]:
    """Fetch every source and merge into ``(merged_fields, provenance)``.

    ``provenance`` maps each populated field to the name of the source
    that produced it. Unknown field keys and duplicate contributions
    raise ``ValueError`` so the audit trail stays unambiguous.
    """
    merged: dict[str, object] = {}
    provenance: dict[str, str] = {}
    for source in sources:
        payload = source.fetch(asset_id, as_of)
        unknown = set(payload) - SOURCE_FIELDS
        if unknown:
            raise ValueError(
                f"source {source.name!r} returned non-ContextPacket fields: "
                f"{sorted(unknown)}"
            )
        for field, value in payload.items():
            if field in merged:
                raise ValueError(
                    f"field {field!r} provided by both "
                    f"{provenance[field]!r} and {source.name!r}"
                )
            merged[field] = value
            provenance[field] = source.name
    return merged, provenance
