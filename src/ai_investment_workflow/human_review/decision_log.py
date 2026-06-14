"""Persistence for the decision log: the auditable record of every review.

The decision log is **append-only in intent but byte-deterministic on
disk**: it is deduplicated by ``recommendation_id`` (a re-review replaces
the prior record) and written in a fixed order, so rebuilding the log
from the same set of decisions produces an identical file — the same
reproducibility contract the Phase 7 packet artifact holds to.

This module only reads recommendations and reads/writes the decision log
under ``data/processed/``. It never mutates upstream artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from ..rag import canonical_json
from ..schemas import DecisionRecord, Recommendation

#: Canonical artifacts under ``data/processed/``.
RECOMMENDATIONS_FILENAME: str = "recommendations.jsonl"
DECISION_LOG_FILENAME: str = "decision_log.jsonl"


def load_recommendations(path: str | Path) -> list[Recommendation]:
    """Load Phase 8 recommendations from JSONL, preserving file order."""
    text = Path(path).read_text(encoding="utf-8")
    out: list[Recommendation] = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            out.append(Recommendation.model_validate(json.loads(line)))
    return out


def load_decision_log(path: str | Path) -> list[DecisionRecord]:
    """Load persisted decision records; empty list if the file is absent."""
    path = Path(path)
    if not path.is_file():
        return []
    out: list[DecisionRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(DecisionRecord.model_validate(json.loads(line)))
    return out


def decided_recommendation_ids(records: Iterable[DecisionRecord]) -> set[str]:
    """Recommendation ids that already have a decision on record."""
    return {record.recommendation_id for record in records}


def _dedupe_latest(records: Iterable[DecisionRecord]) -> list[DecisionRecord]:
    """Keep one record per recommendation (last wins), in a fixed order."""
    by_recommendation: dict[str, DecisionRecord] = {}
    for record in records:
        by_recommendation[record.recommendation_id] = record
    return sorted(
        by_recommendation.values(),
        key=lambda r: (r.decision_id, r.recommendation_id),
    )


def decision_log_to_jsonl(records: Iterable[DecisionRecord]) -> str:
    """Deterministic canonical JSONL for a set of decision records."""
    return "".join(canonical_json(r) + "\n" for r in _dedupe_latest(records))


def write_decision_log(
    records: Iterable[DecisionRecord], path: str | Path
) -> Path:
    """Write the full decision log as deterministic canonical JSONL bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(decision_log_to_jsonl(records).encode("utf-8"))
    return path


def append_decisions(
    new_records: Iterable[DecisionRecord], path: str | Path
) -> Path:
    """Merge new records into the existing log and rewrite it deterministically.

    Existing records are loaded first, then the new ones are layered on
    top (a re-review of the same recommendation replaces the old record).
    Running this twice with the same decisions yields a byte-identical file.
    """
    path = Path(path)
    merged = load_decision_log(path) + list(new_records)
    return write_decision_log(merged, path)
