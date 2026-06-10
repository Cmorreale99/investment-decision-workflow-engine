"""Human-notes source (placeholder; read-only in Phase 7).

Reads ``data/processed/human_notes.jsonl`` if present, otherwise yields
an empty list. There is **no write path** here — notes are authored by
the human review layer (Phase 8+).

Expected line format (one JSON object per line)::

    {"asset_id": "AAPL", "timestamp": "2026-04-01", "note": "..."}

Rules:

- lines that are not valid JSON, or lack ``asset_id`` / ``note``, are
  skipped silently (graceful degradation, no fabrication);
- ``timestamp`` is optional; dated notes are only surfaced when
  ``timestamp <= as_of`` (point-in-time discipline), undated notes are
  always surfaced;
- output ordering is deterministic: ``(timestamp, note)`` ascending,
  undated notes first.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from ...utils.paths import processed_dir

#: Canonical notes file under ``data/processed/``.
NOTES_FILENAME: str = "human_notes.jsonl"


def default_notes_path() -> Path:
    return processed_dir() / NOTES_FILENAME


class NotesSource:
    """Surfaces prior human notes for an asset; empty when none exist."""

    name = "notes_source"

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path else default_notes_path()

    def fetch(self, asset_id: str, as_of: date) -> dict:
        if not self._path.is_file():
            return {"human_notes": []}

        dated: list[tuple[date, str]] = []
        undated: list[str] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict) or obj.get("asset_id") != asset_id:
                continue
            note = obj.get("note")
            if not isinstance(note, str) or not note:
                continue
            raw_ts = obj.get("timestamp")
            if raw_ts is None:
                undated.append(note)
                continue
            try:
                ts = date.fromisoformat(str(raw_ts))
            except ValueError:
                continue
            if ts <= as_of:
                dated.append((ts, note))

        notes = sorted(undated) + [n for _, n in sorted(dated)]
        return {"human_notes": notes}
