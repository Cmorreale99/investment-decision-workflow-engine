"""Decision sources: where a human action for a recommendation comes from.

``ScriptedDecisionSource`` is the deterministic, offline default the test
suite and batch pipeline run on — it looks decisions up by
``recommendation_id`` from an in-memory mapping/iterable or a JSONL file,
and returns ``None`` for anything not yet reviewed (so nothing is recorded
without an explicit human action). ``InteractiveDecisionSource`` prompts a
reviewer on stdin; it is never exercised by the tests, and stdin is only
touched inside it.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Mapping

from ..schemas import HumanAction, HumanDecision, Recommendation
from ..utils.paths import processed_dir

#: Canonical scripted-decisions artifact under ``data/processed/``.
HUMAN_DECISIONS_FILENAME: str = "human_decisions.jsonl"


def default_decisions_path() -> Path:
    return processed_dir() / HUMAN_DECISIONS_FILENAME


def load_human_decisions(path: str | Path) -> list[HumanDecision]:
    """Load human decisions from JSONL; empty list if the file is absent."""
    path = Path(path)
    if not path.is_file():
        return []
    out: list[HumanDecision] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(HumanDecision.model_validate(json.loads(line)))
    return out


def _coerce(value: HumanDecision | Mapping) -> HumanDecision:
    return value if isinstance(value, HumanDecision) else HumanDecision.model_validate(value)


class ScriptedDecisionSource:
    """Deterministic offline source keyed by ``recommendation_id``.

    Provide decisions one of three ways (checked in this order): an
    explicit ``decisions`` mapping (``recommendation_id`` -> HumanDecision
    or dict), an iterable of ``HumanDecision`` (indexed by their
    ``recommendation_id``), or a JSONL ``path`` (defaults to
    ``data/processed/human_decisions.jsonl``). Unknown recommendations
    return ``None`` — they stay pending.
    """

    name = "scripted"

    def __init__(
        self,
        decisions: Mapping[str, HumanDecision | Mapping]
        | Iterable[HumanDecision]
        | None = None,
        *,
        path: str | Path | None = None,
    ) -> None:
        by_id: dict[str, HumanDecision] = {}
        if decisions is None:
            for decision in load_human_decisions(path or default_decisions_path()):
                by_id[decision.recommendation_id] = decision
        elif isinstance(decisions, Mapping):
            for rec_id, value in decisions.items():
                decision = _coerce(value)
                if decision.recommendation_id != rec_id:
                    raise ValueError(
                        f"decision under key {rec_id!r} targets "
                        f"{decision.recommendation_id!r}"
                    )
                by_id[rec_id] = decision
        else:
            for value in decisions:
                decision = _coerce(value)
                by_id[decision.recommendation_id] = decision
        self._by_id = by_id

    def decide(self, recommendation: Recommendation) -> HumanDecision | None:
        return self._by_id.get(recommendation.recommendation_id)


_ACTION_KEYS: dict[str, HumanAction] = {
    "a": HumanAction.APPROVE,
    "r": HumanAction.REJECT,
    "w": HumanAction.WATCHLIST,
    "o": HumanAction.OVERRIDE,
    "n": HumanAction.NEEDS_REVIEW,
}


class InteractiveDecisionSource:
    """Prompts a reviewer on stdin for each recommendation.

    Returns ``None`` when the reviewer skips (blank input), leaving the
    item pending. Not used by the test suite; stdin access lives only here.
    """

    name = "interactive"

    def decide(self, recommendation: Recommendation) -> HumanDecision | None:
        prompt = (
            f"\n{recommendation.asset_id} — system={recommendation.action.value} "
            f"(conviction {recommendation.conviction:.2f})\n"
            "  [a]pprove [r]eject [w]atchlist [o]verride [n]eeds-review "
            "(blank = skip): "
        )
        choice = input(prompt).strip().lower()  # noqa: S322 - intentional stdin
        if not choice:
            return None
        action = _ACTION_KEYS.get(choice[0])
        if action is None:
            return None
        notes = input("  notes (optional): ").strip() or None
        override = action is HumanAction.OVERRIDE or (
            input("  override system action? [y/N]: ").strip().lower().startswith("y")
        )
        return HumanDecision(
            asset_id=recommendation.asset_id,
            recommendation_id=recommendation.recommendation_id,
            human_action=action,
            override=override,
            human_notes=notes,
            reviewed_at=datetime.combine(date.today(), datetime.min.time()),
        )
