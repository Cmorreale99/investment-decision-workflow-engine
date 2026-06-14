"""ReviewQueue: the set of recommendations awaiting a human decision.

The queue is resumable: ``pending`` filters out any recommendation that
already has a decision in the log, so re-running the review only surfaces
items a human has not yet acted on. Order is preserved from the Phase 8
recommendations artifact (rank, then asset id).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ..schemas import Recommendation
from .decision_log import load_recommendations


class ReviewQueue:
    """An ordered set of recommendations, filterable by what's decided."""

    def __init__(self, recommendations: Iterable[Recommendation]) -> None:
        self._recommendations = list(recommendations)

    @classmethod
    def from_file(cls, path: str | Path) -> "ReviewQueue":
        return cls(load_recommendations(path))

    def __len__(self) -> int:
        return len(self._recommendations)

    @property
    def recommendations(self) -> list[Recommendation]:
        return list(self._recommendations)

    def pending(self, decided_ids: Iterable[str] = ()) -> list[Recommendation]:
        """Recommendations not yet decided, in queue order."""
        decided = set(decided_ids)
        return [
            rec
            for rec in self._recommendations
            if rec.recommendation_id not in decided
        ]
