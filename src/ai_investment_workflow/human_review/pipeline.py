"""Review pipeline: pull recommendations through a decision source.

Ties the pieces together: for each pending recommendation, ask the
``DecisionSource`` for a human decision and, when one exists, merge it
into a ``DecisionRecord``. Recommendations the source leaves undecided
(``None``) are skipped — they stay pending, never recorded. Nothing here
executes a trade or marks anything approved on its own.
"""

from __future__ import annotations

from typing import Iterable

from ..schemas import DecisionRecord, Recommendation
from .base import DecisionSource, apply_decision


def review_recommendations(
    recommendations: Iterable[Recommendation],
    source: DecisionSource,
) -> list[DecisionRecord]:
    """Apply the source's decision to each recommendation it has reviewed.

    Returns one ``DecisionRecord`` per recommendation the source decided;
    undecided recommendations (source returns ``None``) are skipped, so a
    record only ever exists for an explicit human action.
    """
    records: list[DecisionRecord] = []
    for recommendation in recommendations:
        decision = source.decide(recommendation)
        if decision is None:
            continue
        records.append(apply_decision(recommendation, decision))
    return records
