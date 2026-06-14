"""Phase 9 entry point: route recommendations through human review.

Pipeline
--------
1. Load the Phase 8 recommendations (``data/processed/recommendations.jsonl``).
2. Load any existing decision log and skip recommendations already decided
   (the queue is resumable).
3. Pull each pending recommendation through a ``DecisionSource``. The
   default is the deterministic offline ``ScriptedDecisionSource``, which
   reads human actions from ``data/processed/human_decisions.jsonl``; pass
   ``--interactive`` to review on stdin instead.
4. Merge each human decision with its recommendation into a schema-valid
   ``DecisionRecord`` and append it to ``data/processed/decision_log.jsonl``.

Human review is mandatory: a recommendation with no human decision stays
pending and is never recorded as final. This script never executes a trade.
"""

from __future__ import annotations

import argparse
import sys

from ai_investment_workflow.human_review import (
    DECISION_LOG_FILENAME,
    RECOMMENDATIONS_FILENAME,
    ReviewQueue,
    append_decisions,
    build_decision_source,
    decided_recommendation_ids,
    load_decision_log,
    review_recommendations,
)
from ai_investment_workflow.utils import (
    get_logger,
    load_settings,
    processed_dir,
    setup_logging,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the human review layer.")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Review on stdin instead of the offline scripted source.",
    )
    args = parser.parse_args(argv)

    setup_logging()
    log = get_logger(__name__)
    settings = load_settings()
    if not settings.get("require_human_approval", True):
        log.info("require_human_approval is false; review still records explicit decisions")

    recs_path = processed_dir() / RECOMMENDATIONS_FILENAME
    if not recs_path.is_file():
        log.error(
            "recommendations not found: %s (run scripts/generate_recommendations.py first)",
            recs_path,
        )
        return 1

    queue = ReviewQueue.from_file(recs_path)
    log_path = processed_dir() / DECISION_LOG_FILENAME
    existing = load_decision_log(log_path)
    pending = queue.pending(decided_recommendation_ids(existing))
    log.info(
        "%d recommendation(s); %d already decided; %d pending review",
        len(queue),
        len(existing),
        len(pending),
    )
    if not pending:
        log.info("nothing pending; decision log is up to date")
        return 0

    source = build_decision_source("interactive" if args.interactive else None)
    log.info("decision source: %s", source.name)

    new_records = review_recommendations(pending, source)
    if not new_records:
        log.info("no human decisions provided; %d item(s) remain pending", len(pending))
        return 0

    out_path = append_decisions(new_records, log_path)
    log.info("recorded %d decision(s) to %s (no trades executed)", len(new_records), out_path)
    for record in new_records:
        log.info(
            "  %s: system=%s -> human=%s (final_status=%s%s)",
            record.asset_id,
            record.system_action.value,
            record.human_action.value,
            record.final_status,
            ", override" if record.override else "",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
