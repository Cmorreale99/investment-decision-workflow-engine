"""Phase 10 entry point: decision & override evaluation diagnostics.

Pipeline
--------
1. Load the Phase 9 decision log (``data/processed/decision_log.jsonl``).
2. Load the Phase 6 performance records
   (``data/processed/performance_records.parquet``).
3. Join on ``decision_id`` and compute decision-level diagnostics:
   approval/override rates, hit rate, approved-vs-rejected excess return,
   override impact, and performance by conviction tier.
4. Write ``data/processed/decision_diagnostics.json`` (deterministic).

This script diagnoses outcomes only — it generates no recommendations,
makes no decisions, mutates no inputs, and executes no trades. If either
input is missing it logs and exits cleanly (no fabricated outcomes).
"""

from __future__ import annotations

import sys

import pandas as pd

from ai_investment_workflow.evaluation import (
    diagnose_decisions,
    diagnostics_to_json,
    performance_records_from_frame,
)
from ai_investment_workflow.human_review import (
    DECISION_LOG_FILENAME,
    load_decision_log,
)
from ai_investment_workflow.utils import (
    get_logger,
    processed_dir,
    setup_logging,
)

DECISION_DIAGNOSTICS_FILENAME: str = "decision_diagnostics.json"
PERFORMANCE_RECORDS_FILENAME: str = "performance_records.parquet"


def main() -> int:
    setup_logging()
    log = get_logger(__name__)

    log_path = processed_dir() / DECISION_LOG_FILENAME
    if not log_path.is_file():
        log.error(
            "decision log not found: %s (run scripts/run_human_review.py first)",
            log_path,
        )
        return 1

    decisions = load_decision_log(log_path)
    if not decisions:
        log.warning("decision log is empty; nothing to diagnose")
        return 0

    records_path = processed_dir() / PERFORMANCE_RECORDS_FILENAME
    if records_path.is_file():
        frame = pd.read_parquet(records_path)
        performance_records = performance_records_from_frame(frame)
        log.info("loaded %d performance record(s)", len(performance_records))
    else:
        performance_records = []
        log.info(
            "%s absent; outcome metrics will be empty (decisions are pending)",
            records_path,
        )

    diagnostics = diagnose_decisions(decisions, performance_records)

    out_path = processed_dir() / DECISION_DIAGNOSTICS_FILENAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(diagnostics_to_json(diagnostics) + "\n", encoding="utf-8")

    log.info(
        "decisions=%d outcomes=%d pending=%d | approval_rate=%.2f override_rate=%.2f",
        diagnostics.total_decisions,
        diagnostics.decisions_with_outcomes,
        diagnostics.pending_outcomes,
        diagnostics.approval_rate,
        diagnostics.override_rate,
    )
    if diagnostics.approved_minus_rejected_excess is not None:
        log.info(
            "approved-minus-rejected excess return: %+.4f",
            diagnostics.approved_minus_rejected_excess,
        )
    log.info("wrote %s (diagnoses outcomes only; no trades, no recommendations)", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
