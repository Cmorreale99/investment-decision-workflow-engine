"""Phase 4 entry point: score and rank candidates.

Phase 1 stub. Imports the package, logs intent, and exits cleanly.
"""

from __future__ import annotations

import sys

from ai_investment_workflow.utils import get_logger, setup_logging


def main() -> int:
    setup_logging()
    log = get_logger(__name__)
    log.info("run_strategies: not implemented (Phase 4).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
