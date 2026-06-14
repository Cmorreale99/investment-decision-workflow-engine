"""Phase 11 entry point: launch the Streamlit review dashboard.

Streamlit is an OPTIONAL dependency. This launcher checks for it without
importing it (or the dashboard module) directly, prints install guidance
if it is missing, and otherwise runs ``streamlit run`` on the dashboard.
The dashboard itself is read-only over upstream artifacts and records
human decisions through the human review layer — no trade execution.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

from ai_investment_workflow.utils import get_logger, setup_logging

#: The Streamlit shell module file to run.
_DASHBOARD = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "ai_investment_workflow"
    / "app"
    / "dashboard.py"
)


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    log = get_logger(__name__)

    if importlib.util.find_spec("streamlit") is None:
        log.error(
            "streamlit is not installed (it is optional). Install it with "
            "`pip install streamlit` or `pip install -e .[dashboard]`."
        )
        return 1

    cmd = [sys.executable, "-m", "streamlit", "run", str(_DASHBOARD), *(argv or [])]
    log.info("launching dashboard: %s", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
