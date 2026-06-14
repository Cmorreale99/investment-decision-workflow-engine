"""Application layer: the read-only dashboard (Phase 11).

Presentation only — no strategy, AI, retrieval, or evaluation logic. It
loads the artifacts prior phases wrote and presents them; the only state
it changes is recording human review actions, which it delegates to the
human_review layer (``app.actions``).

This package deliberately does **not** import ``app.dashboard`` (the
Streamlit shell), so importing ``ai_investment_workflow.app`` never pulls
in Streamlit. The pure data/view/action helpers below are offline and
fully testable; ``app.dashboard`` is loaded only by ``streamlit run`` or
``scripts/run_dashboard.py``.
"""

from __future__ import annotations

from .actions import record_decision
from .data import (
    DashboardData,
    latest_decision_by_recommendation,
    load_context_packets,
    load_decision_diagnostics,
    load_portfolio_diagnostics,
)
from .views import (
    CandidateCard,
    EvaluationSummary,
    build_candidate_cards,
    build_evaluation_summary,
)

__all__ = [
    "CandidateCard",
    "DashboardData",
    "EvaluationSummary",
    "build_candidate_cards",
    "build_evaluation_summary",
    "latest_decision_by_recommendation",
    "load_context_packets",
    "load_decision_diagnostics",
    "load_portfolio_diagnostics",
    "record_decision",
]
