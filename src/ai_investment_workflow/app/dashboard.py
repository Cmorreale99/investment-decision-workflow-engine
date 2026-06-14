"""Streamlit shell for the human-in-the-loop review dashboard.

This is the only module that depends on Streamlit, and it is loaded only
by ``streamlit run`` / ``scripts/run_dashboard.py`` — never imported by
the ``app`` package. It contains no business logic: it renders the view
models from ``app.views`` and records actions through ``app.actions``
(which routes through the human review layer). Read-only over every
upstream artifact; no trade execution.
"""

from __future__ import annotations

import streamlit as st

from ..schemas import HumanAction
from ..utils.paths import processed_dir
from .actions import record_decision
from .data import DashboardData
from .views import (
    CandidateCard,
    EvaluationSummary,
    build_candidate_cards,
    build_evaluation_summary,
)


def _render_metrics(summary: EvaluationSummary) -> None:
    st.subheader("Evaluation metrics")
    cols = st.columns(4)
    cols[0].metric("Decisions", summary.total_decisions if summary.total_decisions is not None else "—")
    cols[1].metric(
        "Approval rate",
        f"{summary.approval_rate:.0%}" if summary.approval_rate is not None else "—",
    )
    cols[2].metric(
        "Override rate",
        f"{summary.override_rate:.0%}" if summary.override_rate is not None else "—",
    )
    cols[3].metric(
        "Decision hit rate",
        f"{summary.decision_hit_rate:.0%}" if summary.decision_hit_rate is not None else "—",
    )
    if summary.approved_minus_rejected_excess is not None:
        st.caption(
            f"Approved-minus-rejected excess return: "
            f"{summary.approved_minus_rejected_excess:+.4f}"
        )


def _render_card(card: CandidateCard, recommendations: dict) -> None:
    decided = f" — human: {card.human_action}" if card.human_action else " — pending review"
    header = f"{card.asset_id}  ·  {card.action} ({card.conviction:.2f}){decided}"
    with st.expander(header):
        cols = st.columns(3)
        cols[0].write(f"**Sector:** {card.sector or '—'}")
        cols[1].write(
            f"**Composite:** {card.composite_score:+.2f}"
            if card.composite_score is not None
            else "**Composite:** —"
        )
        cols[2].write(f"**Rank:** {card.rank if card.rank is not None else '—'}")
        if card.strategy_scores:
            st.write(
                "**Strategy scores:** "
                + ", ".join(f"{k} {v:+.2f}" for k, v in sorted(card.strategy_scores.items()))
            )
        if card.strategy_conflict:
            st.warning("Strategy conflict")

        if card.reasons:
            st.write("**Top reasons**")
            for reason in card.reasons:
                st.write(f"- {reason}")
        if card.risks:
            st.write("**Top risks**")
            for risk in card.risks:
                st.write(f"- {risk}")

        if card.human_action:
            st.info(
                f"Recorded: {card.human_action} → {card.final_status}"
                + (f" — {card.human_notes}" if card.human_notes else "")
            )

        recommendation = recommendations.get(card.recommendation_id)
        if recommendation is None:
            return
        with st.form(f"decision_{card.recommendation_id}"):
            action = st.selectbox("Action", [a.value for a in HumanAction])
            override = st.checkbox("Override system action")
            notes = st.text_input("Notes")
            if st.form_submit_button("Record decision"):
                record_decision(
                    recommendation,
                    HumanAction(action),
                    override=override,
                    notes=notes or None,
                )
                st.success(f"Recorded {action} for {card.asset_id}")
                st.rerun()


def main() -> None:
    st.set_page_config(page_title="Investment Workflow", layout="wide")
    st.title("AI Investment Workflow — Review Dashboard")
    st.caption("Human review is required. The dashboard records decisions; it never executes trades.")

    data = DashboardData.load(processed_dir())
    if not data.recommendations:
        st.info(
            "No recommendations found. Run scripts/generate_recommendations.py first."
        )
        return

    _render_metrics(build_evaluation_summary(data))

    st.subheader("Ranked candidates")
    recommendations = {r.recommendation_id: r for r in data.recommendations}
    for card in build_candidate_cards(data):
        _render_card(card, recommendations)

    if data.decisions:
        st.subheader("Decision history")
        st.dataframe(
            [
                {
                    "asset_id": d.asset_id,
                    "system_action": d.system_action.value,
                    "human_action": d.human_action.value,
                    "override": d.override,
                    "final_status": d.final_status,
                }
                for d in data.decisions
            ]
        )


if __name__ == "__main__":
    main()
