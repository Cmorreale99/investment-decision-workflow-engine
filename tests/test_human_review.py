"""Phase 9 human-review-layer tests. All offline and non-interactive."""

from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

import pytest

from ai_investment_workflow.human_review import (
    DecisionSource,
    InteractiveDecisionSource,
    ReviewQueue,
    ScriptedDecisionSource,
    append_decisions,
    apply_decision,
    build_decision_source,
    decided_recommendation_ids,
    decision_id_for,
    decision_log_to_jsonl,
    final_status_for,
    load_decision_log,
    load_recommendations,
    load_human_decisions,
    review_recommendations,
    write_decision_log,
)
from ai_investment_workflow.schemas import (
    DecisionRecord,
    HumanAction,
    HumanDecision,
    Recommendation,
    SystemAction,
)

REVIEWED_AT = datetime(2026, 4, 1, 15, 30)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rec(
    asset: str = "AAPL",
    *,
    action: SystemAction = SystemAction.WATCHLIST,
    conviction: float = 0.6,
    ts: date = date(2026, 4, 1),
) -> Recommendation:
    return Recommendation(
        recommendation_id=f"rec_{ts:%Y_%m_%d}_{asset}",
        asset_id=asset,
        timestamp=ts,
        action=action,
        conviction=conviction,
        rationale=["Strong momentum [cf. strategy_signal]"],
        risks=[],
        suggested_next_step="Review manually before approval",
    )


def _decision(
    rec: Recommendation,
    *,
    action: HumanAction = HumanAction.APPROVE,
    override: bool = False,
    notes: str | None = None,
) -> HumanDecision:
    return HumanDecision(
        asset_id=rec.asset_id,
        recommendation_id=rec.recommendation_id,
        human_action=action,
        override=override,
        human_notes=notes,
        reviewed_at=REVIEWED_AT,
    )


# ---------------------------------------------------------------------------
# apply_decision / status derivation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "action,expected",
    [
        (HumanAction.APPROVE, "approved"),
        (HumanAction.REJECT, "rejected"),
        (HumanAction.WATCHLIST, "watchlisted"),
        (HumanAction.OVERRIDE, "overridden"),
        (HumanAction.NEEDS_REVIEW, "needs_review"),
    ],
)
def test_final_status_for_every_action(action, expected) -> None:
    assert final_status_for(action) == expected


def test_decision_id_format() -> None:
    assert decision_id_for(_rec("AAPL")) == "dec_2026_04_01_AAPL"


def test_apply_decision_merges_recommendation_and_human_action() -> None:
    rec = _rec("AAPL", action=SystemAction.BUY, conviction=0.72)
    decision = _decision(
        rec, action=HumanAction.WATCHLIST, override=True, notes="valuation risk"
    )
    record = apply_decision(rec, decision)

    assert isinstance(record, DecisionRecord)
    assert record.decision_id == "dec_2026_04_01_AAPL"
    assert record.recommendation_id == rec.recommendation_id
    assert record.system_action is SystemAction.BUY
    assert record.system_conviction == pytest.approx(0.72)
    assert record.human_action is HumanAction.WATCHLIST
    assert record.override is True
    assert record.human_notes == "valuation risk"
    assert record.final_status == "watchlisted"
    assert record.timestamp == rec.timestamp


def test_apply_decision_does_not_mutate_inputs() -> None:
    rec = _rec("MSFT")
    before = rec.model_copy(deep=True)
    apply_decision(rec, _decision(rec))
    assert rec == before


def test_apply_decision_rejects_recommendation_mismatch() -> None:
    rec = _rec("AAPL")
    other = _decision(_rec("MSFT"))  # different recommendation_id and asset
    with pytest.raises(ValueError, match="recommendation"):
        apply_decision(rec, other)


def test_apply_decision_rejects_asset_mismatch() -> None:
    rec = _rec("AAPL")
    # Same recommendation_id, wrong asset id.
    bad = HumanDecision(
        asset_id="MSFT",
        recommendation_id=rec.recommendation_id,
        human_action=HumanAction.APPROVE,
        reviewed_at=REVIEWED_AT,
    )
    with pytest.raises(ValueError, match="asset"):
        apply_decision(rec, bad)


# ---------------------------------------------------------------------------
# ScriptedDecisionSource
# ---------------------------------------------------------------------------


def test_scripted_source_from_mapping_and_iterable() -> None:
    rec = _rec("AAPL")
    decision = _decision(rec)

    from_map = ScriptedDecisionSource({rec.recommendation_id: decision})
    from_list = ScriptedDecisionSource([decision])
    assert from_map.decide(rec) == decision
    assert from_list.decide(rec) == decision


def test_scripted_source_accepts_dict_values() -> None:
    rec = _rec("AAPL")
    payload = _decision(rec).model_dump(mode="json")
    source = ScriptedDecisionSource({rec.recommendation_id: payload})
    assert source.decide(rec) == _decision(rec)


def test_scripted_source_returns_none_for_unreviewed() -> None:
    source = ScriptedDecisionSource([])
    assert source.decide(_rec("NVDA")) is None


def test_scripted_source_rejects_key_mismatch() -> None:
    rec = _rec("AAPL")
    with pytest.raises(ValueError, match="targets"):
        ScriptedDecisionSource({"rec_other": _decision(rec)})


def test_scripted_source_loads_from_path(tmp_path) -> None:
    rec = _rec("AAPL")
    path = tmp_path / "human_decisions.jsonl"
    path.write_text(_decision(rec).model_dump_json() + "\n", encoding="utf-8")

    assert load_human_decisions(path) == [_decision(rec)]
    source = ScriptedDecisionSource(path=path)
    assert source.decide(rec) == _decision(rec)


def test_scripted_source_missing_path_degrades(tmp_path) -> None:
    source = ScriptedDecisionSource(path=tmp_path / "absent.jsonl")
    assert source.decide(_rec("AAPL")) is None


# ---------------------------------------------------------------------------
# ReviewQueue
# ---------------------------------------------------------------------------


def test_review_queue_pending_filters_decided_and_preserves_order() -> None:
    recs = [_rec("AAPL"), _rec("MSFT"), _rec("NVDA")]
    queue = ReviewQueue(recs)
    assert len(queue) == 3
    assert queue.pending() == recs  # nothing decided yet

    pending = queue.pending({recs[1].recommendation_id})
    assert pending == [recs[0], recs[2]]


def test_review_queue_from_file_roundtrip(tmp_path) -> None:
    recs = [_rec("AAPL"), _rec("MSFT")]
    path = tmp_path / "recommendations.jsonl"
    path.write_text("".join(r.model_dump_json() + "\n" for r in recs), encoding="utf-8")
    assert load_recommendations(path) == recs
    assert ReviewQueue.from_file(path).recommendations == recs


# ---------------------------------------------------------------------------
# Pipeline — mandatory review
# ---------------------------------------------------------------------------


def test_review_records_only_decided_recommendations() -> None:
    reviewed = _rec("AAPL")
    unreviewed = _rec("MSFT")
    source = ScriptedDecisionSource([_decision(reviewed)])

    records = review_recommendations([reviewed, unreviewed], source)

    # Mandatory review: the unreviewed recommendation produces no record.
    assert [r.recommendation_id for r in records] == [reviewed.recommendation_id]
    assert records[0].final_status == "approved"


# ---------------------------------------------------------------------------
# Decision log persistence
# ---------------------------------------------------------------------------


def test_decision_log_write_is_deterministic(tmp_path) -> None:
    records = [
        apply_decision(_rec("NVDA"), _decision(_rec("NVDA"))),
        apply_decision(_rec("AAPL"), _decision(_rec("AAPL"))),
    ]
    a = write_decision_log(records, tmp_path / "a.jsonl")
    b = write_decision_log(list(reversed(records)), tmp_path / "b.jsonl")
    assert a.read_bytes() == b.read_bytes()  # order-independent
    assert a.read_bytes()

    loaded = load_decision_log(a)
    # Sorted by decision_id, so AAPL precedes NVDA regardless of input order.
    assert [r.asset_id for r in loaded] == ["AAPL", "NVDA"]


def test_decision_log_dedupes_by_recommendation_id(tmp_path) -> None:
    rec = _rec("AAPL")
    first = apply_decision(rec, _decision(rec, action=HumanAction.APPROVE))
    second = apply_decision(rec, _decision(rec, action=HumanAction.REJECT))

    jsonl = decision_log_to_jsonl([first, second])
    assert jsonl.count("\n") == 1  # one record, last wins
    path = write_decision_log([first, second], tmp_path / "log.jsonl")
    [record] = load_decision_log(path)
    assert record.final_status == "rejected"


def test_append_decisions_merges_and_is_idempotent(tmp_path) -> None:
    log = tmp_path / "decision_log.jsonl"
    rec_a, rec_b = _rec("AAPL"), _rec("MSFT")

    append_decisions([apply_decision(rec_a, _decision(rec_a))], log)
    append_decisions([apply_decision(rec_b, _decision(rec_b))], log)
    after_two = log.read_bytes()
    assert decided_recommendation_ids(load_decision_log(log)) == {
        rec_a.recommendation_id,
        rec_b.recommendation_id,
    }

    # Re-applying the same decisions yields a byte-identical log.
    append_decisions(
        [apply_decision(rec_a, _decision(rec_a)), apply_decision(rec_b, _decision(rec_b))],
        log,
    )
    assert log.read_bytes() == after_two


# ---------------------------------------------------------------------------
# Source selection + guardrails
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", [None, "", "scripted", "offline", "SCRIPTED"])
def test_build_decision_source_defaults_to_scripted(name) -> None:
    source = build_decision_source(name)
    assert isinstance(source, ScriptedDecisionSource)
    assert isinstance(source, DecisionSource)


def test_build_decision_source_interactive_selectable() -> None:
    # Selectable without being exercised (no stdin touched).
    assert isinstance(build_decision_source("interactive"), InteractiveDecisionSource)


def test_build_decision_source_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown decision source"):
        build_decision_source("api")


def test_human_review_does_not_import_llm_sdk() -> None:
    import ai_investment_workflow.human_review  # noqa: F401

    assert "anthropic" not in sys.modules
