"""Phase 7 retrieval-layer tests."""

from __future__ import annotations

import ast
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ai_investment_workflow.features import compute_feature_frame
from ai_investment_workflow.ingestion import FixtureProvider
from ai_investment_workflow.rag import (
    DEFAULT_DIM,
    NotesSource,
    PerformanceSource,
    SignalSource,
    SnapshotSource,
    VectorStore,
    build_context_packet,
    build_context_packets,
    canonical_json,
    embed_packet,
    embed_text,
    embedding_artifact_path,
    merge_source_payloads,
    packets_to_jsonl,
    select_top_candidates,
    stable_hash,
    write_context_packets,
    write_embedding_artifact,
)
from ai_investment_workflow.risk import evaluate_risk
from ai_investment_workflow.schemas import (
    CompanySnapshot,
    ContextPacket,
    OutcomeLabel,
    PerformanceRecord,
    StrategySignal,
)
from ai_investment_workflow.strategies import build_strategy_signals

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "SPY"]


@pytest.fixture(scope="module")
def fixture_data():
    """Prices, features, signals, and a rankings frame at the latest date."""
    provider = FixtureProvider()
    prices = provider.fetch(TICKERS, date(2025, 1, 1), date(2025, 6, 1))
    features = compute_feature_frame(prices)
    as_of = features["date"].max()
    signals = build_strategy_signals(features, as_of=as_of)
    rankings = pd.DataFrame(
        [
            {
                "date": as_of,
                "asset_id": sig.asset_id,
                "composite_score": float(sig.composite_score),
                "rank": int(sig.rank),
                "strategy_conflict": bool(sig.strategy_conflict),
            }
            for sig in signals.values()
        ]
    ).sort_values("rank").reset_index(drop=True)
    return prices, features, rankings, signals, as_of


@pytest.fixture(scope="module")
def default_sources(fixture_data):
    prices, features, _rankings, _signals, _as_of = fixture_data
    return [
        SnapshotSource(prices),
        SignalSource(features, prices=prices),
        PerformanceSource(None),
        NotesSource(Path("nonexistent_notes.jsonl")),
    ]


def _sample_records(asset_id: str = "AAPL", n: int = 7) -> list[PerformanceRecord]:
    """``n`` weekly completed records ending well before 2025-04-30."""
    records: list[PerformanceRecord] = []
    base = date(2025, 1, 6)
    for i in range(n):
        start = base + timedelta(days=7 * i)
        excess = 0.01 * i - 0.02
        records.append(
            PerformanceRecord(
                decision_id=f"dec_{start.strftime('%Y%m%d')}_{asset_id}",
                asset_id=asset_id,
                evaluation_start=start,
                evaluation_end=start + timedelta(days=14),
                asset_return=0.01 * i,
                benchmark_return=0.02,
                excess_return=excess,
                max_drawdown=-0.03,
                outcome_label=(
                    OutcomeLabel.OUTPERFORMED if excess > 0 else OutcomeLabel.UNDERPERFORMED
                ),
            )
        )
    return records


# ---------------------------------------------------------------------------
# SnapshotSource
# ---------------------------------------------------------------------------


def test_snapshot_source_fetch_contract(fixture_data) -> None:
    prices, _features, _rankings, _signals, as_of = fixture_data
    source = SnapshotSource(prices, sectors={"AAPL": "Technology"})
    payload = source.fetch("AAPL", as_of)

    assert set(payload) == {"company_snapshot"}
    snapshot = payload["company_snapshot"]
    assert isinstance(snapshot, CompanySnapshot)
    assert snapshot.asset_id == "AAPL"
    assert snapshot.timestamp == as_of
    assert snapshot.sector == "Technology"
    expected_close = prices.loc[
        (prices["asset_id"] == "AAPL") & (prices["date"] == as_of), "close"
    ].iloc[0]
    assert snapshot.price == pytest.approx(float(expected_close))


def test_snapshot_source_missing_asset_returns_empty(fixture_data) -> None:
    prices, _features, _rankings, _signals, as_of = fixture_data
    source = SnapshotSource(prices)
    assert source.fetch("ZZZZ", as_of) == {}


# ---------------------------------------------------------------------------
# SignalSource
# ---------------------------------------------------------------------------


def test_signal_source_recompute_matches_phase4(fixture_data) -> None:
    _prices, features, _rankings, signals, as_of = fixture_data
    source = SignalSource(features)
    asset_id = next(iter(signals))
    payload = source.fetch(asset_id, as_of)

    assert isinstance(payload["strategy_signal"], StrategySignal)
    assert payload["strategy_signal"] == signals[asset_id]
    assert "risk_flags" not in payload  # no prices, no artifacts -> no flags


def test_signal_source_recomputes_risk_flags_with_prices(fixture_data) -> None:
    prices, features, _rankings, signals, as_of = fixture_data
    source = SignalSource(features, prices=prices)
    asset_id = next(iter(signals))
    payload = source.fetch(asset_id, as_of)

    flags = payload["risk_flags"]
    assert flags and all(isinstance(v, bool) for v in flags.values())

    snapshot = SnapshotSource(prices).fetch(asset_id, as_of)["company_snapshot"]
    feature_row = features.loc[
        (features["date"] == as_of) & (features["asset_id"] == asset_id)
    ].iloc[0]
    feats = {
        k: float(v)
        for k, v in feature_row.items()
        if k not in ("date", "asset_id") and pd.notna(v)
    }
    assert flags == evaluate_risk(snapshot, feats)


def test_signal_source_loads_from_artifacts(fixture_data, tmp_path) -> None:
    _prices, features, rankings, signals, as_of = fixture_data
    score_rows = [
        {"date": as_of, "asset_id": aid, "strategy": strat, "score": float(s)}
        for aid, sig in signals.items()
        for strat, s in sig.strategy_scores.items()
    ]
    pd.DataFrame(score_rows).to_parquet(tmp_path / "signals.parquet", index=False)
    rankings.to_parquet(tmp_path / "rankings.parquet", index=False)
    flag_rows = [
        {"date": as_of, "asset_id": "AAPL", "risk_flag": "high_volatility", "value": True}
    ]
    pd.DataFrame(flag_rows).to_parquet(tmp_path / "risk_flags.parquet", index=False)

    source = SignalSource(features, artifacts_dir=tmp_path)
    payload = source.fetch("AAPL", as_of)
    assert payload["strategy_signal"] == signals["AAPL"]
    assert payload["risk_flags"] == {"high_volatility": True}


def test_signal_source_missing_asset_returns_empty(fixture_data) -> None:
    _prices, features, _rankings, _signals, as_of = fixture_data
    assert SignalSource(features).fetch("ZZZZ", as_of) == {}


# ---------------------------------------------------------------------------
# PerformanceSource
# ---------------------------------------------------------------------------


def test_performance_source_returns_most_recent_n(fixture_data) -> None:
    *_rest, as_of = fixture_data
    records = _sample_records(n=7)
    payload = PerformanceSource(records).fetch("AAPL", as_of)

    assert len(payload["prior_decisions"]) == 5  # default N=5
    expected_ids = [r.decision_id for r in sorted(records, key=lambda r: r.evaluation_end, reverse=True)][:5]
    assert payload["prior_decisions"] == expected_ids

    summary = payload["recent_performance"]
    assert summary["n_records"] == 5.0
    assert all(isinstance(v, float) for v in summary.values())
    assert summary["last_excess_return"] == pytest.approx(0.01 * 6 - 0.02)


def test_performance_source_excludes_pending_and_future(fixture_data) -> None:
    *_rest, as_of = fixture_data
    completed = _sample_records(n=2)
    pending = PerformanceRecord(
        decision_id="dec_pending_AAPL",
        asset_id="AAPL",
        evaluation_start=date(2025, 2, 1),
        evaluation_end=date(2025, 2, 15),
        asset_return=0.0,
        benchmark_return=0.0,
        excess_return=0.0,
        max_drawdown=0.0,
        outcome_label=OutcomeLabel.PENDING,
    )
    future = PerformanceRecord(
        decision_id="dec_future_AAPL",
        asset_id="AAPL",
        evaluation_start=as_of,
        evaluation_end=as_of + timedelta(days=30),
        asset_return=0.5,
        benchmark_return=0.0,
        excess_return=0.5,
        max_drawdown=-0.01,
        outcome_label=OutcomeLabel.OUTPERFORMED,
    )
    payload = PerformanceSource(completed + [pending, future]).fetch("AAPL", as_of)
    assert payload["prior_decisions"] == [
        completed[1].decision_id,
        completed[0].decision_id,
    ]


def test_performance_source_empty_history_degrades(fixture_data) -> None:
    *_rest, as_of = fixture_data
    for source in (PerformanceSource(None), PerformanceSource(_sample_records(n=3))):
        payload = source.fetch("MSFT", as_of)  # no MSFT records exist
        assert payload == {"recent_performance": {}, "prior_decisions": []}


def test_performance_source_accepts_dataframe(fixture_data) -> None:
    *_rest, as_of = fixture_data
    records = _sample_records(n=3)
    frame = pd.DataFrame(
        [
            {
                "decision_id": r.decision_id,
                "asset_id": r.asset_id,
                "evaluation_start": r.evaluation_start,
                "evaluation_end": r.evaluation_end,
                "asset_return": r.asset_return,
                "benchmark_return": r.benchmark_return,
                "excess_return": r.excess_return,
                "max_drawdown": r.max_drawdown,
                "outcome_label": r.outcome_label.value,
            }
            for r in records
        ]
    )
    assert PerformanceSource(frame).fetch("AAPL", as_of) == PerformanceSource(
        records
    ).fetch("AAPL", as_of)


# ---------------------------------------------------------------------------
# NotesSource
# ---------------------------------------------------------------------------


def test_notes_source_missing_file_returns_empty(tmp_path) -> None:
    source = NotesSource(tmp_path / "human_notes.jsonl")
    assert source.fetch("AAPL", date(2025, 4, 30)) == {"human_notes": []}


def test_notes_source_filters_and_skips_malformed(tmp_path) -> None:
    path = tmp_path / "human_notes.jsonl"
    lines = [
        json.dumps({"asset_id": "AAPL", "timestamp": "2025-03-01", "note": "early note"}),
        json.dumps({"asset_id": "AAPL", "timestamp": "2025-05-01", "note": "future note"}),
        json.dumps({"asset_id": "MSFT", "timestamp": "2025-03-01", "note": "other asset"}),
        json.dumps({"asset_id": "AAPL", "note": "undated note"}),
        "{not valid json",
        json.dumps({"asset_id": "AAPL", "timestamp": "2025-03-02"}),  # missing note
    ]
    path.write_text("\n".join(lines), encoding="utf-8")

    payload = NotesSource(path).fetch("AAPL", date(2025, 4, 30))
    assert payload == {"human_notes": ["undated note", "early note"]}


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def test_stable_hash_is_deterministic_and_distinct() -> None:
    assert stable_hash("AAPL") == stable_hash("AAPL")
    assert stable_hash("AAPL") != stable_hash("MSFT")
    assert stable_hash("AAPL") >= 0


def test_embed_text_deterministic_unit_vector() -> None:
    a = embed_text("hello", dim=32)
    b = embed_text("hello", dim=32)
    c = embed_text("world", dim=32)
    assert a.shape == (32,)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    assert np.linalg.norm(a) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        embed_text("x", dim=0)


def test_embed_packet_matches_canonical_text(fixture_data, default_sources) -> None:
    *_rest, as_of = fixture_data
    packet = build_context_packet("AAPL", as_of, default_sources)
    assert packet is not None
    np.testing.assert_array_equal(
        embed_packet(packet), embed_text(canonical_json(packet), dim=DEFAULT_DIM)
    )


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------


def test_vector_store_put_query_roundtrip() -> None:
    store = VectorStore(dim=16)
    for key in ("AAPL", "MSFT", "NVDA"):
        store.put(key, embed_text(key, dim=16), {"asset_id": key})

    results = store.query(embed_text("MSFT", dim=16), top_k=2)
    assert len(results) == 2
    top_key, top_score, top_meta = results[0]
    assert top_key == "MSFT"
    assert top_score == pytest.approx(1.0)
    assert top_meta == {"asset_id": "MSFT"}
    assert results[0][1] >= results[1][1]


def test_vector_store_query_is_deterministic() -> None:
    a = VectorStore(dim=8)
    b = VectorStore(dim=8)
    keys = ["C", "A", "B"]
    for key in keys:
        a.put(key, embed_text(key, dim=8))
    for key in reversed(keys):
        b.put(key, embed_text(key, dim=8))
    q = embed_text("query", dim=8)
    assert a.query(q, top_k=3) == b.query(q, top_k=3)


def test_vector_store_save_load_roundtrip(tmp_path) -> None:
    store = VectorStore(dim=16)
    store.put("AAPL", embed_text("AAPL", dim=16), {"rank": 1})
    store.put("MSFT", embed_text("MSFT", dim=16), {"rank": 2})
    path = store.save(tmp_path / "vector_store.parquet")

    loaded = VectorStore.load(path)
    assert loaded.dim == 16
    assert loaded.keys() == store.keys()
    for key in store.keys():
        orig_vec, orig_meta = store.get(key)
        new_vec, new_meta = loaded.get(key)
        np.testing.assert_array_equal(orig_vec, new_vec)
        assert orig_meta == new_meta


def test_vector_store_rejects_wrong_dim() -> None:
    store = VectorStore(dim=8)
    with pytest.raises(ValueError):
        store.put("AAPL", np.zeros(4))
    with pytest.raises(ValueError):
        store.query(np.zeros(4))


# ---------------------------------------------------------------------------
# ContextPacket construction
# ---------------------------------------------------------------------------


def test_build_context_packet_grounded_fields(fixture_data) -> None:
    prices, features, _rankings, signals, as_of = fixture_data
    notes_path = Path("nonexistent_notes.jsonl")
    sources = [
        SnapshotSource(prices),
        SignalSource(features, prices=prices),
        PerformanceSource(_sample_records(n=7)),
        NotesSource(notes_path),
    ]
    packet = build_context_packet("AAPL", as_of, sources)

    assert packet is not None
    assert packet.asset_id == "AAPL"
    assert packet.timestamp == as_of
    assert packet.strategy_signal == signals["AAPL"]
    assert packet.prior_decisions  # AAPL history exists
    assert packet.recent_performance["n_records"] == 5.0
    assert packet.risk_flags  # recomputed from prices+features
    assert packet.human_notes == []
    assert packet.research_notes == []  # no source -> schema default


def test_build_context_packet_returns_none_without_required_fields(
    fixture_data,
) -> None:
    _prices, features, _rankings, _signals, as_of = fixture_data
    # Signal only — no snapshot source, so the packet cannot be grounded.
    assert build_context_packet("AAPL", as_of, [SignalSource(features)]) is None
    assert build_context_packet("ZZZZ", as_of, []) is None


def test_build_context_packet_rejects_identity_mismatch(fixture_data) -> None:
    prices, features, _rankings, _signals, as_of = fixture_data

    class WrongAssetSource:
        name = "wrong_asset"

        def fetch(self, asset_id: str, as_of_: date) -> dict:
            return SnapshotSource(prices).fetch("MSFT", as_of_)

    sources = [WrongAssetSource(), SignalSource(features)]
    with pytest.raises(ValueError, match="asset"):
        build_context_packet("AAPL", as_of, sources)


def test_merge_rejects_duplicate_and_unknown_fields(fixture_data) -> None:
    prices, _features, _rankings, _signals, as_of = fixture_data

    class UnknownFieldSource:
        name = "unknown_field"

        def fetch(self, asset_id: str, as_of_: date) -> dict:
            return {"made_up_field": 1}

    with pytest.raises(ValueError, match="non-ContextPacket"):
        merge_source_payloads([UnknownFieldSource()], "AAPL", as_of)

    dup = [SnapshotSource(prices), SnapshotSource(prices)]
    with pytest.raises(ValueError, match="company_snapshot"):
        merge_source_payloads(dup, "AAPL", as_of)


def test_context_packet_json_roundtrip(fixture_data, default_sources) -> None:
    *_rest, as_of = fixture_data
    packet = build_context_packet("AAPL", as_of, default_sources)
    assert packet is not None
    restored = ContextPacket.model_validate(json.loads(canonical_json(packet)))
    assert restored == packet


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def test_select_top_candidates_orders_by_rank(fixture_data) -> None:
    _prices, _features, rankings, _signals, as_of = fixture_data
    top3 = select_top_candidates(rankings, as_of, top_n=3)
    expected = rankings.sort_values(["rank", "asset_id"]).head(3)["asset_id"].tolist()
    assert top3 == expected
    assert select_top_candidates(rankings, as_of, top_n=0) == []


def test_build_context_packets_top_n(fixture_data, default_sources) -> None:
    _prices, _features, rankings, _signals, as_of = fixture_data
    packets = build_context_packets(
        rankings, as_of=as_of, top_n=3, sources=default_sources
    )
    assert len(packets) == 3
    assert set(packets) == set(select_top_candidates(rankings, as_of, top_n=3))
    for asset_id, packet in packets.items():
        assert packet.asset_id == asset_id
        assert packet.timestamp == as_of


def test_pipeline_reproducible_byte_identical_jsonl(
    fixture_data, default_sources, tmp_path
) -> None:
    _prices, _features, rankings, _signals, as_of = fixture_data
    kwargs = {"as_of": as_of, "top_n": 4, "sources": default_sources}
    first = build_context_packets(rankings, **kwargs)
    second = build_context_packets(rankings, **kwargs)

    assert packets_to_jsonl(first) == packets_to_jsonl(second)

    path_a = write_context_packets(first, tmp_path / "a.jsonl")
    path_b = write_context_packets(second, tmp_path / "b.jsonl")
    assert path_a.read_bytes() == path_b.read_bytes()
    assert path_a.read_bytes()  # non-empty


def test_pipeline_skips_ungroundable_assets(fixture_data) -> None:
    prices, features, rankings, _signals, as_of = fixture_data
    # Drop AAPL prices so its snapshot cannot be grounded at as_of.
    pruned = prices.loc[prices["asset_id"] != "AAPL"]
    sources = [SnapshotSource(pruned), SignalSource(features)]
    packets = build_context_packets(
        rankings, as_of=as_of, top_n=len(rankings), sources=sources
    )
    assert "AAPL" not in packets
    assert packets  # other assets still built


def test_write_embedding_artifact(fixture_data, default_sources, tmp_path) -> None:
    *_rest, as_of = fixture_data
    packet = build_context_packet("AAPL", as_of, default_sources)
    assert packet is not None
    vector = embed_packet(packet)
    path = write_embedding_artifact(packet, vector, tmp_path)

    assert path == embedding_artifact_path(packet, tmp_path)
    assert path.name == f"AAPL_{as_of.strftime('%Y%m%d')}.parquet"
    frame = pd.read_parquet(path)
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["asset_id"] == "AAPL"
    assert row["as_of"] == as_of.isoformat()
    assert int(row["dim"]) == DEFAULT_DIM
    np.testing.assert_array_almost_equal(np.asarray(row["vector"]), vector)
    assert int(row["rank"]) == packet.strategy_signal.rank


# ---------------------------------------------------------------------------
# Guardrails: no LLM SDKs anywhere in rag/ or the Phase 7 script
# ---------------------------------------------------------------------------

FORBIDDEN_IMPORT_ROOTS = {
    "anthropic",
    "openai",
    "cohere",
    "mistralai",
    "groq",
    "litellm",
    "langchain",
    "google",
    "requests",
    "httpx",
    "aiohttp",
    "urllib3",
}


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def test_no_llm_or_network_imports_in_rag() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    rag_dir = repo_root / "src" / "ai_investment_workflow" / "rag"
    targets = sorted(rag_dir.rglob("*.py")) + [
        repo_root / "scripts" / "build_embeddings.py"
    ]
    assert targets, "rag/ sources not found"
    for path in targets:
        offending = _imported_roots(path) & FORBIDDEN_IMPORT_ROOTS
        assert not offending, f"{path} imports forbidden modules: {sorted(offending)}"


def test_build_embeddings_script_asserts_offline() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "build_embeddings.py"
    tree = ast.parse(script.read_text(encoding="utf-8"))
    guard_calls = [
        node
        for node in tree.body  # module scope only
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "_assert_no_llm_sdk_loaded"
    ]
    assert guard_calls, "script must call _assert_no_llm_sdk_loaded() at module scope"
