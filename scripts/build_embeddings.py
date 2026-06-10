"""Phase 7 entry point: grounded context packets + deterministic embeddings.

Pipeline
--------
1. Read ``data/processed/{prices,features}.parquet`` (Phases 2 & 3).
2. Use ``rankings.parquet`` if present; otherwise recompute via the
   Phase 4 strategy pipeline.
3. Load ``performance_records.parquet`` (Phase 6) if present; otherwise
   the performance history degrades to empty — never fabricated.
4. Build one grounded ``ContextPacket`` per top-N candidate at the most
   recent ``as_of`` and persist:
     - ``data/processed/context_packets.jsonl`` (canonical, reproducible)
     - ``data/embeddings/{asset_id}_{YYYYMMDD}.parquet`` per packet

Offline only: no LLM, no network. The module refuses to run if an LLM
SDK has been imported (see ``_assert_no_llm_sdk_loaded``); real
reasoning providers arrive behind the Phase 8 ``ReasoningProvider`` seam.
"""

from __future__ import annotations

import sys

import pandas as pd

from ai_investment_workflow.rag import (
    NotesSource,
    PerformanceSource,
    SignalSource,
    SnapshotSource,
    build_context_packets,
    embed_packet,
    write_context_packets,
    write_embedding_artifact,
)
from ai_investment_workflow.strategies import build_strategy_signals
from ai_investment_workflow.utils import (
    embeddings_dir,
    get_logger,
    load_settings,
    processed_dir,
    setup_logging,
)

#: Module roots that must never be loaded in the pre-LLM retrieval phase.
FORBIDDEN_LLM_SDKS: tuple[str, ...] = (
    "anthropic",
    "openai",
    "cohere",
    "mistralai",
    "groq",
    "litellm",
    "google.generativeai",
)


def _assert_no_llm_sdk_loaded() -> None:
    """Phase 7 is pre-LLM: fail loudly if any LLM SDK is in sys.modules."""
    loaded = sorted(name for name in FORBIDDEN_LLM_SDKS if name in sys.modules)
    assert not loaded, (
        f"Phase 7 is pre-LLM; refusing to run with LLM SDKs loaded: {loaded}"
    )


_assert_no_llm_sdk_loaded()


def _recompute_rankings(features: pd.DataFrame) -> pd.DataFrame:
    """Rankings frame at the latest ``as_of`` via Phase 4 primitives."""
    as_of = features["date"].max()
    signals = build_strategy_signals(features, as_of=as_of)
    rows = [
        {
            "date": as_of,
            "asset_id": sig.asset_id,
            "composite_score": float(sig.composite_score),
            "rank": int(sig.rank),
            "strategy_conflict": bool(sig.strategy_conflict),
        }
        for sig in signals.values()
    ]
    if not rows:
        return pd.DataFrame(
            columns=["date", "asset_id", "composite_score", "rank", "strategy_conflict"]
        )
    return pd.DataFrame(rows).sort_values("rank").reset_index(drop=True)


def main() -> int:
    setup_logging()
    log = get_logger(__name__)
    _assert_no_llm_sdk_loaded()

    features_path = processed_dir() / "features.parquet"
    prices_path = processed_dir() / "prices.parquet"
    if not features_path.is_file():
        log.error(
            "features parquet not found: %s (run scripts/build_features.py first)",
            features_path,
        )
        return 1
    if not prices_path.is_file():
        log.error(
            "prices parquet not found: %s (run scripts/run_ingestion.py first)",
            prices_path,
        )
        return 1

    settings = load_settings()
    top_n = int(settings.get("top_n_candidates", 10))

    prices = pd.read_parquet(prices_path)
    prices["date"] = pd.to_datetime(prices["date"]).dt.date
    features = pd.read_parquet(features_path)
    features["date"] = pd.to_datetime(features["date"]).dt.date

    rankings_path = processed_dir() / "rankings.parquet"
    if rankings_path.is_file():
        rankings = pd.read_parquet(rankings_path)
        rankings["date"] = pd.to_datetime(rankings["date"]).dt.date
        log.info("loaded rankings.parquet (%d rows)", len(rankings))
    else:
        log.info("rankings.parquet absent; recomputing via Phase 4 pipeline")
        rankings = _recompute_rankings(features)

    if rankings.empty:
        log.warning("no rankings available; nothing to retrieve")
        return 0
    as_of = rankings["date"].max()

    records_path = processed_dir() / "performance_records.parquet"
    if records_path.is_file():
        performance_records = pd.read_parquet(records_path)
        log.info(
            "loaded performance_records.parquet (%d rows)", len(performance_records)
        )
    else:
        performance_records = None
        log.info("performance_records.parquet absent; prior history will be empty")

    sources = [
        SnapshotSource(prices),
        SignalSource(features, prices=prices, artifacts_dir=processed_dir()),
        PerformanceSource(performance_records),
        NotesSource(),
    ]

    log.info("building context packets for top %d at %s", top_n, as_of)
    packets = build_context_packets(rankings, as_of=as_of, top_n=top_n, sources=sources)
    if not packets:
        log.warning("no context packets could be grounded; nothing written")
        return 0

    packets_out = write_context_packets(packets, processed_dir() / "context_packets.jsonl")
    log.info("wrote %d context packets to %s", len(packets), packets_out)

    for asset_id in sorted(packets):
        packet = packets[asset_id]
        vector = embed_packet(packet)
        artifact = write_embedding_artifact(packet, vector, embeddings_dir())
        log.info("wrote embedding artifact %s", artifact)

    return 0


if __name__ == "__main__":
    sys.exit(main())
