"""Phase 8 entry point: agent-generated structured recommendations.

Pipeline
--------
1. Load grounded context packets. Prefer the canonical Phase 7 artifact
   ``data/processed/context_packets.jsonl``; if it is absent, rebuild them
   in-memory via ``rag.build_context_packets`` from the Phase 2-6 artifacts.
2. Select a reasoning provider from ``settings.reasoning_provider``
   (default: the deterministic offline ``StubProvider``; ``"anthropic"``
   opts into the SDK-backed provider, imported lazily).
3. Run the Analyst / Strategy / Risk agents through the orchestrator to
   produce exactly one schema-validated ``Recommendation`` per candidate,
   each rationale/risk citing a packet field.
4. Write ``data/processed/recommendations.jsonl`` and log that the
   recommendations are queued for human review.

This script never executes a trade and never records an approval — every
recommendation is routed to the human review layer (Phase 9+).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from ai_investment_workflow.agents import Orchestrator, build_provider
from ai_investment_workflow.rag import (
    NotesSource,
    PerformanceSource,
    SignalSource,
    SnapshotSource,
    build_context_packets,
    canonical_json,
)
from ai_investment_workflow.schemas import ContextPacket, Recommendation
from ai_investment_workflow.strategies import build_strategy_signals
from ai_investment_workflow.utils import (
    get_logger,
    load_settings,
    processed_dir,
    setup_logging,
)

#: Canonical recommendation artifact under ``data/processed/``.
RECS_FILENAME: str = "recommendations.jsonl"
PACKETS_FILENAME: str = "context_packets.jsonl"

log = get_logger(__name__)


def load_context_packets(path: str | Path) -> list[ContextPacket]:
    """Load packets from a Phase 7 JSONL file, preserving file order."""
    text = Path(path).read_text(encoding="utf-8")
    packets: list[ContextPacket] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        packets.append(ContextPacket.model_validate(json.loads(line)))
    return packets


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


def _build_packets_from_artifacts(top_n: int) -> list[ContextPacket]:
    """Fallback: rebuild packets from Phase 2-6 artifacts (no LLM, no network)."""
    features_path = processed_dir() / "features.parquet"
    prices_path = processed_dir() / "prices.parquet"
    if not features_path.is_file() or not prices_path.is_file():
        log.error(
            "no context_packets.jsonl and missing features/prices parquet; "
            "run scripts/build_embeddings.py (or the Phase 2-3 scripts) first"
        )
        return []

    prices = pd.read_parquet(prices_path)
    prices["date"] = pd.to_datetime(prices["date"]).dt.date
    features = pd.read_parquet(features_path)
    features["date"] = pd.to_datetime(features["date"]).dt.date

    rankings_path = processed_dir() / "rankings.parquet"
    if rankings_path.is_file():
        rankings = pd.read_parquet(rankings_path)
        rankings["date"] = pd.to_datetime(rankings["date"]).dt.date
    else:
        rankings = _recompute_rankings(features)
    if rankings.empty:
        return []
    as_of = rankings["date"].max()

    records_path = processed_dir() / "performance_records.parquet"
    performance_records = (
        pd.read_parquet(records_path) if records_path.is_file() else None
    )

    sources = [
        SnapshotSource(prices),
        SignalSource(features, prices=prices, artifacts_dir=processed_dir()),
        PerformanceSource(performance_records),
        NotesSource(),
    ]
    packets = build_context_packets(rankings, as_of=as_of, top_n=top_n, sources=sources)
    # Deterministic order: by rank, then asset id (mirrors the JSONL artifact).
    return sorted(
        packets.values(), key=lambda p: (p.strategy_signal.rank, p.asset_id)
    )


def write_recommendations(recs: list[Recommendation], path: str | Path) -> Path:
    """Write recommendations as deterministic canonical JSONL bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(canonical_json(rec) + "\n" for rec in recs)
    path.write_bytes(body.encode("utf-8"))
    return path


def main() -> int:
    setup_logging()
    settings = load_settings()
    top_n = int(settings.get("top_n_candidates", 10))

    packets_path = processed_dir() / PACKETS_FILENAME
    if packets_path.is_file():
        packets = load_context_packets(packets_path)
        log.info("loaded %d context packets from %s", len(packets), packets_path)
    else:
        log.info("%s absent; rebuilding packets via rag pipeline", packets_path)
        packets = _build_packets_from_artifacts(top_n)

    if not packets:
        log.warning("no context packets available; nothing to recommend")
        return 0

    provider = build_provider(settings.get("reasoning_provider"))
    log.info("reasoning provider: %s", provider.name)
    orchestrator = Orchestrator(provider)

    recommendations = [orchestrator.recommend(packet) for packet in packets]

    out_path = write_recommendations(recommendations, processed_dir() / RECS_FILENAME)
    log.info(
        "wrote %d recommendations to %s; queued for human review (no trades executed)",
        len(recommendations),
        out_path,
    )
    for rec in recommendations:
        log.info(
            "  %s -> %s (conviction %.2f)", rec.asset_id, rec.action.value, rec.conviction
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
