"""Retrieval pipeline: top-N candidates → grounded packets → artifacts.

Reproducibility contract: building twice from the same inputs produces
byte-identical JSONL (canonical JSON, fixed rank ordering, ``\\n`` line
endings written as bytes).
"""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ..schemas import ContextPacket
from ..utils.logging import get_logger
from .base import ContextSource, canonical_json
from .context_packet import build_context_packet
from .embeddings import packet_text

#: Canonical packet artifact under ``data/processed/``.
PACKETS_FILENAME: str = "context_packets.jsonl"

log = get_logger(__name__)


def select_top_candidates(
    rankings: pd.DataFrame, as_of: date, top_n: int
) -> list[str]:
    """Top-N asset ids at ``as_of`` by rank (asset_id tiebreak)."""
    if rankings.empty or top_n <= 0:
        return []
    slice_ = rankings.loc[rankings["date"] == as_of]
    picks = slice_.sort_values(["rank", "asset_id"]).head(top_n)
    return picks["asset_id"].tolist()


def build_context_packets(
    rankings: pd.DataFrame,
    as_of: date | None = None,
    top_n: int = 10,
    sources: Sequence[ContextSource] = (),
) -> dict[str, ContextPacket]:
    """One grounded ``ContextPacket`` per top-N asset at ``as_of``.

    ``as_of=None`` uses the most recent date in ``rankings``. Assets
    whose required fields cannot be grounded are skipped with a warning
    — never fabricated.
    """
    if rankings.empty:
        return {}
    if as_of is None:
        as_of = rankings["date"].max()

    out: dict[str, ContextPacket] = {}
    for asset_id in select_top_candidates(rankings, as_of, top_n):
        packet = build_context_packet(asset_id, as_of, sources)
        if packet is None:
            log.warning(
                "skipping %s at %s: required context fields unavailable",
                asset_id,
                as_of,
            )
            continue
        out[asset_id] = packet
    return out


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def packets_to_jsonl(packets: Mapping[str, ContextPacket]) -> str:
    """Line-delimited canonical JSON, ordered by (rank, asset_id)."""
    ordered = sorted(
        packets.values(), key=lambda p: (p.strategy_signal.rank, p.asset_id)
    )
    return "".join(canonical_json(p) + "\n" for p in ordered)


def write_context_packets(
    packets: Mapping[str, ContextPacket], path: str | Path
) -> Path:
    """Write packets as JSONL bytes (deterministic, overwrite semantics).

    The format is line-delimited so future phases can append review
    cycles; this builder always rewrites the file so a rebuild from the
    same inputs is byte-identical.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(packets_to_jsonl(packets).encode("utf-8"))
    return path


def embedding_artifact_path(packet: ContextPacket, directory: str | Path) -> Path:
    """``{asset_id}_{YYYYMMDD}.parquet`` under ``directory``."""
    fname = f"{packet.asset_id}_{packet.timestamp.strftime('%Y%m%d')}.parquet"
    return Path(directory) / fname


def write_embedding_artifact(
    packet: ContextPacket,
    vector: np.ndarray,
    directory: str | Path,
) -> Path:
    """Persist one packet's embedding (vector + grounded metadata)."""
    path = embedding_artifact_path(packet, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = packet_text(packet)
    frame = pd.DataFrame(
        [
            {
                "asset_id": packet.asset_id,
                "as_of": packet.timestamp.isoformat(),
                "dim": int(np.asarray(vector).shape[0]),
                "vector": np.asarray(vector, dtype=float).tolist(),
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "composite_score": float(packet.strategy_signal.composite_score),
                "rank": int(packet.strategy_signal.rank),
            }
        ]
    )
    frame.to_parquet(path, index=False)
    return path
