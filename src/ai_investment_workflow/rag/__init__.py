"""Retrieval layer: grounded ContextPacket construction.

Phase 7. Must not invent missing facts or rank candidates. Pure
pandas/numpy + stdlib hashing — no LLM, no network. The embedding stub
is a deterministic placeholder Phase 8 swaps for a real model.
"""

from .base import SOURCE_FIELDS, ContextSource, canonical_json, merge_source_payloads
from .context_packet import REQUIRED_FIELDS, build_context_packet
from .embeddings import (
    DEFAULT_DIM,
    embed_packet,
    embed_text,
    packet_text,
    stable_hash,
)
from .pipeline import (
    PACKETS_FILENAME,
    build_context_packets,
    embedding_artifact_path,
    packets_to_jsonl,
    select_top_candidates,
    write_context_packets,
    write_embedding_artifact,
)
from .sources import NotesSource, PerformanceSource, SignalSource, SnapshotSource
from .vector_store import DEFAULT_STORE_FILENAME, VectorStore

__all__ = [
    "DEFAULT_DIM",
    "DEFAULT_STORE_FILENAME",
    "ContextSource",
    "NotesSource",
    "PACKETS_FILENAME",
    "PerformanceSource",
    "REQUIRED_FIELDS",
    "SOURCE_FIELDS",
    "SignalSource",
    "SnapshotSource",
    "VectorStore",
    "build_context_packet",
    "build_context_packets",
    "canonical_json",
    "embed_packet",
    "embed_text",
    "embedding_artifact_path",
    "merge_source_payloads",
    "packet_text",
    "packets_to_jsonl",
    "select_top_candidates",
    "stable_hash",
    "write_context_packets",
    "write_embedding_artifact",
]
