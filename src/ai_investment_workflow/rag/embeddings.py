"""Deterministic, non-LLM embedding stub.

This is a **pure offline placeholder**, not a semantic embedding model.
It hashes the canonical JSON of a schema object into a seed and draws a
fixed-dimension unit vector from a seeded ``numpy`` generator:

    seed   = stable_hash(text)            # SHA-256, process-independent
    vector = default_rng(seed).standard_normal(dim), L2-normalized

Properties Phase 8 can rely on:

- identical text -> byte-identical vector (across runs and machines);
- different text -> a different (effectively random) vector;
- no network, no LLM SDK, stdlib hashing + numpy only.

There is **no similarity structure**: near-identical texts produce
unrelated vectors. Phase 8 swaps in a real embedding model behind this
same public surface (``embed_text`` / ``embed_packet``) without touching
callers.
"""

from __future__ import annotations

import hashlib

import numpy as np

from ..schemas import ContextPacket
from .base import canonical_json

#: Dimension of the stub embedding space.
DEFAULT_DIM: int = 64


def stable_hash(text: str) -> int:
    """First 8 bytes of SHA-256 as an unsigned int.

    Unlike builtin ``hash()``, this is stable across processes and
    platforms (no ``PYTHONHASHSEED`` dependence), so embeddings are
    reproducible everywhere.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def embed_text(text: str, dim: int = DEFAULT_DIM) -> np.ndarray:
    """Deterministic unit vector for ``text`` in ``dim`` dimensions."""
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    rng = np.random.default_rng(stable_hash(text))
    vector = rng.standard_normal(dim)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0.0 else vector


def packet_text(packet: ContextPacket) -> str:
    """Canonical string representation of a packet (the embedded text)."""
    return canonical_json(packet)


def embed_packet(packet: ContextPacket, dim: int = DEFAULT_DIM) -> np.ndarray:
    """Deterministic embedding of a ``ContextPacket``."""
    return embed_text(packet_text(packet), dim=dim)
