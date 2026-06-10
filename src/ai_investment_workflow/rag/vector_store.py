"""File-based vector store persisted under ``data/embeddings/``.

Deterministic in-memory store with parquet persistence. Query uses
cosine similarity with a stable ``(score desc, key asc)`` ordering so
results never depend on insertion order. Pure pandas/numpy — no
network, no LLM, no external index.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .embeddings import DEFAULT_DIM

#: Default index filename inside ``data/embeddings/``.
DEFAULT_STORE_FILENAME: str = "vector_store.parquet"


class VectorStore:
    """Keyed store of fixed-dimension vectors with JSON metadata."""

    def __init__(self, dim: int = DEFAULT_DIM) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = dim
        self._vectors: dict[str, np.ndarray] = {}
        self._metadata: dict[str, dict] = {}

    def __len__(self) -> int:
        return len(self._vectors)

    def __contains__(self, key: str) -> bool:
        return key in self._vectors

    def keys(self) -> list[str]:
        return sorted(self._vectors)

    # ------------------------------------------------------------------
    # Mutation / lookup
    # ------------------------------------------------------------------

    def put(self, key: str, vector, metadata: dict | None = None) -> None:
        """Insert or replace ``key``. Metadata must be JSON-serializable."""
        if not key:
            raise ValueError("key must be a non-empty string")
        arr = np.asarray(vector, dtype=float)
        if arr.shape != (self.dim,):
            raise ValueError(
                f"vector for {key!r} has shape {arr.shape}; expected ({self.dim},)"
            )
        meta = dict(metadata or {})
        json.dumps(meta)  # fail fast on non-serializable metadata
        self._vectors[key] = arr.copy()
        self._metadata[key] = meta

    def get(self, key: str) -> tuple[np.ndarray, dict] | None:
        if key not in self._vectors:
            return None
        return self._vectors[key].copy(), dict(self._metadata[key])

    def query(self, vector, top_k: int = 5) -> list[tuple[str, float, dict]]:
        """Top-k by cosine similarity: ``[(key, score, metadata), ...]``.

        Ordering is deterministic: score descending, then key ascending.
        Zero-norm vectors score 0.0 against everything.
        """
        if top_k <= 0:
            return []
        q = np.asarray(vector, dtype=float)
        if q.shape != (self.dim,):
            raise ValueError(
                f"query vector has shape {q.shape}; expected ({self.dim},)"
            )
        q_norm = float(np.linalg.norm(q))

        scored: list[tuple[str, float, dict]] = []
        for key in sorted(self._vectors):
            v = self._vectors[key]
            v_norm = float(np.linalg.norm(v))
            if q_norm == 0.0 or v_norm == 0.0:
                score = 0.0
            else:
                score = float(np.dot(q, v) / (q_norm * v_norm))
            scored.append((key, score, dict(self._metadata[key])))

        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:top_k]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Write the store to parquet (rows sorted by key)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "key": key,
                "dim": self.dim,
                "vector": self._vectors[key].tolist(),
                "metadata_json": json.dumps(
                    self._metadata[key], sort_keys=True, separators=(",", ":")
                ),
            }
            for key in sorted(self._vectors)
        ]
        frame = pd.DataFrame(rows, columns=["key", "dim", "vector", "metadata_json"])
        frame.to_parquet(path, index=False)
        return path

    @classmethod
    def load(cls, path: str | Path, dim: int | None = None) -> "VectorStore":
        """Rebuild a store from a parquet written by :meth:`save`."""
        path = Path(path)
        frame = pd.read_parquet(path)
        if dim is None:
            dim = int(frame["dim"].iloc[0]) if len(frame) else DEFAULT_DIM
        store = cls(dim=dim)
        for row in frame.itertuples(index=False):
            store.put(
                str(row.key),
                np.asarray(row.vector, dtype=float),
                json.loads(row.metadata_json),
            )
        return store
