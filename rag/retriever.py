"""Hybrid retrieval: BM25 (exact words, numbers, names) + dense vectors (meaning, paraphrases),
merged with Reciprocal Rank Fusion.

RRF score = sum over retrievers of 1 / (k + rank). It needs no score calibration between the two
retrievers, which is why it is a common default for hybrid search.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rag.bm25 import BM25Index
from rag.documents import Chunk
from rag.embeddings import Embedder
from rag.store import DocumentStore


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    bm25_rank: int | None
    dense_rank: int | None


class Retriever:
    def __init__(
        self,
        store: DocumentStore,
        embedder: Embedder | None = None,
        *,
        candidates: int = 20,
        rrf_k: int = 60,
        min_similarity: float = 0.3,
        bm25_prefix_len: int | None = 5,
    ) -> None:
        self.store = store
        self.bm25_prefix_len = bm25_prefix_len
        self.embedder = embedder
        self.candidates = candidates
        self.rrf_k = rrf_k
        self.min_similarity = min_similarity
        self._bm25 = BM25Index([])
        self.refresh()

    @property
    def dense_enabled(self) -> bool:
        return self.embedder is not None and self.store.vectors is not None

    def refresh(self) -> None:
        """Rebuild the BM25 index after documents are added or removed."""
        self._bm25 = BM25Index([c.text for c in self.store.chunks], prefix_len=self.bm25_prefix_len)

    def bm25_ranking(self, query: str) -> list[int]:
        scores = self._bm25.scores(query)
        ranked = sorted((i for i, s in enumerate(scores) if s > 0), key=lambda i: -scores[i])
        return ranked[: self.candidates]

    def dense_ranking(self, query: str) -> list[int]:
        if not self.dense_enabled:
            return []
        assert self.embedder is not None and self.store.vectors is not None
        query_vector = self.embedder.embed([query])[0]
        similarities = self.store.vectors @ query_vector
        order = np.argsort(-similarities)[: self.candidates]
        return [int(i) for i in order if similarities[i] >= self.min_similarity]

    def search(self, query: str, top_k: int = 4, mode: str = "hybrid") -> list[Hit]:
        if not self.store.chunks:
            return []
        bm25 = self.bm25_ranking(query) if mode in ("hybrid", "bm25") else []
        dense = self.dense_ranking(query) if mode in ("hybrid", "dense") else []

        fused: dict[int, float] = {}
        for ranking in (bm25, dense):
            for rank, index in enumerate(ranking):
                fused[index] = fused.get(index, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        bm25_pos = {idx: r + 1 for r, idx in enumerate(bm25)}
        dense_pos = {idx: r + 1 for r, idx in enumerate(dense)}
        best = sorted(fused, key=lambda i: -fused[i])[:top_k]
        return [Hit(self.store.chunks[i], round(fused[i], 5), bm25_pos.get(i), dense_pos.get(i)) for i in best]
