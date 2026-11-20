"""Ties loading, chunking, embedding, retrieval and generation together."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from rag.documents import chunk_pages, document_id, load_pdf
from rag.embeddings import Embedder
from rag.generator import Answerer, cited_numbers
from rag.retriever import Retriever
from rag.store import DocumentInfo, DocumentStore

logger = logging.getLogger(__name__)

NOT_FOUND = "The uploaded documents do not contain relevant information for this question."


@dataclass(frozen=True)
class Source:
    n: int
    doc_id: str
    filename: str
    page: int
    text: str
    score: float
    cited: bool


@dataclass(frozen=True)
class Answer:
    answer: str
    sources: list[Source]
    mode: Literal["llm", "extractive", "not_found"]


class RAGPipeline:
    def __init__(
        self,
        store: DocumentStore,
        embedder: Embedder | None = None,
        answerer: Answerer | None = None,
        *,
        top_k: int = 4,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        min_similarity: float = 0.3,
        bm25_prefix_len: int | None = 5,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.answerer = answerer
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._lock = threading.Lock()
        self._ensure_vectors()
        self.retriever = Retriever(store, embedder, min_similarity=min_similarity, bm25_prefix_len=bm25_prefix_len)

    def _ensure_vectors(self) -> None:
        """Re-embed the collection if the embedding model changed or vectors are missing."""
        if self.embedder is None or not self.store.chunks:
            return
        if self.store.vectors is not None and self.store.embedding_model == self.embedder.name:
            return
        logger.info("Embedding %d chunks with %s", len(self.store.chunks), self.embedder.name)
        vectors = self.embedder.embed([c.text for c in self.store.chunks])
        self.store.set_vectors(vectors, self.embedder.name)

    # ---------- documents ----------
    def ingest(self, filename: str, data: bytes) -> DocumentInfo:
        doc_id = document_id(data)
        with self._lock:
            if doc_id in self.store.documents:
                return self.store.documents[doc_id]
            pages = load_pdf(data)
            chunks = chunk_pages(pages, doc_id, self.chunk_size, self.chunk_overlap)
            vectors = self.embedder.embed([c.text for c in chunks]) if self.embedder else None
            info = DocumentInfo(
                id=doc_id,
                filename=filename,
                pages=len(pages),
                chunks=len(chunks),
                added_at=datetime.now(UTC).isoformat(timespec="seconds"),
            )
            if self.embedder and not self.store.chunks:
                self.store.embedding_model = self.embedder.name
            self.store.add(info, chunks, vectors)
            self.retriever.refresh()
            return info

    def delete(self, doc_id: str) -> bool:
        with self._lock:
            removed = self.store.remove(doc_id)
            if removed:
                self.retriever.refresh()
            return removed

    def documents(self) -> list[DocumentInfo]:
        return list(self.store.documents.values())

    # ---------- questions ----------
    def ask(self, question: str) -> Answer:
        hits = self.retriever.search(question, top_k=self.top_k)
        titles = {doc_id: info.filename for doc_id, info in self.store.documents.items()}
        if not hits:
            return Answer(NOT_FOUND, [], "not_found")

        if self.answerer is None:
            text = "No LLM is configured, so here are the most relevant passages:\n\n" + "\n\n".join(
                f"[{n}] {hit.chunk.text}" for n, hit in enumerate(hits, start=1)
            )
            mode: Literal["llm", "extractive", "not_found"] = "extractive"
            cited = set(range(1, len(hits) + 1))
        else:
            text = self.answerer.answer(question, hits, titles)
            mode = "llm"
            cited = cited_numbers(text)

        sources = [
            Source(
                n=n,
                doc_id=hit.chunk.doc_id,
                filename=titles.get(hit.chunk.doc_id, hit.chunk.doc_id),
                page=hit.chunk.page,
                text=hit.chunk.text,
                score=hit.score,
                cited=n in cited,
            )
            for n, hit in enumerate(hits, start=1)
        ]
        return Answer(text, sources, mode)
