from __future__ import annotations

import numpy as np
import pytest

from rag.bm25 import BM25Index
from rag.documents import Chunk
from rag.retriever import Retriever
from rag.store import DocumentInfo, DocumentStore
from tests.conftest import FakeEmbedder

TEXTS = [
    "Yillik mehnat ta'tili 21 ish kuni.",
    "Parol kamida 12 belgidan iborat bo'lishi kerak.",
    "Ish haqi har oyning 5 va 20-sanasida to'lanadi.",
]


def make_store(embedder=None) -> DocumentStore:
    store = DocumentStore()
    chunks = [Chunk(f"d:{i}", "d", i + 1, t) for i, t in enumerate(TEXTS)]
    vectors = embedder.embed(TEXTS) if embedder else None
    store.add(DocumentInfo("d", "doc.pdf", 3, 3, ""), chunks, vectors)
    store.embedding_model = embedder.name if embedder else None
    return store


def test_bm25_ranks_the_matching_document_first():
    scores = BM25Index(TEXTS).scores("parol necha belgi")
    assert int(np.argmax(scores)) == 1


def test_bm25_rare_terms_weigh_more_than_common_ones():
    index = BM25Index(["kun kun ta'til", "kun", "kun"])
    assert index.scores("ta'til")[0] > index.scores("kun")[0]


def test_bm25_without_matching_terms_scores_zero():
    assert BM25Index(TEXTS).scores("kvant fizikasi") == [0.0, 0.0, 0.0]


def test_bm25_prefix_stemming_matches_inflected_query():
    plain, stemmed = BM25Index(TEXTS), BM25Index(TEXTS, prefix_len=5)
    assert plain.scores("parollar")[1] == 0
    assert stemmed.scores("parollar")[1] > 0


def test_bm25_only_retrieval_without_embedder():
    retriever = Retriever(make_store())
    hits = retriever.search("ta'til necha kun", top_k=2)
    assert not retriever.dense_enabled
    assert hits[0].chunk.page == 1
    assert hits[0].bm25_rank == 1 and hits[0].dense_rank is None


def test_hybrid_fuses_both_rankings():
    embedder = FakeEmbedder()
    retriever = Retriever(make_store(embedder), embedder, min_similarity=0.1)
    hits = retriever.search("ish haqi qachon to'lanadi", top_k=3)

    assert retriever.dense_enabled
    assert hits[0].chunk.page == 3
    assert hits[0].bm25_rank == 1 and hits[0].dense_rank == 1
    assert hits[0].score == pytest.approx(2 / 61, abs=1e-5)  # 1/(60+1) from each retriever


def test_empty_store_returns_nothing():
    assert Retriever(DocumentStore()).search("anything") == []


def test_store_persists_and_reloads(tmp_path):
    embedder = FakeEmbedder()
    store = DocumentStore(tmp_path)
    chunks = [Chunk(f"d:{i}", "d", i + 1, t) for i, t in enumerate(TEXTS)]
    store.add(DocumentInfo("d", "doc.pdf", 3, 3, "2026-01-01"), chunks, embedder.embed(TEXTS))

    reloaded = DocumentStore(tmp_path)
    assert reloaded.documents["d"].filename == "doc.pdf"
    assert [c.text for c in reloaded.chunks] == TEXTS
    assert reloaded.vectors is not None and reloaded.vectors.shape == (3, embedder.dim)


def test_store_remove_keeps_vectors_aligned():
    embedder = FakeEmbedder()
    store = make_store(embedder)
    extra = [Chunk("e:0", "e", 1, "Boshqa hujjat matni")]
    store.add(DocumentInfo("e", "e.pdf", 1, 1, ""), extra, embedder.embed([extra[0].text]))

    assert store.remove("d") is True
    assert [c.doc_id for c in store.chunks] == ["e"]
    assert store.vectors is not None and len(store.vectors) == 1
    assert store.remove("missing") is False


def test_adding_without_vectors_disables_dense_index():
    embedder = FakeEmbedder()
    store = make_store(embedder)
    store.add(DocumentInfo("x", "x.pdf", 1, 1, ""), [Chunk("x:0", "x", 1, "matn")], None)
    assert store.vectors is None
