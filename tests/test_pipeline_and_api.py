from __future__ import annotations

from fastapi.testclient import TestClient

from rag.api import create_app
from rag.config import Settings
from rag.generator import build_messages, cited_numbers
from rag.pipeline import RAGPipeline
from rag.store import DocumentStore
from tests.conftest import FakeAnswerer, FakeEmbedder, make_pdf


def test_ingest_and_extractive_answer_cites_correct_page(sample_pdf):
    pipeline = RAGPipeline(DocumentStore())
    info = pipeline.ingest("qoidalar.pdf", sample_pdf)

    answer = pipeline.ask("Yillik ta'til necha kun?")

    assert info.pages == 6
    assert answer.mode == "extractive"
    assert answer.sources[0].page == 2
    assert answer.sources[0].filename == "qoidalar.pdf"
    assert "21 ish kunidan" in answer.answer


def test_llm_answer_marks_only_cited_sources(sample_pdf):
    answerer = FakeAnswerer("Yiliga 21 ish kuni [1].")
    pipeline = RAGPipeline(DocumentStore(), answerer=answerer, top_k=3)
    pipeline.ingest("q.pdf", sample_pdf)

    answer = pipeline.ask("Yillik ta'til necha kun?")

    assert answer.mode == "llm"
    assert [s.cited for s in answer.sources] == [True, False, False][: len(answer.sources)]
    question, hits = answerer.seen[0]
    assert question == "Yillik ta'til necha kun?"
    assert len(hits) <= 3


def test_question_without_matches_is_not_sent_to_the_llm(sample_pdf):
    answerer = FakeAnswerer()
    pipeline = RAGPipeline(DocumentStore(), answerer=answerer)
    pipeline.ingest("q.pdf", sample_pdf)

    answer = pipeline.ask("quantum chromodynamics")

    assert answer.mode == "not_found"
    assert answer.sources == []
    assert answerer.seen == []


def test_duplicate_upload_is_ignored_and_delete_works(sample_pdf):
    pipeline = RAGPipeline(DocumentStore())
    first = pipeline.ingest("a.pdf", sample_pdf)
    second = pipeline.ingest("b.pdf", sample_pdf)

    assert first.id == second.id
    assert len(pipeline.documents()) == 1
    assert pipeline.delete(first.id) is True
    assert pipeline.ask("ta'til").mode == "not_found"


def test_changing_embedding_model_reembeds_existing_chunks(tmp_path, sample_pdf):
    RAGPipeline(DocumentStore(tmp_path), FakeEmbedder("model-a")).ingest("q.pdf", sample_pdf)

    new_model = FakeEmbedder("model-b")
    pipeline = RAGPipeline(DocumentStore(tmp_path), new_model)

    assert new_model.calls == 1
    assert pipeline.store.embedding_model == "model-b"
    assert pipeline.retriever.dense_enabled


def test_prompt_numbers_sources_with_file_and_page(sample_pdf):
    pipeline = RAGPipeline(DocumentStore())
    pipeline.ingest("q.pdf", sample_pdf)
    hits = pipeline.retriever.search("parol", top_k=2)

    messages = build_messages("Parol?", hits, {hits[0].chunk.doc_id: "q.pdf"})

    assert "ONLY the numbered sources" in messages[0]["content"]
    assert "[1] q.pdf, page 5" in messages[1]["content"]
    assert cited_numbers("A [1] va B [2][3].") == {1, 2, 3}


# ---------- HTTP API ----------
def make_client(pipeline: RAGPipeline | None = None) -> TestClient:
    pipeline = pipeline or RAGPipeline(DocumentStore())
    return TestClient(create_app(pipeline=pipeline, settings=Settings(_env_file=None, max_upload_mb=1)))


def test_api_upload_list_ask_delete(sample_pdf):
    with make_client() as client:
        uploaded = client.post("/api/documents", files={"file": ("q.pdf", sample_pdf, "application/pdf")})
        assert uploaded.status_code == 201
        doc_id = uploaded.json()["id"]

        assert client.get("/api/documents").json()[0]["filename"] == "q.pdf"
        health = client.get("/health").json()
        assert health == {"status": "ok", "documents": 1, "chunks": 7, "retrieval": "bm25", "generation": "extractive"}

        answer = client.post("/api/ask", json={"question": "Parol uzunligi qancha?"}).json()
        assert answer["sources"][0]["page"] == 5

        assert client.delete(f"/api/documents/{doc_id}").status_code == 204
        assert client.delete(f"/api/documents/{doc_id}").status_code == 404


def test_api_rejects_bad_uploads():
    with make_client() as client:
        assert client.post("/api/documents", files={"file": ("notes.txt", b"hi", "text/plain")}).status_code == 415
        broken = client.post("/api/documents", files={"file": ("x.pdf", b"not a pdf", "application/pdf")})
        assert broken.status_code == 422
        big = make_pdf(["x"]) + b"0" * (1024 * 1024)
        assert client.post("/api/documents", files={"file": ("big.pdf", big, "application/pdf")}).status_code == 413


def test_api_validates_question_and_serves_ui():
    with make_client() as client:
        assert client.post("/api/ask", json={"question": ""}).status_code == 422
        assert "PDF RAG Assistant" in client.get("/").text


def test_api_provider_failure_returns_502(sample_pdf):
    class Broken:
        def answer(self, question, hits, titles):
            raise RuntimeError("quota exceeded")

    pipeline = RAGPipeline(DocumentStore(), answerer=Broken())
    pipeline.ingest("q.pdf", sample_pdf)
    with make_client(pipeline) as client:
        response = client.post("/api/ask", json={"question": "ta'til necha kun"})
    assert response.status_code == 502
    assert "quota exceeded" in response.json()["detail"]
