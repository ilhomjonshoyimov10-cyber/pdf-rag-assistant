"""HTTP API and web UI. Run: `uvicorn rag.api:app --reload` and open http://localhost:8000"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from rag.config import Settings
from rag.documents import DocumentError
from rag.embeddings import OpenAICompatibleEmbedder
from rag.generator import LLMAnswerer
from rag.pipeline import RAGPipeline
from rag.store import DocumentStore

logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)


def build_pipeline(settings: Settings) -> RAGPipeline:
    key = settings.llm_api_key.get_secret_value()
    embedder = (
        OpenAICompatibleEmbedder(key, settings.llm_base_url, settings.embedding_model)
        if key and settings.embedding_model
        else None
    )
    answerer = LLMAnswerer(key, settings.llm_base_url, settings.llm_model) if key else None
    return RAGPipeline(
        DocumentStore(settings.storage_dir),
        embedder,
        answerer,
        top_k=settings.top_k,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        min_similarity=settings.min_similarity,
        bm25_prefix_len=settings.bm25_prefix_len or None,
    )


def create_app(pipeline: RAGPipeline | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = pipeline or await run_in_threadpool(build_pipeline, settings)
        if pipeline is None and settings.seed_samples and not app.state.pipeline.documents():
            for pdf in sorted(Path(settings.samples_dir).glob("*.pdf")):
                await run_in_threadpool(app.state.pipeline.ingest, pdf.name, pdf.read_bytes())
                logger.info("Indexed sample %s", pdf.name)
        yield

    app = FastAPI(title="PDF RAG Assistant", version="1.0.0", lifespan=lifespan)

    def current() -> RAGPipeline:
        return app.state.pipeline

    @app.get("/health")
    async def health() -> dict:
        p = current()
        return {
            "status": "ok",
            "documents": len(p.documents()),
            "chunks": len(p.store.chunks),
            "retrieval": "hybrid" if p.retriever.dense_enabled else "bm25",
            "generation": "llm" if p.answerer else "extractive",
        }

    @app.get("/api/documents")
    async def list_documents() -> list[dict]:
        return [asdict(d) for d in current().documents()]

    @app.post("/api/documents", status_code=201)
    async def upload(file: UploadFile) -> dict:
        if not (file.filename or "").lower().endswith(".pdf"):
            raise HTTPException(415, "Only PDF files are supported")
        data = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
        if len(data) > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(413, f"File is larger than {settings.max_upload_mb} MB")
        try:
            info = await run_in_threadpool(current().ingest, file.filename, data)
        except DocumentError as exc:
            raise HTTPException(422, str(exc)) from exc
        return asdict(info)

    @app.delete("/api/documents/{doc_id}", status_code=204)
    async def delete(doc_id: str) -> None:
        if not await run_in_threadpool(current().delete, doc_id):
            raise HTTPException(404, "Document not found")

    @app.post("/api/ask")
    async def ask(request: AskRequest) -> dict:
        try:
            answer = await run_in_threadpool(current().ask, request.question)
        except Exception as exc:  # provider errors (quota, network) should not look like a 500 bug
            logger.exception("Answer generation failed")
            raise HTTPException(502, f"Model provider error: {exc}") from exc
        return asdict(answer)

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
