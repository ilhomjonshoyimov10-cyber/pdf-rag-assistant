"""Guards retrieval quality: a change that makes BM25 retrieval worse on the eval set fails CI."""

from __future__ import annotations

import json
from pathlib import Path

from rag.pipeline import RAGPipeline
from rag.store import DocumentStore

QUESTIONS = Path(__file__).resolve().parent.parent / "eval" / "questions.jsonl"


def test_default_retrieval_finds_the_right_page(sample_pdf):
    pipeline = RAGPipeline(DocumentStore(), top_k=3)  # defaults: BM25 with prefix-5 stemming
    pipeline.ingest("q.pdf", sample_pdf)
    items = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]

    top1 = sum(pipeline.ask(i["question"]).sources[:1][0].page == i["page"] for i in items) / len(items)
    top3 = sum(i["page"] in [s.page for s in pipeline.ask(i["question"]).sources] for i in items) / len(items)

    assert top1 >= 0.85
    assert top3 >= 0.90
