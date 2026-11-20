"""Retrieval evaluation: does the right page come back for each question?

Metrics (per configuration):
  hit@1  share of questions whose top result is on the expected page
  hit@3  share with the expected page anywhere in the top 3
  MRR    mean of 1 / rank of the first result on the expected page (0 if not in the top 10)

Usage:
  python eval/run_eval.py                      # BM25 variants, no API key needed
  python eval/run_eval.py --embeddings         # adds dense and hybrid (needs LLM_API_KEY)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.bm25 import BM25Index  # noqa: E402
from rag.config import Settings  # noqa: E402
from rag.documents import chunk_pages, document_id, load_pdf  # noqa: E402
from rag.embeddings import OpenAICompatibleEmbedder  # noqa: E402
from rag.retriever import Retriever  # noqa: E402
from rag.store import DocumentInfo, DocumentStore  # noqa: E402

PDF = ROOT / "samples" / "ichki-tartib-qoidalari.pdf"
QUESTIONS = ROOT / "eval" / "questions.jsonl"


def metrics(ranked_pages: list[list[int]], expected: list[int]) -> dict[str, float]:
    hit1 = hit3 = rr = 0.0
    for pages, want in zip(ranked_pages, expected, strict=True):
        hit1 += pages[:1] == [want]
        hit3 += want in pages[:3]
        rr += next((1 / (i + 1) for i, p in enumerate(pages[:10]) if p == want), 0.0)
    n = len(expected)
    return {"hit@1": hit1 / n, "hit@3": hit3 / n, "MRR": rr / n}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", action="store_true", help="also evaluate dense and hybrid retrieval")
    parser.add_argument("--chunk-size", type=int, default=Settings().chunk_size)
    args = parser.parse_args()

    data = PDF.read_bytes()
    doc_id = document_id(data)
    pages = load_pdf(data)
    chunks = chunk_pages(pages, doc_id, chunk_size=args.chunk_size, overlap=min(150, args.chunk_size // 4))
    items = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected = [item["page"] for item in items]
    print(f"{len(pages)} pages, {len(chunks)} chunks (chunk_size={args.chunk_size}), {len(items)} questions\n")

    results: dict[str, dict[str, float]] = {}
    texts = [c.text for c in chunks]
    for label, prefix in [("BM25, whole words", None), ("BM25, prefix-5 stems", 5), ("BM25, prefix-6 stems", 6)]:
        index = BM25Index(texts, prefix_len=prefix)
        ranked = []
        for item in items:
            scores = index.scores(item["question"])
            order = sorted(range(len(chunks)), key=lambda i: -scores[i])
            ranked.append([chunks[i].page for i in order if scores[i] > 0])
        results[label] = metrics(ranked, expected)

    if args.embeddings:
        settings = Settings()
        embedder = OpenAICompatibleEmbedder(
            settings.llm_api_key.get_secret_value(), settings.llm_base_url, settings.embedding_model
        )
        store = DocumentStore()
        store.add(DocumentInfo(doc_id, PDF.name, len(pages), len(chunks), ""), chunks, embedder.embed(texts))
        store.embedding_model = embedder.name
        retriever = Retriever(store, embedder, min_similarity=-1.0)
        for label, mode in [(f"Dense ({embedder.name})", "dense"), ("Hybrid (BM25 + dense, RRF)", "hybrid")]:
            ranked = [[h.chunk.page for h in retriever.search(i["question"], top_k=10, mode=mode)] for i in items]
            results[label] = metrics(ranked, expected)

    width = max(len(k) for k in results)
    print(f"{'Configuration':<{width}} | hit@1 | hit@3 |  MRR")
    print(f"{'-' * width}-|-------|-------|------")
    for label, m in results.items():
        print(f"{label:<{width}} | {m['hit@1']:.2f}  | {m['hit@3']:.2f}  | {m['MRR']:.2f}")


if __name__ == "__main__":
    main()
