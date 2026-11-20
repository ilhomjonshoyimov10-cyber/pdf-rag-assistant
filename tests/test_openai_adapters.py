"""Real embedding and chat adapters against a mocked OpenAI-compatible server (no network, no key)."""

from __future__ import annotations

import json

import numpy as np
from openai import OpenAI

from rag.embeddings import OpenAICompatibleEmbedder
from rag.generator import LLMAnswerer
from rag.pipeline import RAGPipeline
from rag.store import DocumentStore

try:  # openai>=3 ships its own fork of httpx; older versions use httpx itself
    import httpx2 as sdk_httpx
except ImportError:  # pragma: no cover
    import httpx as sdk_httpx

BASE_URL = "https://llm.example.test/v1"


class FakeServer:
    def __init__(self) -> None:
        self.requests: list[tuple[str, dict]] = []

    def handler(self, request):
        body = json.loads(request.content)
        self.requests.append((request.url.path, body))
        if request.url.path.endswith("/embeddings"):
            inputs = body["input"]
            # return items in reverse order to check that the adapter sorts by "index"
            data = [
                {"object": "embedding", "index": i, "embedding": [float(len(t)), 1.0, 0.0]}
                for i, t in enumerate(inputs)
            ][::-1]
            return sdk_httpx.Response(200, json={"object": "list", "data": data, "model": body["model"]})
        return sdk_httpx.Response(
            200,
            json={
                "id": "c1",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "21 ish kuni [1]."},
                    }
                ],
            },
        )

    def client(self) -> OpenAI:
        http = sdk_httpx.Client(transport=sdk_httpx.MockTransport(self.handler))
        return OpenAI(api_key="test", base_url=BASE_URL, http_client=http, max_retries=0)


def test_embedder_batches_sorts_and_normalises():
    server = FakeServer()
    embedder = OpenAICompatibleEmbedder("k", BASE_URL, "emb-model", batch_size=2, client=server.client())

    vectors = embedder.embed(["a", "bbb", "cc"])

    assert len(server.requests) == 2  # 3 texts, batch size 2
    assert vectors.shape == (3, 3)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert vectors[1, 0] > vectors[2, 0] > vectors[0, 0]  # order preserved despite shuffled response


def test_full_pipeline_with_real_adapters(sample_pdf):
    server = FakeServer()
    client = server.client()
    pipeline = RAGPipeline(
        DocumentStore(),
        OpenAICompatibleEmbedder("k", BASE_URL, "emb-model", client=client),
        LLMAnswerer("k", BASE_URL, "chat-model", client=client),
        min_similarity=-1.0,
    )
    pipeline.ingest("q.pdf", sample_pdf)

    answer = pipeline.ask("Yillik ta'til necha kun?")

    assert answer.mode == "llm"
    assert answer.answer == "21 ish kuni [1]."
    assert answer.sources[0].cited is True
    path, body = server.requests[-1]
    assert path.endswith("/chat/completions")
    assert body["model"] == "chat-model"
    assert "Question: Yillik ta'til necha kun?" in body["messages"][1]["content"]
