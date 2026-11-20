"""Dense embeddings through any OpenAI-compatible /embeddings endpoint (Gemini, OpenAI, Ollama...)."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from openai import OpenAI


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalised vectors."""
        ...


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class OpenAICompatibleEmbedder:
    def __init__(self, api_key: str, base_url: str, model: str, batch_size: int = 64, client: OpenAI | None = None):
        self._client = client or OpenAI(api_key=api_key, base_url=base_url, timeout=60, max_retries=2)
        self._model = model
        self._batch_size = batch_size
        self.name = model

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            response = self._client.embeddings.create(model=self._model, input=batch)
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
        return l2_normalize(np.asarray(vectors, dtype=np.float32))
