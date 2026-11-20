"""Answer generation grounded in retrieved passages, with numbered citations."""

from __future__ import annotations

import re
from typing import Protocol

from openai import OpenAI

from rag.retriever import Hit

SYSTEM_PROMPT = """You answer questions about the user's documents.

Rules:
1. Use ONLY the numbered sources below. Do not use outside knowledge.
2. After every claim, cite its source like [1] or [2][3].
3. If the sources do not contain the answer, say that the documents do not contain this information.
   Do not guess.
4. Answer in the same language as the question. Be concise and keep exact numbers, dates and names."""


class Answerer(Protocol):
    def answer(self, question: str, hits: list[Hit], titles: dict[str, str]) -> str: ...


def format_sources(hits: list[Hit], titles: dict[str, str]) -> str:
    blocks = []
    for n, hit in enumerate(hits, start=1):
        title = titles.get(hit.chunk.doc_id, hit.chunk.doc_id)
        blocks.append(f"[{n}] {title}, page {hit.chunk.page}\n{hit.chunk.text}")
    return "\n\n".join(blocks)


def build_messages(question: str, hits: list[Hit], titles: dict[str, str]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Sources:\n\n{format_sources(hits, titles)}\n\nQuestion: {question}"},
    ]


_CITATION = re.compile(r"\[(\d+)\]")


def cited_numbers(answer: str) -> set[int]:
    return {int(n) for n in _CITATION.findall(answer)}


class LLMAnswerer:
    def __init__(self, api_key: str, base_url: str, model: str, temperature: float = 0.1, client: OpenAI | None = None):
        self._client = client or OpenAI(api_key=api_key, base_url=base_url, timeout=60, max_retries=2)
        self._model = model
        self._temperature = temperature

    def answer(self, question: str, hits: list[Hit], titles: dict[str, str]) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=build_messages(question, hits, titles),  # type: ignore[arg-type]
            temperature=self._temperature,
        )
        return (response.choices[0].message.content or "").strip()
