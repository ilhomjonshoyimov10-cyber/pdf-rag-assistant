from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from rag.embeddings import l2_normalize
from rag.text import tokenize

SAMPLE_PDF = Path(__file__).resolve().parent.parent / "samples" / "ichki-tartib-qoidalari.pdf"


class FakeEmbedder:
    """Deterministic hashed bag-of-stems vectors: similar wording -> similar vectors. Tests only."""

    def __init__(self, name: str = "fake-embedder", dim: int = 256) -> None:
        self.name = name
        self.dim = dim
        self.calls = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text, prefix_len=5):
                bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim
                matrix[row, bucket] += 1.0
        return l2_normalize(matrix)


class FakeAnswerer:
    def __init__(self, reply: str = "Javob [1].") -> None:
        self.reply = reply
        self.seen: list[tuple[str, list]] = []

    def answer(self, question, hits, titles) -> str:
        self.seen.append((question, hits))
        return self.reply


def make_pdf(pages: list[str]) -> bytes:
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4, invariant=1)
    for text in pages:
        y = 800
        for line in text.split("\n"):
            pdf.drawString(50, y, line)
            y -= 16
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


@pytest.fixture
def sample_pdf() -> bytes:
    return SAMPLE_PDF.read_bytes()
