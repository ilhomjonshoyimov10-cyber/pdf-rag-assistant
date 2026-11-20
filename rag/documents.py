"""PDF loading and chunking.

Chunks never cross a page boundary, so every chunk can be cited with an exact page number.
Inside a page, text is split on paragraph and sentence boundaries and packed into chunks of about
`chunk_size` characters, with `overlap` characters repeated between neighbours so a fact that sits on
a boundary is still retrievable.
"""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from rag.text import normalize


class DocumentError(ValueError):
    """The file is not a readable PDF or contains no extractable text."""


@dataclass(frozen=True)
class Page:
    number: int  # 1-based, as shown in a PDF viewer
    text: str


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    page: int
    text: str


def load_pdf(data: bytes) -> list[Page]:
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise DocumentError("Encrypted PDFs are not supported")
        pages = [Page(i + 1, normalize(page.extract_text() or "")) for i, page in enumerate(reader.pages)]
    except (PdfReadError, ValueError, KeyError) as exc:
        if isinstance(exc, DocumentError):
            raise
        raise DocumentError(f"Could not read PDF: {exc}") from exc
    pages = [p for p in pages if p.text]
    if not pages:
        raise DocumentError("The PDF has no extractable text (scanned images need OCR first)")
    return pages


def document_id(data: bytes) -> str:
    """Content hash: uploading the same file twice does not create duplicates."""
    return hashlib.sha256(data).hexdigest()[:16]


_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+|\n+")


def _split_units(text: str, max_len: int) -> list[str]:
    units: list[str] = []
    for piece in _SENTENCE_END.split(text):
        piece = piece.strip()
        while len(piece) > max_len:  # a single very long "sentence": cut on a space
            cut = piece.rfind(" ", 0, max_len)
            cut = cut if cut > max_len // 2 else max_len
            units.append(piece[:cut].strip())
            piece = piece[cut:].strip()
        if piece:
            units.append(piece)
    return units


def chunk_pages(pages: list[Page], doc_id: str, chunk_size: int = 800, overlap: int = 150) -> list[Chunk]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks: list[Chunk] = []
    for page in pages:
        current: list[str] = []
        length = 0
        for unit in _split_units(page.text, chunk_size):
            if current and length + len(unit) + 1 > chunk_size:
                chunks.append(_make_chunk(doc_id, page.number, len(chunks), current))
                # carry the tail of the previous chunk over as overlap
                tail: list[str] = []
                tail_len = 0
                for prev in reversed(current):
                    if tail_len + len(prev) > overlap:
                        break
                    tail.insert(0, prev)
                    tail_len += len(prev) + 1
                current, length = tail, tail_len
            current.append(unit)
            length += len(unit) + 1
        if current:
            chunks.append(_make_chunk(doc_id, page.number, len(chunks), current))
    return chunks


def _make_chunk(doc_id: str, page: int, index: int, units: list[str]) -> Chunk:
    return Chunk(id=f"{doc_id}:{index}", doc_id=doc_id, page=page, text=" ".join(units))
