"""A small persistent document store: metadata (JSON), chunks (JSONL) and vectors (NumPy).

For a few thousand chunks, a NumPy matrix and a dot product are as fast as a vector database and
need no extra service. The class boundary is where a FAISS / pgvector / Qdrant backend would plug in.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from rag.documents import Chunk


@dataclass(frozen=True)
class DocumentInfo:
    id: str
    filename: str
    pages: int
    chunks: int
    added_at: str


class DocumentStore:
    def __init__(self, directory: str | Path | None = None) -> None:
        """`directory=None` keeps everything in memory (used in tests)."""
        self.directory = Path(directory) if directory else None
        self.documents: dict[str, DocumentInfo] = {}
        self.chunks: list[Chunk] = []
        self.vectors: np.ndarray | None = None  # rows aligned with self.chunks
        self.embedding_model: str | None = None
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._load()

    # ---------- mutations ----------
    def add(self, info: DocumentInfo, chunks: list[Chunk], vectors: np.ndarray | None) -> None:
        if vectors is not None and len(vectors) != len(chunks):
            raise ValueError("vectors and chunks must have the same length")
        self.documents[info.id] = info
        self.chunks.extend(chunks)
        if vectors is None:
            self.vectors = None  # dense index is incomplete -> disabled until re-embedded
        elif self.vectors is None and len(self.chunks) == len(chunks):
            self.vectors = vectors
        elif self.vectors is not None:
            self.vectors = np.vstack([self.vectors, vectors])
        self._save()

    def remove(self, doc_id: str) -> bool:
        if doc_id not in self.documents:
            return False
        keep = [i for i, c in enumerate(self.chunks) if c.doc_id != doc_id]
        self.chunks = [self.chunks[i] for i in keep]
        if self.vectors is not None:
            self.vectors = self.vectors[keep] if keep else None
        del self.documents[doc_id]
        self._save()
        return True

    def set_vectors(self, vectors: np.ndarray | None, model: str | None) -> None:
        self.vectors = vectors
        self.embedding_model = model
        self._save()

    # ---------- persistence ----------
    def _load(self) -> None:
        assert self.directory is not None
        meta_path = self.directory / "documents.json"
        if not meta_path.exists():
            return
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.documents = {d["id"]: DocumentInfo(**d) for d in meta["documents"]}
        self.embedding_model = meta.get("embedding_model")
        with (self.directory / "chunks.jsonl").open(encoding="utf-8") as f:
            self.chunks = [Chunk(**json.loads(line)) for line in f if line.strip()]
        vectors_path = self.directory / "vectors.npy"
        if vectors_path.exists():
            vectors = np.load(vectors_path)
            self.vectors = vectors if len(vectors) == len(self.chunks) else None

    def _save(self) -> None:
        if not self.directory:
            return
        meta = {"embedding_model": self.embedding_model, "documents": [asdict(d) for d in self.documents.values()]}
        _atomic_write(self.directory / "documents.json", json.dumps(meta, ensure_ascii=False, indent=2))
        _atomic_write(
            self.directory / "chunks.jsonl",
            "".join(json.dumps(asdict(c), ensure_ascii=False) + "\n" for c in self.chunks),
        )
        vectors_path = self.directory / "vectors.npy"
        if self.vectors is not None:
            tmp = vectors_path.with_suffix(".tmp.npy")
            np.save(tmp, self.vectors)
            os.replace(tmp, vectors_path)
        elif vectors_path.exists():
            vectors_path.unlink()


def _atomic_write(path: Path, content: str) -> None:
    """Write to a temp file and rename, so a crash never leaves a half-written index."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)
