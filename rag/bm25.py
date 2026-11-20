"""Okapi BM25, implemented directly (about 40 lines) instead of pulled in as a dependency.

score(q, d) = sum over query terms t of  idf(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
idf(t)      = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
"""

from __future__ import annotations

import math
from collections import Counter

from rag.text import tokenize


class BM25Index:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75, prefix_len: int | None = None) -> None:
        self.k1, self.b = k1, b
        self.prefix_len = prefix_len
        self._docs = [Counter(tokenize(t, prefix_len)) for t in texts]
        self._lengths = [sum(d.values()) for d in self._docs]
        self._avgdl = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        df: Counter[str] = Counter()
        for doc in self._docs:
            df.update(doc.keys())
        n = len(self._docs)
        self._idf = {term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    def __len__(self) -> int:
        return len(self._docs)

    def scores(self, query: str) -> list[float]:
        terms = [t for t in dict.fromkeys(tokenize(query, self.prefix_len)) if t in self._idf]
        results = []
        for doc, length in zip(self._docs, self._lengths, strict=True):
            norm = self.k1 * (1 - self.b + self.b * length / self._avgdl) if self._avgdl else self.k1
            score = 0.0
            for term in terms:
                tf = doc.get(term, 0)
                if tf:
                    score += self._idf[term] * tf * (self.k1 + 1) / (tf + norm)
            results.append(score)
        return results
