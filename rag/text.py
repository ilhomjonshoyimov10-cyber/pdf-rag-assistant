"""Text normalisation and tokenisation shared by indexing and querying.

Uzbek Latin is written with several look-alike apostrophes (o' o‘ oʻ o’ o`). If the document uses one
and the user types another, a naive tokenizer sees two different words. Everything is normalised to
the ASCII apostrophe before tokenising, so "Oʻzbekiston" and "O'zbekiston" match.
"""

from __future__ import annotations

import re
import unicodedata

APOSTROPHES = "ʻʼ‘’`´ʹ′"
_APOSTROPHE_TABLE = str.maketrans(dict.fromkeys(APOSTROPHES, "'"))
_TOKEN = re.compile(r"[^\W_]+(?:'[^\W_]+)*", re.UNICODE)

# Small, high-frequency function words. Kept short on purpose: aggressive stop-word lists hurt recall.
STOPWORDS = frozenset(
    """
    va bilan uchun ham bu shu u bir yoki emas edi ekan esa lekin ammo agar qanday nima qaysi necha
    qancha kim qachon qayerda nega mi chi da ga ni ning dan the a an of to in on for and or is are was
    be by with what how which who when where why do does can i my you your it this that
    и в на с по для что как это не
    """.split()
)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_APOSTROPHE_TABLE)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # join words hyphenated across lines
    return re.sub(r"[ \t\r\f\v]+", " ", text).strip()


def tokenize(text: str, prefix_len: int | None = None) -> list[str]:
    """Lower-case word tokens without stop words.

    `prefix_len` truncates words to their first N characters, a crude but strong stemmer for
    agglutinative languages: "ta'tilni", "ta'tilga" and "ta'til" all become "ta'til" with N=6.
    Fixed-prefix stemming is a known baseline for Turkish retrieval, and Uzbek is also a Turkic language.
    """
    tokens = _TOKEN.findall(normalize(text).lower())
    tokens = [t for t in tokens if t not in STOPWORDS and (len(t) > 1 or t.isdigit())]
    if prefix_len:
        tokens = [t if t.isdigit() else t[:prefix_len] for t in tokens]
    return tokens
