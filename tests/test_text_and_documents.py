from __future__ import annotations

import pytest

from rag.documents import DocumentError, Page, chunk_pages, document_id, load_pdf
from rag.text import normalize, tokenize
from tests.conftest import make_pdf


@pytest.mark.parametrize("variant", ["Oʻzbekiston", "O‘zbekiston", "O’zbekiston", "O`zbekiston", "O'zbekiston"])
def test_all_uzbek_apostrophe_variants_normalise_to_one_token(variant):
    assert tokenize(variant) == ["o'zbekiston"]


def test_tokenize_drops_stopwords_and_keeps_numbers():
    assert tokenize("Ta'til va kasallik: 21 kun") == ["ta'til", "kasallik", "21", "kun"]


def test_prefix_stemming_merges_suffixed_forms_but_not_numbers():
    forms = ["ta'tilni", "ta'tilga", "ta'tildan", "ta'til"]
    assert {tokenize(f, prefix_len=6)[0] for f in forms} == {"ta'til"}
    assert tokenize("150000 so'm", prefix_len=3) == ["150000", "so'"]


def test_normalize_joins_hyphenated_line_breaks():
    assert normalize("kompen-\nsatsiya") == "kompensatsiya"


def test_load_pdf_returns_numbered_pages(sample_pdf):
    pages = load_pdf(sample_pdf)
    assert len(pages) == 6
    assert pages[0].number == 1
    assert "21 ish kunidan" in pages[1].text


def test_load_pdf_rejects_non_pdf_and_empty_pdf():
    with pytest.raises(DocumentError):
        load_pdf(b"definitely not a pdf")
    with pytest.raises(DocumentError, match="no extractable text"):
        load_pdf(make_pdf([""]))


def test_document_id_is_stable_content_hash(sample_pdf):
    assert document_id(sample_pdf) == document_id(bytes(sample_pdf))
    assert document_id(sample_pdf) != document_id(sample_pdf + b" ")


def test_chunks_never_cross_pages_and_respect_size():
    pages = [Page(1, "Birinchi gap. " * 60), Page(2, "Ikkinchi sahifa matni. " * 40)]
    chunks = chunk_pages(pages, "doc", chunk_size=300, overlap=60)

    assert {c.page for c in chunks} == {1, 2}
    assert all(len(c.text) <= 300 for c in chunks)
    assert all("Ikkinchi" not in c.text for c in chunks if c.page == 1)
    assert len({c.id for c in chunks}) == len(chunks)


def test_neighbouring_chunks_overlap():
    sentences = " ".join(f"Gap raqami {i}." for i in range(80))
    chunks = chunk_pages([Page(1, sentences)], "doc", chunk_size=200, overlap=50)
    first_tail = chunks[0].text.split(". ")[-1]
    assert first_tail.rstrip(".") in chunks[1].text


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_pages([Page(1, "x")], "doc", chunk_size=100, overlap=100)
