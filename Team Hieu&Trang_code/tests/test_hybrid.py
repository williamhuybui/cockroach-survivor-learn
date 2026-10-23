"""Check lexical retrieval, document scoping, and bounded parent expansion."""

from pathlib import Path

from app import retrieval
from app.document_processing import classify_format
from app.paper_store import parent_collection


def test_bm25_finds_exact_term_and_ignores_no_matches():
    records = [
        {"id": "a", "content": "GRPO uses group scores"},
        {"id": "b", "content": "Attention has keys and queries"},
    ]
    assert retrieval.keyword_candidates("GRPO", records)[0]["id"] == "a"
    assert retrieval.keyword_candidates("xyzabsent", records) == []


def test_parent_expansion_keeps_hit_and_respects_budget():
    parent_collection().add(
        ids=["p"],
        documents=["context " * 100 + "THE IMPORTANT FACT" + " continuation" * 1000],
        embeddings=[[1.0]],
    )
    row = {
        "id": "child",
        "parent_id": "p",
        "content": "Paper: A\nSection: Methods\nTHE IMPORTANT FACT",
    }
    result = retrieval.expand_parents([row, {**row, "id": "duplicate"}], 5)
    assert len(result) == 1
    assert "THE IMPORTANT FACT" in result[0]["content"]
    assert len(result[0]["content"]) <= 2400


def test_format_classification_does_not_call_every_pdf_a_paper():
    assert classify_format(Path("notes.PDF")) == "pdf"
    assert classify_format(Path("slides.pptx")) == "slides"
    assert classify_format(Path("unknown.exe")) == "unsupported"


def test_internal_paper_scope_matches_exact_title(monkeypatch):
    """A comparison scope must not mix a paper with another similarly named paper."""
    from types import SimpleNamespace

    saved = {
        "ids": ["a", "b"],
        "documents": ["Original evidence", "Survey evidence"],
        "metadatas": [{"title": "Attention.pdf"}, {"title": "Attention.pdf survey"}],
    }
    monkeypatch.setattr(
        retrieval, "collection", lambda: SimpleNamespace(get=lambda **kwargs: saved)
    )
    assert [row["id"] for row in retrieval.matching_records("Attention.pdf")] == ["a"]
    assert len(retrieval.matching_records()) == 2


def test_dense_paper_filter_runs_before_top_k(monkeypatch):
    """The real local Chroma filter can find a scoped hit outside global top one."""
    from app.paper_store import collection

    store = collection()
    meta = {
        "source_url": "https://example.com",
        "location": "page 1",
        "document_type": "sectioned_paper",
    }
    store.add(
        ids=["a", "b"],
        documents=["A", "B"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        metadatas=[{**meta, "title": "A"}, {**meta, "title": "B"}],
    )
    monkeypatch.setattr(retrieval, "embed_query", lambda query: [1.0, 0.0])
    assert retrieval.retrieve_candidates("question", count=1)[0]["title"] == "A"
    assert (
        retrieval.retrieve_candidates("question", count=1, paper="B")[0]["title"] == "B"
    )
