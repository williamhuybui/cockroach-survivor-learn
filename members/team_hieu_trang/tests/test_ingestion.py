"""Exercise the real Chroma store with deterministic test-only embeddings."""

import json

import pymupdf
import pytest

from app import ingest, paper_store
from app.config import EMBED_DIM


def make_paper(
    tmp_path, text="Abstract\nSummary\n1 Introduction\nIntro\n2 Methods\nMethod"
):
    """Create a tiny PDF fixture; this never enters the production index."""

    path = tmp_path / "paper.pdf"
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 72), text)
        pdf.save(path)
    return {
        "id": "paper-1",
        "title": "paper.pdf",
        "path": str(path),
        "_path": path,
        "source_type": "google_drive",
        "source_url": "https://drive.google.com/open?id=paper-1",
    }


def vectors(texts):
    """Use a valid fixed vector so storage tests need no model download."""

    return [[1.0] + [0.0] * (EMBED_DIM - 1) for text in texts]


def test_changed_paper_replaces_chunks_and_unchanged_skips_embedding(
    tmp_path, monkeypatch
):
    row = make_paper(tmp_path)
    store = paper_store.collection()
    monkeypatch.setattr(ingest, "embed_texts", vectors)
    assert ingest.index_paper(store, row)[0] == "indexed"
    old_ids = set(store.get()["ids"])
    monkeypatch.setattr(
        ingest, "embed_texts", lambda texts: pytest.fail("Unchanged PDF was embedded")
    )
    assert ingest.index_paper(store, row)[0] == "unchanged"
    row["title"] = "Renamed paper.pdf"
    monkeypatch.setattr(ingest, "embed_texts", vectors)
    assert ingest.index_paper(store, row)[0] == "indexed"
    assert not old_ids.intersection(store.get()["ids"])
    assert paper_store.collection().count() == store.count()


def test_embedding_failure_preserves_existing_paper(tmp_path, monkeypatch):
    row = make_paper(tmp_path)
    store = paper_store.collection()
    monkeypatch.setattr(ingest, "embed_texts", vectors)
    ingest.index_paper(store, row)
    before = store.get()["ids"]
    row["title"] = "Changed"

    def fail(texts):
        raise RuntimeError("Model unavailable")

    monkeypatch.setattr(ingest, "embed_texts", fail)
    with pytest.raises(RuntimeError):
        ingest.index_paper(store, row)
    assert store.get()["ids"] == before


def test_manifest_ignores_non_pdf_and_rejects_missing_pdf(tmp_path):
    manifest = tmp_path / "sources.json"
    row = make_paper(tmp_path)
    row.pop("_path")
    manifest.write_text(json.dumps([row, {**row, "id": "other", "path": "notes.docx"}]))
    assert len(ingest.load_manifest(manifest)) == 1
    row["path"] = "missing.pdf"
    manifest.write_text(json.dumps([row]))
    with pytest.raises(ValueError, match="missing"):
        ingest.load_manifest(manifest)


def test_embedding_signature_mismatch_requires_rebuild(monkeypatch):
    paper_store.collection()
    monkeypatch.setattr(paper_store, "EMBED_MODEL", "another-model")
    with pytest.raises(ValueError, match="rebuild"):
        paper_store.collection()


def test_real_chroma_query_keeps_source_metadata(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from app import retrieval

    row = make_paper(tmp_path)
    store = paper_store.collection()
    monkeypatch.setattr(ingest, "embed_texts", vectors)
    ingest.index_paper(store, row)
    monkeypatch.setattr(retrieval, "embed_query", lambda query: vectors([query])[0])
    monkeypatch.setattr(
        retrieval,
        "reranker",
        lambda: SimpleNamespace(predict=lambda pairs: [2.0] * len(pairs)),
    )
    result = retrieval.search("What method is used?")
    assert result
    assert all(item["source_url"] == row["source_url"] for item in result)
    assert all("page 1" in item["location"] for item in result)
    assert all(item["rerank_score"] == 2.0 for item in result)


def test_explicit_reset_replaces_incompatible_collection(monkeypatch):
    paper_store.collection()
    monkeypatch.setattr(paper_store, "EMBED_MODEL", "another-model")
    paper_store.reset_collection()
    assert paper_store.collection().count() == 0


def test_cleanup_removes_only_missing_paper_children_and_parents(tmp_path, monkeypatch):
    """One manifest deletion removes both levels while preserving the remaining paper."""

    row = make_paper(tmp_path)
    other = {**row, "id": "paper-2", "title": "Retained paper.pdf"}
    store = paper_store.collection()
    monkeypatch.setattr(ingest, "embed_texts", vectors)
    ingest.index_paper(store, row)
    ingest.index_paper(store, other)
    removed = len(store.get(where={"document_id": row["id"]})["ids"])
    assert ingest.remove_missing_papers(store, {other["id"]}) == removed
    for index in (store, paper_store.parent_collection()):
        assert not index.get(where={"document_id": row["id"]})["ids"]
        assert index.get(where={"document_id": other["id"]})["ids"]
