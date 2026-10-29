"""Test PDF-to-Chroma indexing using temporary stores and deterministic vectors.

Exercise changed/unchanged papers, source metadata, flat section breadcrumbs and
failure preservation without live embedding calls.
"""

import json

import pymupdf
import pytest

from app import document_processing, ingest, paper_store
from app.config import EMBED_DIM


@pytest.fixture(autouse=True)
def deterministic_pdf_pages(monkeypatch):
    """Keep store tests independent of model predictions; neural tests cover the model boundary."""

    def read_pages(path, **kwargs):
        """Read fixture text with source pages so Chroma tests still exercise real PDFs."""
        with pymupdf.open(path) as pdf:
            return [
                {
                    "metadata": {"page_number": n + 1},
                    "text": page.get_text("text"),
                    "page_boxes": [
                        {"class": "text", "pos": (0, len(page.get_text("text")))}
                    ],
                }
                for n, page in enumerate(pdf)
            ]

    monkeypatch.setattr(document_processing, "extract_text_pages", read_pages)


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
    store = paper_store.open_chunk_store()
    monkeypatch.setattr(ingest, "embed_document_chunks", vectors)
    assert ingest.index_research_paper(store, row)[0] == "indexed"
    old_ids = set(store.get()["ids"])
    monkeypatch.setattr(
        ingest,
        "embed_document_chunks",
        lambda texts: pytest.fail("Unchanged PDF was embedded"),
    )
    assert ingest.index_research_paper(store, row)[0] == "unchanged"
    row["title"] = "Renamed paper.pdf"
    monkeypatch.setattr(ingest, "embed_document_chunks", vectors)
    assert ingest.index_research_paper(store, row)[0] == "indexed"
    assert not old_ids.intersection(store.get()["ids"])
    assert paper_store.open_chunk_store().count() == store.count()


def test_embedding_failure_preserves_existing_paper(tmp_path, monkeypatch):
    row = make_paper(tmp_path)
    store = paper_store.open_chunk_store()
    monkeypatch.setattr(ingest, "embed_document_chunks", vectors)
    ingest.index_research_paper(store, row)
    before = store.get()["ids"]
    row["title"] = "Changed"

    def fail(texts):
        raise RuntimeError("Model unavailable")

    monkeypatch.setattr(ingest, "embed_document_chunks", fail)
    with pytest.raises(RuntimeError):
        ingest.index_research_paper(store, row)
    assert store.get()["ids"] == before


@pytest.mark.parametrize("extension", [".pdf", ".PDF"])
def test_manifest_ignores_non_pdf_and_rejects_missing_pdf(tmp_path, extension):
    manifest = tmp_path / "sources.json"
    row = make_paper(tmp_path)
    original = row.pop("_path")
    pdf_path = original.with_suffix(extension)
    pdf_path.write_bytes(original.read_bytes())
    row["path"] = str(pdf_path)
    manifest.write_text(json.dumps([row, {**row, "id": "other", "path": "notes.docx"}]))
    assert len(ingest.load_source_manifest(manifest)) == 1
    row["path"] = "missing.pdf"
    manifest.write_text(json.dumps([row]))
    with pytest.raises(ValueError, match="missing"):
        ingest.load_source_manifest(manifest)


def test_embedding_signature_mismatch_requires_rebuild(monkeypatch):
    paper_store.open_chunk_store()
    monkeypatch.setattr(paper_store, "EMBED_MODEL", "another-model")
    with pytest.raises(ValueError, match="rebuild"):
        paper_store.open_chunk_store()


def test_real_chroma_query_keeps_source_metadata(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from app import retrieval

    row = make_paper(tmp_path)
    store = paper_store.open_chunk_store()
    monkeypatch.setattr(ingest, "embed_document_chunks", vectors)
    ingest.index_research_paper(store, row)
    monkeypatch.setattr(
        retrieval, "embed_search_question", lambda query: vectors([query])[0]
    )
    monkeypatch.setattr(
        retrieval,
        "get_reranker_model",
        lambda: SimpleNamespace(predict=lambda pairs: [2.0] * len(pairs)),
    )
    result = retrieval.retrieve_relevant_passages("What method is used?")
    assert result
    assert all(item["source_url"] == row["source_url"] for item in result)
    assert all("page 1" in item["location"] for item in result)
    assert all(item["rerank_score"] == 2.0 for item in result)
    # The child ID and parent_id are sufficient to trace and expand the source.
    assert all("child_index" not in item for item in result)
    for item in result:
        assert item["id"] in store.get()["ids"]
        parent = paper_store.open_parent_section_store().get(ids=[item["parent_id"]])
        assert parent["metadatas"][0]["document_id"] == row["id"]


def test_explicit_reset_replaces_incompatible_collection(monkeypatch):
    paper_store.open_chunk_store()
    monkeypatch.setattr(paper_store, "EMBED_MODEL", "another-model")
    paper_store.reset_paper_stores()
    assert paper_store.open_chunk_store().count() == 0


def test_cleanup_removes_only_missing_paper_children_and_parents(tmp_path, monkeypatch):
    """One manifest deletion removes both levels while preserving the remaining paper."""

    row = make_paper(tmp_path)
    other = {**row, "id": "paper-2", "title": "Retained paper.pdf"}
    store = paper_store.open_chunk_store()
    monkeypatch.setattr(ingest, "embed_document_chunks", vectors)
    ingest.index_research_paper(store, row)
    ingest.index_research_paper(store, other)
    removed = len(store.get(where={"document_id": row["id"]})["ids"])
    assert ingest.remove_unlisted_papers(store, {other["id"]}) == removed
    for index in (store, paper_store.open_parent_section_store()):
        assert not index.get(where={"document_id": row["id"]})["ids"]
        assert index.get(where={"document_id": other["id"]})["ids"]


def test_extraction_failure_preserves_existing_paper(tmp_path, monkeypatch):
    """A broken extraction must not remove a previously searchable paper."""
    row = make_paper(tmp_path)
    store = paper_store.open_chunk_store()
    monkeypatch.setattr(ingest, "embed_document_chunks", vectors)
    ingest.index_research_paper(store, row)
    children_before = store.get()["ids"]
    parents_before = paper_store.open_parent_section_store().get()["ids"]
    row["title"] = "Changed paper"

    def fail(path, **kwargs):
        """Simulate the PDF library failing before any replacement is written."""
        raise RuntimeError("PDF extraction failed")

    monkeypatch.setattr(ingest, "extract_pdf_sections", fail)
    with pytest.raises(RuntimeError, match="PDF extraction failed"):
        ingest.index_research_paper(store, row)
    assert store.get()["ids"] == children_before
    assert paper_store.open_parent_section_store().get()["ids"] == parents_before


def test_flat_sections_keep_breadcrumbs_and_child_to_section_links(
    tmp_path, monkeypatch
):
    """Keep nested titles and expand a child into its own section, without a tree."""
    from app.document_processing import ExtractedSection

    row = make_paper(tmp_path)
    sections = [
        ExtractedSection(
            "page 1", "Method overview", "paper-section", "2 Methods", ("2 Methods",), 1
        ),
        ExtractedSection(
            "page 2",
            "Training details",
            "paper-section",
            "2.1 Training",
            ("2 Methods", "2.1 Training"),
            2,
        ),
    ]
    monkeypatch.setattr(
        ingest,
        "extract_pdf_sections",
        lambda path, **kwargs: ("sectioned_paper", sections),
    )
    monkeypatch.setattr(ingest, "embed_document_chunks", vectors)
    store = paper_store.open_chunk_store()
    ingest.index_research_paper(store, row)
    parents = paper_store.open_parent_section_store().get(
        include=["documents", "metadatas"]
    )
    training_section_id = next(
        key
        for key, text in zip(parents["ids"], parents["documents"])
        if text == "Training details"
    )
    meta = next(
        meta for meta in store.get()["metadatas"] if meta["heading"] == "2.1 Training"
    )
    assert meta["parent_id"] == training_section_id
    assert meta["heading_level"] == 2
    assert meta["section_path"] == "2 Methods > 2.1 Training"
    nested = next(
        meta for meta in parents["metadatas"] if meta["heading"] == "2.1 Training"
    )
    assert nested["section_path"] == "2 Methods > 2.1 Training"
    assert all("ancestor_section_id" not in saved for saved in store.get()["metadatas"])
    assert all("ancestor_section_id" not in saved for saved in parents["metadatas"])
    from app.retrieval import expand_parent_section_windows

    children = store.get(include=["documents", "metadatas"])
    training_child = next(
        {"id": key, "content": text, **saved}
        for key, text, saved in zip(
            children["ids"], children["documents"], children["metadatas"]
        )
        if saved["heading"] == "2.1 Training"
    )
    evidence = expand_parent_section_windows([training_child], limit=1)
    assert evidence[0]["content"] == "Training details"
    assert evidence[0]["section_path"] == "2 Methods > 2.1 Training"
