"""Test the non-neural recovery path and caption-label recognition.

Ensure fallback avoids expensive reconstruction, retains exact caption pages and
carries uncertainty metadata into Chroma and grounding.
"""

import pymupdf
import pymupdf4llm

from app import document_processing, ingest, paper_store
from app.config import EMBED_DIM


def test_caption_labels_do_not_match_prose_mentions_or_code():
    text = (
        "## Results\n\nTable 2 shows an improvement.\n\n"
        "**Table 2:** Method comparison.\nCaption continuation.\n\n"
        "Figure A1. Architecture overview.\n\n"
        "```python\n\n# Not a heading\n\nTable 9: Example string\n\n```\n\n"
        "## Discussion"
    )
    boxes = document_processing.identify_markdown_blocks(text)
    captions = [b for b in boxes if b["class"] == "caption"]
    assert [b["reference_label"] for b in captions] == ["Table 2", "Figure A1"]
    assert "Caption continuation" in text[slice(*captions[0]["pos"])]
    assert sum(b["class"] == "section-header" for b in boxes) == 2


def test_real_fast_path_never_calls_layout_tables_or_image_rendering(
    tmp_path, monkeypatch
):
    """A real PDF exercises the library while expensive APIs are forbidden."""
    path = tmp_path / "paper.pdf"
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 72), "2 Methods", fontname="hebo", fontsize=11)
        page.insert_text((72, 105), "The method uses training examples.", fontsize=11)
        page = pdf.new_page()
        page.insert_text((72, 72), "Table 2: Method comparison.", fontsize=11)
        pdf.save(path)

    def forbidden(*args, **kwargs):
        """Fail if the fast path accidentally activates neural/visual extraction."""
        raise AssertionError("Expensive extraction was called")

    for name in ("get_layout", "find_tables", "get_pixmap", "get_drawings"):
        monkeypatch.setattr(pymupdf.Page, name, forbidden)
    with pymupdf.open(path) as pdf:
        page = document_processing.extract_fast_page(pdf, 1)
    assert page["metadata"]["page_number"] == 2
    caption = next(
        b for b in page["page_boxes"] if b.get("reference_label") == "Table 2"
    )
    assert "Method comparison" in page["text"][slice(*caption["pos"])]


def test_library_options_skip_table_reconstruction(tmp_path, monkeypatch):
    path = tmp_path / "paper.pdf"
    with pymupdf.open() as pdf:
        pdf.new_page()
        pdf.save(path)

    def extract(pdf, **options):
        """Check the actual legacy API flags, not flags ignored by layout mode."""
        assert options["table_strategy"] is None
        assert options["detect_bg_color"] is False
        assert options["ignore_images"] and options["ignore_graphics"]
        assert options["pages"] == [0]
        return [{"metadata": {"page": 1}, "text": "Table 1: Overview."}]

    monkeypatch.setattr(pymupdf4llm, "to_markdown", extract)
    with pymupdf.open(path) as pdf:
        pages = [document_processing.extract_fast_page(pdf, 0)]
    assert pages[0]["metadata"]["page_number"] == 1
    assert pages[0]["page_boxes"][0]["reference_label"] == "Table 1"


def test_caption_metadata_reaches_chroma_and_progress_is_visible(
    tmp_path, monkeypatch, capsys
):
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"fixture")
    sections = [
        document_processing.ExtractedSection(
            "page 7",
            "Table 2: Method comparison.",
            "pdf-caption",
            heading="Table 2",
            reference_label="Table 2",
            extraction_method="fast_fallback",
            structure_confidence="uncertain",
            review_reason="page layout timed out",
        )
    ]
    monkeypatch.setattr(
        ingest,
        "extract_pdf_sections",
        lambda path, **kwargs: ("pdf_page_fallback", sections),
    )
    monkeypatch.setattr(
        ingest,
        "embed_document_chunks",
        lambda texts: [[1.0] + [0.0] * (EMBED_DIM - 1) for _ in texts],
    )
    store = paper_store.open_chunk_store()
    row = {
        "id": "paper",
        "title": "Paper",
        "source_url": "https://example.org/paper",
        "_path": path,
    }
    ingest.index_research_paper(store, row)
    meta = store.get()["metadatas"][0]
    assert meta["evidence_type"] == "caption" and meta["reference_label"] == "Table 2"
    assert meta["page_start"] == meta["page_end"] == 7
    parent = paper_store.open_parent_section_store().get()["metadatas"][0]
    for saved in (meta, parent):
        assert saved["structure_confidence"] == "uncertain"
        assert saved["extraction_method"] == "fast_fallback"
        assert saved["review_reason"] == "page layout timed out"
    output = capsys.readouterr().out
    assert all(stage in output for stage in ("EXTRACT", "EMBED", "STORE", "total"))
    assert ingest.index_research_paper(store, row)[0] == "unchanged"
    monkeypatch.setattr(ingest, "EXTRACTION_PROFILE", "changed-extractor")
    assert ingest.index_research_paper(store, row)[0] == "indexed"


def test_page_numbers_and_heading_body_on_same_paragraph_keep_correct_boundaries():
    text = (
        "## 3 Methods\nBody text continues here.\n\n## 3\n\n## 3.1 Training\nMore text."
    )
    boxes = document_processing.identify_markdown_blocks(text, page_number=3)
    headings = [
        text[slice(*b["pos"])].strip() for b in boxes if b["class"] == "section-header"
    ]
    assert headings == ["## 3 Methods", "## 3.1 Training"]
    assert any(b["class"] == "page-footer" for b in boxes)
    assert any(
        "Body text continues" in text[slice(*b["pos"])]
        for b in boxes
        if b["class"] == "text"
    )


def test_grounded_prompt_distinguishes_captions_from_unread_cells(monkeypatch):
    from app import grounding

    def generate(schema, instruction, evidence, request):
        """Check the shared answer instructions without calling a model."""
        assert "never claim to have read its cells" in instruction
        assert "Do not infer" in instruction
        assert "structure_confidence=uncertain" in instruction
        return schema(answers=[])

    monkeypatch.setattr(grounding, "generate_from_evidence", generate)
    grounding.generate_grounded_draft("What value is in Table 2?", [], [])
