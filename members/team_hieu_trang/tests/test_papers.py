"""Test text cleanup, section boundaries, captions and child-chunk metadata.

Use small PDF/block fixtures to check breadcrumbs, page spans, fallback isolation
and separation of section text.
"""

import pymupdf
import pytest

from app import document_processing
from app.document_processing import (
    MAX_CHUNK_CHARACTERS,
    ExtractedSection,
    clean_pdf_text,
    extract_pdf_sections,
    group_layout_pages_into_sections,
    read_detected_heading,
    split_sections_into_chunks,
)


def layout_page(number, blocks):
    """Build the library's page/offset format without running its model in unit tests."""

    text = ""
    boxes = []
    for kind, content in blocks:
        start = len(text)
        text += content + "\n\n"
        boxes.append({"class": kind, "pos": (start, len(text))})
    return {"metadata": {"page_number": number}, "text": text, "page_boxes": boxes}


def test_sections_continue_across_pages_and_keep_unfamiliar_headings():
    pages = [
        layout_page(
            1,
            [
                ("text", "Authors and opening text"),
                ("section-header", "## A. Unfamiliar Training Recipe"),
                ("text", "Start of recipe"),
            ],
        ),
        layout_page(
            2,
            [
                ("text", "Continuation"),
                ("section-header", "### A.1 Why does this work?"),
                ("text", "Explanation"),
            ],
        ),
    ]
    sections = group_layout_pages_into_sections(pages)
    assert sections[0].heading == ""
    assert sections[0].text == "Authors and opening text"
    assert sections[1].heading == "A. Unfamiliar Training Recipe"
    assert sections[1].location == "page 1 to page 2"
    assert "Start of recipe\n\nContinuation" in sections[1].text
    assert sections[2].heading == "A.1 Why does this work?"
    assert sections[2].location == "page 2"


def test_tables_lists_and_code_do_not_become_section_boundaries():
    page = layout_page(
        3,
        [
            ("section-header", "## Results"),
            ("list-item", "1. An ordinary list item"),
            ("table", "| Model | Score |\n| A | 2020 |"),
            ("text", "```\n# a code comment\n```"),
            ("section-header", "## Appendix"),
            ("text", "Extra results"),
        ],
    )
    sections = group_layout_pages_into_sections([page])
    assert [section.heading for section in sections] == ["Results", "Appendix"]
    assert "| A | 2020 |" not in sections[0].text
    assert "# a code comment" in sections[0].text


def test_pdf_without_section_headings_falls_back_to_pages(tmp_path, monkeypatch):
    pages = [
        layout_page(1, [("title", "# Paper title"), ("text", "Body")]),
        layout_page(2, [("text", "More body")]),
    ]
    monkeypatch.setattr(
        document_processing, "extract_text_pages", lambda *a, **kw: pages
    )
    kind, sections = extract_pdf_sections(tmp_path / "paper.pdf")
    assert kind == "pdf_page_fallback"
    assert [section.location for section in sections] == ["page 1", "page 2"]
    assert all(section.strategy == "pdf-page" for section in sections)
    assert "Paper title" in sections[0].text


def test_detected_section_needs_no_abstract_or_known_name(tmp_path, monkeypatch):
    pages = [layout_page(1, [("section-header", "## New Topic"), ("text", "Body")])]

    def extract(path, **kwargs):
        """Return an already detected heading without requiring familiar paper labels."""
        return pages

    monkeypatch.setattr(document_processing, "extract_text_pages", extract)
    kind, sections = extract_pdf_sections(tmp_path / "paper.pdf")
    assert kind == "sectioned_paper"
    assert sections[0].heading == "New Topic"


def test_blank_pages_are_reported_and_do_not_extend_section_span(
    tmp_path, monkeypatch, capsys
):
    pages = [
        layout_page(1, [("section-header", "## Methods"), ("text", "Details")]),
        layout_page(2, []),
    ]
    monkeypatch.setattr(
        document_processing, "extract_text_pages", lambda *a, **kw: pages
    )
    _, sections = extract_pdf_sections(tmp_path / "scan.pdf")
    assert "page 2: no text" in capsys.readouterr().out
    assert sections[0].location == "page 1"


def test_extraction_errors_propagate(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        """Simulate a parser failure so ingestion can preserve the old index."""
        raise RuntimeError("Broken PDF")

    monkeypatch.setattr(document_processing, "extract_text_pages", fail)
    with pytest.raises(RuntimeError, match="Broken PDF"):
        extract_pdf_sections(tmp_path / "paper.pdf")


def test_non_pdf_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Only research PDF"):
        extract_pdf_sections(tmp_path / "notes.docx")


def test_real_pdf_extraction_preserves_text_and_page_locations(tmp_path):
    """Run the installed fast extractor; exact heading predictions belong to corpus review."""
    path = tmp_path / "paper.pdf"
    with pymupdf.open() as pdf:
        for body in (
            "Training uses examples from the dataset.",
            "Evaluation measures held out accuracy.",
        ):
            page = pdf.new_page()
            page.insert_text((72, 72), "Research Methods", fontsize=20, fontname="hebo")
            page.insert_text((72, 120), body, fontsize=11)
        pdf.save(path)
    kind, sections = extract_pdf_sections(path)
    assert kind in {"sectioned_paper", "pdf_page_fallback"}
    text = " ".join(section.text for section in sections)
    assert "Training uses examples" in text
    assert "Evaluation measures" in text
    assert all(section.location.startswith("page ") for section in sections)
    assert any("2" in section.location for section in sections)


def test_chunks_keep_identity_and_size_limit():
    chunks = split_sections_into_chunks(
        [ExtractedSection("page 3", "word " * 1500, "paper-section", "2 Methods")],
        "RAG.pdf",
    )
    assert len(chunks) > 1
    assert all(
        chunk.page_content.startswith("Paper: RAG.pdf\nSection: 2 Methods")
        for chunk in chunks
    )
    assert all(len(chunk.page_content) <= MAX_CHUNK_CHARACTERS for chunk in chunks)
    assert all("page 3" in chunk.metadata["location"] for chunk in chunks)
    assert all(chunk.metadata["section_index"] == 0 for chunk in chunks)


def test_no_text_produces_no_chunks():
    assert (
        split_sections_into_chunks(
            [ExtractedSection("page 1", "", "pdf-page")], "Scan.pdf"
        )
        == []
    )


def test_malformed_characters_are_cleaned():
    assert clean_pdf_text("model\x00 size \ud835") == "model size ?"


def test_numbered_headings_keep_breadcrumbs_and_separate_section_text():
    page = layout_page(
        1,
        [
            ("section-header", "## 2 Methods"),
            ("text", "General method"),
            ("section-header", "## 2.1 Training"),
            ("text", "Training details"),
            ("section-header", "## 2.1.1 Optimizer"),
            ("text", "Optimizer details"),
            ("section-header", "## 2.2 Evaluation"),
            ("text", "Evaluation details"),
            ("section-header", "## 3 Results"),
            ("text", "Findings"),
            ("section-header", "## References"),
            ("text", "Bibliography"),
        ],
    )
    sections = group_layout_pages_into_sections([page])
    assert sections[2].heading_path == ("2 Methods", "2.1 Training", "2.1.1 Optimizer")
    assert sections[3].heading_path == ("2 Methods", "2.2 Evaluation")
    assert sections[4].heading_path == ("3 Results",)
    assert sections[5].heading_path == ("References",)
    assert "Training details" not in sections[0].text
    assert "Optimizer details" not in sections[1].text
    assert "Evaluation details" not in sections[2].text
    children = split_sections_into_chunks(sections, "Paper")
    child = next(c for c in children if c.metadata["section_index"] == 2)
    assert (
        child.metadata["section_path"] == "2 Methods > 2.1 Training > 2.1.1 Optimizer"
    )
    assert "2 Methods > 2.1 Training" in child.page_content
    assert "2 Methods > 2.1 Training" in child.metadata["location"]


def test_unnumbered_heading_levels_and_missing_numbered_parent():
    page = layout_page(
        1,
        [
            ("title", "# Paper title"),
            ("section-header", "## Methods"),
            ("text", "General"),
            ("section-header", "### Training"),
            ("text", "Training details"),
            ("section-header", "## Results"),
            ("text", "Findings"),
            ("section-header", "## 4.1 Missing parent"),
            ("text", "Extra"),
        ],
    )
    sections = group_layout_pages_into_sections([page])
    assert sections[0].heading == ""
    assert sections[2].heading_path == ("Methods", "Training")
    assert sections[-1].heading_path == ("4.1 Missing parent",)


def test_repeated_continuation_heading_does_not_duplicate_a_section():
    pages = [
        layout_page(1, [("section-header", "## 2 Methods"), ("text", "Start")]),
        layout_page(2, [("section-header", "## 2 Methods"), ("text", "Continuation")]),
    ]
    sections = group_layout_pages_into_sections(pages)
    assert len(sections) == 1
    assert sections[0].location == "page 1 to page 2"
    assert sections[0].text.count("2 Methods") == 1
    assert "Continuation" in sections[0].text


@pytest.mark.parametrize("with_heading", [True, False])
def test_visuals_equations_tables_and_page_furniture_are_excluded(
    tmp_path, monkeypatch, with_heading
):
    blocks = [("section-header", "## Methods")] if with_heading else []
    blocks += [
        ("text", "Useful paragraph"),
        ("picture", "Secret chart text"),
        ("formula", "Secret equation"),
        ("table", "Secret cells"),
        ("caption", "Secret caption"),
        ("page-header", "Secret running header"),
        ("page-footer", "Secret footer"),
        ("list-item", "- Useful list item"),
    ]
    pages = [layout_page(1, blocks)]
    monkeypatch.setattr(
        document_processing, "extract_text_pages", lambda *a, **kw: pages
    )
    _, sections = extract_pdf_sections(tmp_path / "paper.pdf")
    text = " ".join(s.text for s in sections)
    assert "Useful paragraph" in text and "Useful list item" in text
    assert "Secret caption" in text
    assert all(
        item not in text
        for item in (
            "Secret chart",
            "Secret equation",
            "Secret cells",
            "Secret running header",
            "Secret footer",
        )
    )


def test_text_cleanup_preserves_meaningful_hyphens_and_numbers():
    assert (
        clean_pdf_text("learn\u00ad\ning ef\ufb01cient well-known 1.25 -0.5")
        == "learning efficient well-known 1.25 -0.5"
    )
    assert read_detected_heading("### **2.1 Training**") == ("2.1 Training", 2)
    assert read_detected_heading("### **Training**") == ("Training", 2)
    assert read_detected_heading("## <u>A.1 Extra details</u>") == (
        "A.1 Extra details",
        2,
    )


def test_missing_layout_is_reported_instead_of_indexing_unfiltered_text(
    tmp_path, monkeypatch
):
    pages = [
        {"metadata": {"page_number": 1}, "text": "Unknown content", "page_boxes": []}
    ]
    monkeypatch.setattr(
        document_processing, "extract_text_pages", lambda *a, **kw: pages
    )
    with pytest.raises(ValueError, match="missing layout blocks"):
        extract_pdf_sections(tmp_path / "paper.pdf")


@pytest.mark.parametrize("with_heading", [True, False])
def test_captions_have_exact_pages_and_never_inherit_section_spans(
    tmp_path, monkeypatch, with_heading
):
    pages = [
        layout_page(1, [("section-header", "## Results"), ("text", "Finding")])
        if with_heading
        else layout_page(1, [("text", "Finding")]),
        layout_page(
            2,
            [("text", "Continuation"), ("caption", "Table 2: Comparison of methods.")],
        ),
    ]
    pages[1]["page_boxes"][-1]["reference_label"] = "Table 2"
    monkeypatch.setattr(
        document_processing, "extract_text_pages", lambda path, **kwargs: pages
    )
    _, sections = extract_pdf_sections(tmp_path / "paper.pdf")
    caption = next(s for s in sections if s.reference_label)
    assert caption.location == "page 2"
    assert caption.text == "Table 2: Comparison of methods."
    assert all(
        "Comparison of methods" not in s.text for s in sections if not s.reference_label
    )
    child = next(
        c
        for c in split_sections_into_chunks(sections, "Paper")
        if c.metadata["reference_label"]
    )
    assert child.metadata["evidence_type"] == "caption"
    assert child.metadata["location"].startswith("page 2: Table 2")


def test_child_indices_reset_for_each_section():
    sections = [
        ExtractedSection(
            "page 1", "word " * 1500,
            "paper-section", "1 Introduction"
        ),
        ExtractedSection(
            "page 3", "word " * 1500,
            "paper-section", "2 Methods"
        ),
    ]

    chunks = chunk_sections(sections, "RAG.pdf")

    intro = [
        chunk for chunk in chunks
        if chunk.metadata["section_index"] == 0
    ]

    methods = [
        chunk for chunk in chunks
        if chunk.metadata["section_index"] == 1
    ]

    assert len(intro) > 1
    assert len(methods) > 1

    assert [c.metadata["child_index"] for c in intro] == list(range(len(intro)))
    assert [c.metadata["child_index"] for c in methods] == list(range(len(methods)))


def test_whitespace_only_section_produces_no_chunks():
    ### This prevents whitespace-only chunks from reaching the embedding pipeline.
    sections = [
        ExtractedSection("page 1", "   \n  ", "pdf-page")
    ]

    assert chunk_sections(sections, "Scan.pdf") == []


def test_actual_research_paper():
    pdf_files = list(
        Path("synced_sources/drive").rglob("*Attention*All*Need*.pdf")
    )

    assert pdf_files, "Attention Is All You Need PDF not found"

    path = pdf_files[0]

    document_type, sections = extract_sections(path)
    chunks = chunk_sections(sections, path.stem)

    print("\nDocument type:", document_type)
    print("Total sections:", len(sections))
    print("Total chunks:", len(chunks))

    print("\nDETECTED SECTIONS:")
    for section in sections:
        print(section.heading, "|", section.location)

    print("\nFIRST FIVE CHUNKS:")
    for index, chunk in enumerate(chunks[:5]):
        print(f"\n--- Chunk {index} ---")
        print("Location:", chunk.metadata["location"])
        print("Child index:", chunk.metadata["child_index"])
        print("Length:", len(chunk.page_content))
        print(chunk.page_content)

    assert len(chunks) > 0
    assert all(
        len(chunk.page_content) <= MAX_CHUNK_CHARACTERS
        for chunk in chunks
    )