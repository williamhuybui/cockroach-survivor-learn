"""Check PDF classification, section boundaries, size limits and citations."""

from pathlib import Path

import pymupdf
import pytest

from app.document_processing import (
    MAX_CHUNK_CHARACTERS,
    ExtractedSection,
    chunk_sections,
    classify_document,
    clean_text,
    extract_sections,
    paper_sections,
)


def test_paper_section_continues_across_pages():
    pages = [
        ExtractedSection(
            "page 1",
            "Abstract\nSummary\n1 Introduction\nIntro\n2 Methods\nStart",
            "pdf-page",
        ),
        ExtractedSection("page 2", "Continuation\n3 Results\nFindings", "pdf-page"),
    ]
    sections = paper_sections(pages)
    method = next(section for section in sections if section.heading == "2 Methods")
    assert "Start\nContinuation" in method.text
    assert method.location == "page 1 to page 2"


def test_ordinary_list_is_not_a_paper():
    pages = [ExtractedSection("page 1", "1 Apples\n2 Oranges", "pdf-page")]
    assert classify_document(Path("list.pdf"), pages) == "pdf_page_fallback"


def test_paper_without_abstract_label_uses_other_structure_signals():
    pages = [
        ExtractedSection(
            "page 1",
            "Title\nSummary\n1. Introduction\nIntro\n2. Methods\nMethod\nReferences\nCitation",
            "pdf-page",
        )
    ]
    assert classify_document(Path("paper.pdf"), pages) == "sectioned_paper"


@pytest.mark.parametrize(
    "headings", ["1\nIntroduction\nIntro\n2\nMethods", "Introduction\nIntro\nMethods"]
)
def test_split_and_unnumbered_headings_are_recognized(headings):
    pages = [
        ExtractedSection("page 1", f"Abstract\nSummary\n{headings}\nBody", "pdf-page")
    ]
    assert classify_document(Path("paper.pdf"), pages) == "sectioned_paper"


def test_reference_year_is_not_a_heading():
    pages = [
        ExtractedSection(
            "page 1", "References\n2020. A study of models\nDetails", "pdf-page"
        )
    ]
    sections = paper_sections(pages)
    assert len(sections) == 1
    assert "2020. A study" in sections[0].text


def test_real_pdf_extraction_preserves_locations(tmp_path):
    path = tmp_path / "paper.pdf"
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        for index, line in enumerate(
            [
                "Title",
                "Abstract",
                "Summary",
                "1 Introduction",
                "Intro",
                "2 Methods",
                "Method",
            ]
        ):
            page.insert_text(
                (72, 72 + index * 16),
                line,
                fontname="hebo"
                if line in {"Abstract", "1 Introduction", "2 Methods"}
                else "helv",
            )
        pdf.save(path)
    kind, sections = extract_sections(path)
    assert kind == "sectioned_paper"
    assert any(
        section.heading == "2 Methods" and section.location == "page 1"
        for section in sections
    )


def test_chunks_keep_identity_and_size_limit():
    chunks = chunk_sections(
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


def test_no_text_produces_no_chunks():
    assert (
        chunk_sections([ExtractedSection("page 1", "", "pdf-page")], "Scan.pdf") == []
    )


def test_table_values_do_not_become_headings_when_font_is_plain():
    page = ExtractedSection(
        "page 1",
        "1 Introduction\nBody\n1.24\nBooks\nC. Appendix details\nMore",
        "pdf-page",
    )
    styles = {"page 1": {"1 Introduction", "C. Appendix details"}}
    sections = paper_sections([page], styles)
    assert [section.heading for section in sections] == [
        "1 Introduction",
        "C. Appendix details",
    ]
    assert "Books" in sections[0].text


def test_malformed_characters_are_cleaned():
    assert clean_text("model\x00 size \ud835") == "model size ?"
