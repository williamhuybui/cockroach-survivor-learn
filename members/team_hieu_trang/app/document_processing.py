"""Convert local research PDFs into sections and searchable child chunks.

Read extract_pdf_sections() first: it extracts pages, groups text under headings and
saves captions with exact pages. Ingest.py then calls split_sections_into_chunks()
before embedding. The separate neural_pdf_extraction.py file manages the model
worker.
"""

import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import (
    CHUNK_OVERLAP_CHARACTERS,
    MAX_CHUNK_CHARACTERS,
    PDF_EXTRACTION_REPORT_PATH,
)
from .neural_pdf_extraction import NeuralPageReader

EXTRACTION_PROFILE = "neural-regions-v3-1.28.2"

# Recognize explicit caption labels such as "Table 2:" or "Figure A1.".
# Prose such as "Table 2 shows an improvement" is not treated as a caption.
CAPTION_LABEL_PATTERN = re.compile(
    r"^(?:\*\*)?(Table|Figure|Fig\.)\s+([A-Z]?\d+(?:\.\d+)*|[IVXLCDM]+)"
    r"(?:\*\*)?(?:\s*[:.]|\s*$)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ExtractedSection:
    """Store one section or caption before it is split into searchable chunks.

    Text belongs only to this record. Location identifies its PDF page span;
    heading_path is a breadcrumb such as Methods > Training. Extraction fields tell
    retrieval whether structure came from the model or uncertain page-based
    fallback.
    """

    location: str
    text: str
    strategy: str
    heading: str = ""
    heading_path: tuple[str, ...] = ()
    heading_level: int = 0
    reference_label: str = ""
    extraction_method: str = "unspecified"
    structure_confidence: str = "unknown"
    review_reason: str = ""


# These blocks do not belong in section prose; captions are saved separately below.
EXCLUDED_BLOCK_TYPES = {
    "picture",
    "formula",
    "table",
    "caption",
    "page-header",
    "page-footer",
}


def extract_pdf_sections(path: Path, reader=None) -> tuple[str, list[ExtractedSection]]:
    """Read a PDF and return its detected structure type and section records.

    Use detected headings to group text across pages; otherwise keep text grouped by
    page. Save captions separately so their citations point to an exact page. Failed
    neural pages use uncertain page-based text, while unreadable PDFs raise errors
    for ingestion to handle.
    """

    if path.suffix.lower() != ".pdf":
        raise ValueError("Only research PDF files are supported")
    pages = extract_text_pages(path, reader=reader)
    for page in pages:
        number = page["metadata"]["page_number"]
        if not page["text"].strip():
            print(f"WARNING {path.name}, page {number}: no text; OCR is not configured")
        elif "\ufffd" in page["text"]:
            print(
                f"WARNING {path.name}, page {number}: unreadable characters; review extraction"
            )
        if page["text"].strip() and not page["page_boxes"]:
            raise ValueError(f"{path.name}, page {number}: missing layout blocks")
    has_sections = any(
        box["class"] == "section-header" and box["pos"][1] > box["pos"][0]
        for page in pages
        if page["metadata"].get("extraction_method") != "fast_fallback"
        for box in page["page_boxes"]
    )
    if has_sections:
        sections = group_layout_pages_into_sections(pages)
    else:
        sections = []
        for page in pages:
            # Keep captions separate so fallback pages also have exact caption locations.
            text = "\n\n".join(
                clean_pdf_text(page["text"][slice(*box["pos"])]).strip()
                for box in page["page_boxes"]
                if box["class"] not in EXCLUDED_BLOCK_TYPES
            ).strip()
            sections.append(
                ExtractedSection(
                    f"page {page['metadata']['page_number']}",
                    text,
                    "pdf-page",
                    extraction_method=page["metadata"].get(
                        "extraction_method", "unspecified"
                    ),
                    structure_confidence=page["metadata"].get(
                        "structure_confidence", "unknown"
                    ),
                    review_reason=page["metadata"].get("review_reason", ""),
                )
            )
    # Captions get their own records so their citation points to one exact page,
    # rather than inheriting the containing section's multi-page span.
    for page in pages:
        for box in page["page_boxes"]:
            if box["class"] == "caption":
                label = box.get("reference_label", "")
                sections.append(
                    ExtractedSection(
                        location=f"page {page['metadata']['page_number']}",
                        text=clean_pdf_text(page["text"][slice(*box["pos"])]).strip(),
                        strategy="pdf-caption",
                        heading=label or "Caption",
                        reference_label=label,
                        extraction_method=page["metadata"].get(
                            "extraction_method", "unspecified"
                        ),
                        structure_confidence=page["metadata"].get(
                            "structure_confidence", "unknown"
                        ),
                        review_reason=page["metadata"].get("review_reason", ""),
                    )
                )
    return "sectioned_paper" if has_sections else "pdf_page_fallback", sections


def extract_text_pages(
    path: Path, reader: NeuralPageReader | None = None
) -> list[dict]:
    """Extract text and region metadata for every page, then save a coverage report.

    Reuse a supplied neural worker across papers, or close a worker created for this
    call. Recover from extraction failures or missing text with page-based fallback.
    Slow work only prints warnings; each page records how it was extracted and why
    fallback was needed.
    """

    owned_reader = reader is None
    reader = reader or NeuralPageReader()
    started = perf_counter()
    pages = []
    try:
        with pymupdf.open(path) as pdf:
            for number in range(len(pdf)):
                page_started = perf_counter()
                reason = ""
                try:
                    print(
                        f"EXTRACT {path.name}: page {number + 1}/{len(pdf)} starting neural extraction",
                        file=sys.stderr,
                        flush=True,
                    )
                    page = reader.read_page(path, number)
                    if (
                        not page["text"].strip()
                        and pdf[number].get_text("text").strip()
                    ):
                        raise RuntimeError(
                            "neural regions omitted all available page text"
                        )
                except (RuntimeError, EOFError, OSError) as error:
                    reason = f"{type(error).__name__}: {error}"
                    page = extract_fast_page(pdf, number)
                method = "fast_fallback" if reason else "neural_regions"
                page["metadata"].update(
                    {
                        "extraction_method": method,
                        "structure_confidence": "uncertain"
                        if reason
                        else "model_detected",
                        "review_reason": reason,
                        "seconds": round(perf_counter() - page_started, 3),
                        "replacement_characters": page["text"].count("\ufffd"),
                    }
                )
                # Both neural captions and explicit caption text retain their labels.
                for box in page["page_boxes"]:
                    start, stop = box["pos"]
                    content = page["text"][start:stop].strip().lstrip("# ")
                    label = CAPTION_LABEL_PATTERN.match(content)
                    if label and box["class"] in {"text", "caption"}:
                        box["class"] = "caption"
                        prefix = "Table" if label[1].lower() == "table" else "Figure"
                        box["reference_label"] = f"{prefix} {label[2]}"
                pages.append(page)
                print(
                    f"EXTRACT {path.name}: page {number + 1}/{len(pdf)} {method} "
                    f"({perf_counter() - page_started:.2f}s)"
                    + (f" — {reason}" if reason else ""),
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        if owned_reader:
            reader.close()
    # Larger detected heading fonts suggest shallower levels (Methods vs Training).
    # Explicit numbering such as 2.1 overrides this hint when building breadcrumbs.
    sizes = sorted(
        {
            box.get("font_size", 0)
            for page in pages
            for box in page["page_boxes"]
            if box["class"] == "section-header" and box.get("font_size", 0) > 0
        },
        reverse=True,
    )
    for page in pages:
        for box in page["page_boxes"]:
            if box["class"] == "section-header" and box.get("font_size") in sizes:
                box["heading_level"] = min(6, sizes.index(box["font_size"]) + 1)
    save_extraction_report(path, pages, perf_counter() - started)
    fallback_count = sum(
        page["metadata"]["extraction_method"] == "fast_fallback" for page in pages
    )
    print(
        f"EXTRACT {path.name}: {len(pages) - fallback_count}/{len(pages)} neural pages; "
        f"{fallback_count} fallback pages need structure review",
        file=sys.stderr,
        flush=True,
    )
    return pages


def styled_heading_lines(page) -> set[str]:
    """Find bold or larger PDF lines so table numbers do not become headings."""

    lines = [
        line
        for block in page.get_text("dict")["blocks"]
        for line in block.get("lines", [])
    ]
    sizes = Counter()
    for line in lines:
        for span in line["spans"]:
            sizes[round(span["size"], 1)] += len(span["text"])
    body_size = sizes.most_common(1)[0][0] if sizes else 0
    # PyMuPDF uses bit 16 for bold. Weighting font sizes by text length finds
    # the body style rather than a large title that occurs only once.
    return {
        clean_text("".join(span["text"] for span in line["spans"])).strip()
        for line in lines
        if any(
            span["flags"] & 16 or span["size"] > body_size + 0.5
            for span in line["spans"]
        )
    }


def paper_sections(
    pages: list[ExtractedSection], styles: dict[str, set[str]] | None = None
) -> list[ExtractedSection]:
    """Keep a paper section together across pages and record its page span."""

    sections = []
    heading = ""
    level = 0
    heading_path = ()
    heading_stack = []
    parts = []
    page_numbers = []

    def save_current_section() -> None:
        """Save the text collected so far with its heading breadcrumb and page span.

        Called before a new heading, before fallback, and at the end so text is not
        lost between section boundaries.
        """

        if not parts:
            return
        first, last = page_numbers[0], page_numbers[-1]
        location = f"page {first}" if first == last else f"page {first} to page {last}"
        sections.append(
            ExtractedSection(
                location=location,
                text="\n\n".join(parts),
                strategy="paper-section",
                heading=heading,
                heading_path=heading_path,
                heading_level=level,
                extraction_method="neural_regions",
                structure_confidence="model_detected",
            )
        )

    for page in pages:
        number = page["metadata"]["page_number"]
        text = page["text"]
        if page["metadata"].get("extraction_method") == "fast_fallback":
            # An uncertain page cannot extend a neural section or its breadcrumb.
            save_current_section()
            body = "\n\n".join(
                clean_pdf_text(text[slice(*box["pos"])]).strip()
                for box in page["page_boxes"]
                if box["class"] not in EXCLUDED_BLOCK_TYPES
            ).strip()
            sections.append(
                ExtractedSection(
                    location=f"page {number}",
                    text=body,
                    strategy="pdf-page-fallback",
                    extraction_method="fast_fallback",
                    structure_confidence="uncertain",
                    review_reason=page["metadata"].get("review_reason", ""),
                )
            )
            heading, level, heading_path = "", 0, ()
            heading_stack, parts, page_numbers = [], [], []
            continue
        for box in page["page_boxes"]:
            if box["class"] in EXCLUDED_BLOCK_TYPES:
                continue
            start, stop = box["pos"]
            content = clean_pdf_text(text[start:stop]).strip()
            if not content:
                continue
            # Keep the paper title as opening text, not a heading above every section.
            if box["class"] == "section-header":
                new_heading, new_level = read_detected_heading(
                    content, box.get("heading_level")
                )
                # Ignore a repeated continuation heading on a later page of this section.
                if (
                    new_heading == heading
                    and new_level == level
                    and page_numbers
                    and number > page_numbers[-1]
                ):
                    continue
                save_current_section()
                # Replace the previous heading at this depth, including its subheadings.
                while heading_stack and heading_stack[-1]["level"] >= new_level:
                    heading_stack.pop()
                # A numbered subsection must not attach to an unrelated number.
                # For example, 3.1 cannot belong to 2 Methods if 3 was missed.
                label = new_heading.partition(" ")[0].rstrip(".")
                if new_level > 1 and label.count(".") == new_level - 1:
                    while heading_stack and not label.startswith(
                        heading_stack[-1]["title"].partition(" ")[0].rstrip(".") + "."
                    ):
                        heading_stack.pop()
                heading_stack.append({"level": new_level, "title": new_heading})
                heading_path = tuple(item["title"] for item in heading_stack)
                heading, level = new_heading, new_level
                parts, page_numbers = [], []
            parts.append(content)
            page_numbers.append(number)
    save_current_section()
    return sections

def extract_sections(path: Path) -> tuple[str, list[ExtractedSection]]:
    """Read every PDF page; use section boundaries when confidently detected."""

    if path.suffix.lower() != ".pdf":
        raise ValueError("Only research PDF files are supported")
    pages = []
    styles = {}
    with pymupdf.open(path) as pdf:
        for number, page in enumerate(pdf, 1):
            # Preserve the PDF's native order: geometric sorting can interleave columns.
            text = clean_text(page.get_text("text")).strip()
            if not text:
                print(
                    f"WARNING {path.name}, page {number}: no text; OCR is not configured"
                )
            pages.append(ExtractedSection(f"page {number}", text, "pdf-page"))
            styles[f"page {number}"] = styled_heading_lines(page)
    document_type = classify_document(path, pages)
    if document_type == "sectioned_paper":
        sections = paper_sections(pages, styles)
        main_sections = [
            section
            for section in sections
            if section.heading.casefold()
            not in SPECIAL_HEADINGS | {"title and authors"}
        ]
        if len(main_sections) >= 2:
            return document_type, sections
    return "pdf_page_fallback", pages


def chunk_sections(sections: list[ExtractedSection], title: str) -> list[Document]:
    """Repeat paper title and heading in each bounded, overlapping chunk."""

    chunks = []
    for section_index, section in enumerate(sections):
        ## The .strip() method removes leading and trailing whitespace.
        if not section.text.strip():
            continue
        prefix = f"Paper: {title}\n"
        section_path = " > ".join(section.heading_path) or section.heading
        if section_path:
            prefix += f"Section: {section_path}\n"
        budget = MAX_CHUNK_CHARACTERS - len(prefix)
        if budget <= CHUNK_OVERLAP_CHARACTERS:
            raise ValueError(
                "Paper title and section path are too long for the chunk size"
            )
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=budget,
            chunk_overlap=CHUNK_OVERLAP_CHARACTERS,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        pieces = splitter.split_text(section.text)
        for number, piece in enumerate(pieces, 1):
            location = section.location
            if section_path:
                location += f": {section_path}"
            if len(pieces) > 1:
                location += f", part {number}"
            chunks.append(
                Document(
                    page_content=prefix + piece,
                    metadata={
                        "location": location,
                        "strategy": section.strategy,
                        "heading": section.heading,
                        "section_path": section_path,
                        "heading_level": section.heading_level,
                        "reference_label": section.reference_label,
                        "extraction_method": section.extraction_method,
                        "structure_confidence": section.structure_confidence,
                        "review_reason": section.review_reason,
                        "evidence_type": "caption"
                        if section.strategy == "pdf-caption"
                        else "prose",
                        "section_index": section_index,
                    },
                )
            )
    return chunks
    