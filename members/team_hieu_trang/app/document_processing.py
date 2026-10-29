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


def extract_fast_page(pdf, number: int) -> dict:
    """Recover a failed neural page using the PDF's existing text layer.

    Ask PyMuPDF4LLM for text without neural layout, table reconstruction or image
    processing. Heading guesses are disabled because this page will be stored
    without section structure. Raw table text may remain; explicit caption labels
    can still be recognized.
    """

    import pymupdf4llm

    pymupdf4llm.use_layout(False)
    page = pymupdf4llm.to_markdown(
        pdf,
        pages=[number],
        hdr_info=False,
        page_chunks=True,
        detect_bg_color=False,
        ignore_images=True,
        ignore_graphics=True,
        table_strategy=None,
        show_progress=False,
    )[0]
    page["metadata"]["page_number"] = number + 1
    page["page_boxes"] = identify_markdown_blocks(page["text"], number + 1)
    return page


def identify_markdown_blocks(text: str, page_number: int | None = None) -> list[dict]:
    """Mark body text, explicit captions and heading lines in extracted Markdown.

    Return character ranges into the unchanged text so later code can read each
    block. Protect fenced code from being mistaken for headings or captions and
    ignore numeric page headings. This supplies block metadata for the non-neural
    extraction path.
    """

    boxes = []
    position = 0
    in_code = False
    for paragraph in text.split("\n\n"):
        content = paragraph.strip()
        first_line = content.split("\n", 1)[0].lstrip("# ")
        caption = CAPTION_LABEL_PATTERN.match(first_line) if not in_code else None
        if caption and not content.startswith(("```", "~~~")):
            prefix = "Table" if caption[1].lower() == "table" else "Figure"
            boxes.append(
                {
                    "class": "caption",
                    "pos": (position, position + len(paragraph)),
                    "reference_label": f"{prefix} {caption[2]}",
                }
            )
        else:
            # Separate each heading line from body lines, even without a blank separator.
            start = position
            for line in paragraph.splitlines(keepends=True):
                stripped = line.strip()
                kind = "text"
                if (
                    not in_code
                    and stripped.startswith("#")
                    and stripped.lstrip("#").startswith(" ")
                ):
                    title = stripped.lstrip("# ").replace("**", "").strip()
                    if page_number is not None and title == str(page_number):
                        kind = "page-footer"
                    elif (
                        any(char.isalpha() for char in title)
                        and len(title.split()) <= 16
                        and len(title) <= 160
                        and not title.endswith((".", ",", ";"))
                    ):
                        kind = (
                            "title"
                            if page_number == 1
                            and not boxes
                            and stripped.startswith("# ")
                            else "section-header"
                        )
                if stripped:
                    # Consecutive body lines form one paragraph so wrapping is preserved.
                    if (
                        kind == "text"
                        and boxes
                        and boxes[-1]["class"] == "text"
                        and boxes[-1]["pos"][1] == start
                    ):
                        boxes[-1]["pos"] = (boxes[-1]["pos"][0], start + len(line))
                    else:
                        boxes.append({"class": kind, "pos": (start, start + len(line))})
                if stripped.startswith(("```", "~~~")):
                    in_code = not in_code
                start += len(line)
        position += len(paragraph) + 2
    return boxes


def save_extraction_report(path: Path, pages: list[dict], seconds: float) -> None:
    """Write a per-PDF JSON record of extraction coverage and fallback reasons.

    Save page times, methods and warning counts without copying passage text. The
    path hash gives each PDF one report file that is replaced on its next
    extraction.
    """

    PDF_EXTRACTION_REPORT_PATH.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]
    report = {
        "file": str(path.resolve()),
        "extraction_profile": EXTRACTION_PROFILE,
        "seconds": round(seconds, 2),
        "pages": [page["metadata"] for page in pages],
        "neural_pages": sum(
            page["metadata"]["extraction_method"] == "neural_regions" for page in pages
        ),
        "fallback_pages": sum(
            page["metadata"]["extraction_method"] == "fast_fallback" for page in pages
        ),
    }
    (PDF_EXTRACTION_REPORT_PATH / f"{identity}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


def group_layout_pages_into_sections(pages: list[dict]) -> list[ExtractedSection]:
    """Collect each detected heading and its following text into a flat section.

    Keep the current heading names and levels to build breadcrumbs such as Methods >
    Training. A section can span pages, but its text is not copied into containing
    sections. Clear the breadcrumb at uncertain fallback pages to avoid guessing
    their structure.
    """

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


def read_detected_heading(
    content: str, suggested_level: int | None = None
) -> tuple[str, int]:
    """Clean a detected heading and determine its nesting depth.

    For example, 2.1 Training is level 2 even when it uses the same font as 2
    Methods. Otherwise use the model's font-based level or Markdown heading depth.
    This function interprets a heading already detected elsewhere.
    """

    line = content.strip()
    # Markdown ## usually denotes a main section; supplied neural font levels take priority.
    level = suggested_level or max(1, len(line) - len(line.lstrip("#")) - 1)
    title = line.lstrip("# ").replace("**", "").replace("<u>", "").replace("</u>", "")
    title = " ".join(title.strip("_`").split())
    label, separator, _ = title.partition(" ")
    parts = label.rstrip(".").split(".")
    # Accept labels such as 2, 2.1, A and A.1 only on blocks already tagged as headings.
    first_is_label = parts[0].isdigit() or (len(parts[0]) == 1 and parts[0].isupper())
    if separator and first_is_label and all(part.isdigit() for part in parts[1:]):
        level = len(parts)
    return title, level


def clean_pdf_text(value: str) -> str:
    """Remove PDF text artifacts before text is indexed or quoted.

    Expand joined-letter symbols such as the single fi character and remove
    invisible characters. Join soft-hyphen line breaks, but preserve ordinary
    hyphens and numbers because changing them can alter evidence.
    """

    value = value.replace("\r\n", "\n").replace("\u00ad\n", "")
    value = value.translate(
        str.maketrans(
            {
                "\x00": "",
                "\u00ad": "",
                "\u200b": "",
                "\ufeff": "",
                "\u00a0": " ",
                "\ufb00": "ff",
                "\ufb01": "fi",
                "\ufb02": "fl",
                "\ufb03": "ffi",
                "\ufb04": "ffl",
            }
        )
    )
    return value.encode("utf-8", errors="replace").decode("utf-8")


def split_sections_into_chunks(
    sections: list[ExtractedSection], title: str
) -> list[Document]:
    """Split oversized sections into overlapping child chunks for embedding.

    Repeat the paper title and heading breadcrumb so each child has searchable
    context. Reserve space for that prefix within the character limit. Keep the
    section index and citation metadata so a retrieved child can be expanded back
    into its own section.
    """

    chunks = []
    for section_index, section in enumerate(sections):
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
