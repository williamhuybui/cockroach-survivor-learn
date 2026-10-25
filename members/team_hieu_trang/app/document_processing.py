"""Extract research PDFs by section, then split only oversized sections."""

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CHUNK_OVERLAP_CHARACTERS, MAX_CHUNK_CHARACTERS

HEADING = re.compile(
    r"^(?:[A-Z]|[1-9]|1\d)(?:\.\d+)*\.?\s+[A-Z][A-Za-z ,:()/&-]{2,90}$"
)
SPECIAL_HEADINGS = {
    "abstract",
    "references",
    "bibliography",
    "acknowledgments",
    "acknowledgements",
}
SECTION_HEADINGS = {
    "introduction",
    "background",
    "related work",
    "methods",
    "methodology",
    "evaluation",
    "experimental setup",
    "experiments",
    "results",
    "discussion",
    "limitations",
    "conclusion",
    "conclusions",
}


@dataclass(frozen=True)
class ExtractedSection:
    """Text under one heading, with a page location kept for citations."""

    location: str
    text: str
    strategy: str
    heading: str = ""


def clean_text(value: str) -> str:
    """Remove nulls and replace invalid Unicode before indexing PDF text."""

    return value.replace("\x00", "").encode("utf-8", errors="replace").decode("utf-8")


def classify_document(path: Path, pages: list[ExtractedSection]) -> str:
    """Classify a PDF from visible headings, without asking an LLM to guess.

    Classification chooses a chunking path; it is not a claim that the file is
    peer reviewed. Papers without recognized headings still keep page citations.
    """

    if path.suffix.lower() != ".pdf":
        return "unsupported"
    lines = [
        line.strip()
        for page in pages
        for line in normalize_headings(page.text).splitlines()
    ]
    has_abstract = any(line.casefold() == "abstract" for line in lines)
    has_introduction = any(
        re.sub(r"^[\d.]+\s+", "", line).casefold() == "introduction" for line in lines
    )
    has_references = any(
        line.casefold() in {"references", "bibliography"} for line in lines
    )
    has_sections = (
        sum(
            bool(HEADING.fullmatch(line)) or line.casefold() in SECTION_HEADINGS
            for line in lines
        )
        >= 2
    )
    # Some real papers (Chinchilla and Llama 3) omit an explicit Abstract label.
    paper_signals = has_abstract or (has_introduction and has_references)
    return "sectioned_paper" if paper_signals and has_sections else "pdf_page_fallback"


def classify_format(path: Path) -> str:
    """Identify file format separately from PDF structure; only PDF ingestion is enabled."""
    return {
        ".pdf": "pdf",
        ".docx": "word",
        ".pptx": "slides",
        ".md": "markdown",
        ".txt": "text",
        ".csv": "table",
        ".py": "code",
    }.get(path.suffix.lower(), "unsupported")


def normalize_headings(text: str) -> str:
    """Join section numbers placed on a separate line by PDF text extraction."""

    return re.sub(
        r"(?m)^((?:[1-9]|1\d)(?:\.\d+)*)\s*\n([A-Z][A-Za-z ,:()/&-]{2,90})$",
        r"\1 \2",
        text,
    )


# def styled_heading_lines(page) -> set[str]:
#     """Find bold or larger PDF lines so table numbers do not become headings."""

#     lines = [
#         line
#         for block in page.get_text("dict")["blocks"]
#         for line in block.get("lines", [])
#     ]
#     sizes = Counter()
#     for line in lines:
#         for span in line["spans"]:
#             sizes[round(span["size"], 1)] += len(span["text"])
#     body_size = sizes.most_common(1)[0][0] if sizes else 0
#     # PyMuPDF uses bit 16 for bold. Weighting font sizes by text length finds
#     # the body style rather than a large title that occurs only once.
#     return {
#         clean_text("".join(span["text"] for span in line["spans"])).strip()
#         for line in lines
#         if any(
#             span["flags"] & 16 or span["size"] > body_size + 0.5
#             for span in line["spans"]
#         )
#     }

def styled_heading_lines(page) -> set[str]:
    """Find lines where most text is bold or larger than body text."""

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

    styled_lines = set()

    for line in lines:
        spans = line["spans"]

        text = clean_text(
            "".join(span["text"] for span in spans)
        ).strip()

        if not text:
            continue

        total_chars = sum(len(span["text"]) for span in spans)

        if total_chars == 0:
            continue

        styled_chars = sum(
            len(span["text"])
            for span in spans
            if (span["flags"] & 16)
            or span["size"] > body_size + 0.5
        )

        styled_ratio = styled_chars / total_chars

        if styled_ratio >= 0.7:
            styled_lines.add(text)

    return styled_lines

def paper_sections(
    pages: list[ExtractedSection], styles: dict[str, set[str]] | None = None
) -> list[ExtractedSection]:
    """Keep a paper section together across pages and record its page span."""

    sections = []
    heading = "Title and authors"
    lines = []
    locations = []

    def save_section() -> None:
        """Append the current section before moving to the next heading."""

        if not lines:
            return
        span = (
            locations[0]
            if locations[0] == locations[-1]
            else f"{locations[0]} to {locations[-1]}"
        )
        sections.append(
            ExtractedSection(span, "\n".join(lines), "paper-section", heading)
        )

    for page in pages:
        text = normalize_headings(page.text)
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            candidate = (
                bool(HEADING.fullmatch(line)) or line.casefold() in SECTION_HEADINGS
            )
            title = re.sub(r"^(?:[A-Z]|\d+)(?:\.\d+)*\.?\s+", "", line)
            styled = (
                styles is None
                or line in styles[page.location]
                or title in styles[page.location]
            )
            if line.casefold() in SPECIAL_HEADINGS or (candidate and styled):
                save_section()
                heading, lines, locations = line, [], []
            lines.append(line)
            locations.append(page.location)
    save_section()
    return sections


def extract_sections(path: Path) -> tuple[str, list[ExtractedSection]]:
    """Extract PDF text by section, falling back to pages when needed."""

    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported")

    pages = []
    styles = {}

    with pymupdf.open(path) as pdf:
        for number, page in enumerate(pdf, start=1):
            location = f"page {number}"
            text = clean_text(page.get_text("text")).strip()

            if not text:
                print(f"WARNING: {path.name}, {location}: no text extracted")

            pages.append(ExtractedSection(location, text, "pdf-page"))
            styles[location] = styled_heading_lines(page)

    if classify_document(path, pages) != "sectioned_paper":
        return "pdf_page_fallback", pages

    sections = paper_sections(pages, styles)

    main_count = sum(
        section.heading.casefold()
        not in SPECIAL_HEADINGS | {"title and authors"}
        for section in sections
    )

    if main_count < 2:
        print(f"WARNING: {path.name}: insufficient section boundaries")
        return "pdf_page_fallback", pages

    return "sectioned_paper", sections


def chunk_sections(sections: list[ExtractedSection], title: str) -> list[Document]:
    """Repeat paper title and heading in each bounded, overlapping chunk."""

    chunks = []
    for section_index, section in enumerate(sections):
        if not section.text:
            continue
        prefix = f"Paper: {title}\n"
        if section.heading:
            prefix += f"Section: {section.heading}\n"
        budget = MAX_CHUNK_CHARACTERS - len(prefix)
        if budget <= CHUNK_OVERLAP_CHARACTERS:
            raise ValueError("Paper title is too long for the chunk size")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=budget,
            chunk_overlap=CHUNK_OVERLAP_CHARACTERS,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        pieces = splitter.split_text(section.text)
        for number, piece in enumerate(pieces, 1):
            location = section.location
            if section.heading:
                location += f": {section.heading}"
            if len(pieces) > 1:
                location += f", part {number}"
            chunks.append(
                Document(
                    page_content=prefix + piece,
                    metadata={
                        "location": location,
                        "strategy": section.strategy,
                        "heading": section.heading,
                        "section_index": section_index,
                        "child_index": number - 1,
                    },
                )
            )
    return chunks
