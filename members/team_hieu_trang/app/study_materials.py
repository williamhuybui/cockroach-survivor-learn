"""Load the paper catalog and reading guide as knowledge alongside PDF evidence.

These small files are read from the Drive manifest, not embedded into the PDF
collection. Their passages enter keyword search and the shared neural reranker.
This lets the guide change without rebuilding paper vectors or extraction rules.
"""

import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from .config import SOURCE_MANIFEST_PATH

STUDY_FILENAMES = {"index.csv", "how_to_read_llm_papers.md"}


def load_study_sources() -> dict[str, dict]:
    """Find the two allowed supporting files in the current Drive source list.

    Resolve paths relative to the manifest and preserve the original Drive URL
    for citations. A missing manifest means no supporting knowledge is available;
    broken entries raise a clear error rather than silently using an old copy.
    """
    if not SOURCE_MANIFEST_PATH.exists():
        return {}
    sources = {}
    for row in json.loads(SOURCE_MANIFEST_PATH.read_text(encoding="utf-8-sig")):
        name = row["title"].casefold()
        if name not in STUDY_FILENAMES:
            continue
        if name in sources:
            raise ValueError(f"Duplicate study source in manifest: {name}")
        path = Path(row["path"])
        path = path if path.is_absolute() else SOURCE_MANIFEST_PATH.parent / path
        if not path.is_file():
            raise ValueError(f"Study source is missing: {path}")
        if not row.get("source_url", "").startswith("https://"):
            raise ValueError(f"Study source needs an HTTPS citation link: {name}")
        sources[name] = {
            **row,
            "path": path,
            "text": path.read_text(encoding="utf-8-sig"),
        }
    return sources


def load_paper_catalog(sources: dict[str, dict] | None = None) -> list[dict]:
    """Read paper identity/phase metadata and join reading-depth recommendations.

    The CSV supplies numbers, titles, years, links and PDF filenames. The guide's
    Depth/Papers/Why table supplies study depth and its reason; no paper priorities
    are hardcoded. These are learning recommendations, never extraction settings.
    """
    sources = load_study_sources() if sources is None else sources
    source = sources.get("index.csv")
    if not source:
        return []
    rows = list(csv.DictReader(source["text"].splitlines()))
    fields = {"#", "Phase", "Title", "Year", "arXiv link", "File"}
    if not rows or not fields <= rows[0].keys():
        raise ValueError(
            "index.csv needs #, Phase, Title, Year, arXiv link and File columns"
        )
    catalog, seen_numbers, seen_files = [], set(), set()
    for row in rows:
        number, year = int(row["#"]), int(row["Year"])
        filename = row["File"].strip()
        if (
            number < 1
            or year < 1
            or not row["Title"].strip()
            or not row["Phase"].strip()
        ):
            raise ValueError(
                "Catalog rows need positive numbers/years, a title and a phase"
            )
        if Path(filename).name != filename or not filename.lower().endswith(".pdf"):
            raise ValueError("Catalog File must be a PDF filename without directories")
        if number in seen_numbers or filename.casefold() in seen_files:
            raise ValueError("Catalog paper numbers and PDF filenames must be unique")
        seen_numbers.add(number)
        seen_files.add(filename.casefold())
        catalog.append(
            {
                "paper_number": number,
                "catalog_title": row["Title"].strip(),
                "phase": row["Phase"].strip(),
                "year": year,
                "arxiv_url": row["arXiv link"].strip(),
                "pdf_filename": filename,
            }
        )
    # Read only the guide's three-column depth table, not numbers in other prose.
    in_depth_table = False
    for line in (
        sources.get("how_to_read_llm_papers.md", {}).get("text", "").splitlines()
    ):
        cells = [
            cell.strip().replace("**", "")
            for cell in line.strip().strip("|").split("|")
        ]
        if cells == ["Depth", "Papers", "Why"]:
            in_depth_table = True
            continue
        if not in_depth_table:
            continue
        if not line.strip().startswith("|"):
            break
        if len(cells) != 3 or not re.search(r"\bPass\s+[123]\b", cells[0]):
            continue
        # Accept single paper numbers and ranges such as 31–38 in the Papers cell.
        labels = re.findall(r"\d+(?:\s*[–-]\s*\d+)?", cells[1])
        for paper in catalog:
            for label in labels:
                bounds = [int(value) for value in re.split(r"\s*[–-]\s*", label)]
                if bounds[0] <= paper["paper_number"] <= bounds[-1]:
                    paper.update(reading_depth=cells[0], reading_reason=cells[2])
                    break
    return catalog


def load_study_passages() -> list[dict]:
    """Build cited catalog and guide passages for the normal retrieval pipeline.

    Keep catalog entries together by phase, and split the Markdown at its own
    headings. Add per-paper depth lookups so names such as LoRA connect to the
    guide's numeric paper references. Content-based IDs change when a file changes.
    """
    sources = load_study_sources()
    catalog = load_paper_catalog(sources)
    passages = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1800, chunk_overlap=120)

    def add_passage(name: str, location: str, text: str, evidence_type: str) -> None:
        """Keep citation identity and bounded text together for one supporting source."""
        source = sources[name]
        for index, content in enumerate(splitter.split_text(text)):
            digest = hashlib.sha256(f"{location}\n{content}".encode()).hexdigest()[:16]
            passages.append(
                {
                    "id": f"study:{source['id']}:{digest}:{index}",
                    "content": content,
                    "title": source["title"],
                    "source_url": source["source_url"],
                    "location": location,
                    "document_type": evidence_type,
                    "evidence_type": evidence_type,
                }
            )

    phases, depths = defaultdict(list), defaultdict(list)
    for paper in catalog:
        phases[paper["phase"]].append(
            f"Paper #{paper['paper_number']}: {paper['catalog_title']} ({paper['year']})\n"
            f"PDF: {paper['pdf_filename']}\narXiv: {paper['arxiv_url']}"
        )
        if paper.get("reading_depth"):
            depths[(paper["reading_depth"], paper["reading_reason"])].append(
                f"Paper #{paper['paper_number']}: {paper['catalog_title']}"
            )
            add_passage(
                "how_to_read_llm_papers.md",
                f"Reading depth, paper #{paper['paper_number']}",
                f"Paper #{paper['paper_number']} is {paper['catalog_title']} "
                f"(identity mapped using index.csv).\n"
                f"The reading guide recommends: {paper['reading_depth']}.\n"
                f"Reason: {paper['reading_reason']}",
                "study_guide",
            )
    if catalog:
        add_passage(
            "index.csv",
            "Catalog overview",
            f"Paper catalog: {len(catalog)} papers.\nPhases:\n"
            + "\n".join(
                f"{phase}: {len(entries)} papers" for phase, entries in phases.items()
            ),
            "paper_catalog",
        )
    # Grouped recommendations keep complete priority lists together for broad questions.
    for (depth, reason), entries in depths.items():
        add_passage(
            "how_to_read_llm_papers.md",
            f"Reading priority: {depth}",
            f"The reading guide recommends: {depth}.\nReason: {reason}\n"
            "Paper names mapped from index.csv:\n" + "\n".join(entries),
            "study_guide",
        )
    for phase, entries in phases.items():
        add_passage(
            "index.csv",
            f"Phase: {phase}",
            f"Phase: {phase}\n\n" + "\n\n".join(entries),
            "paper_catalog",
        )
    guide = sources.get("how_to_read_llm_papers.md")
    if guide:
        headers = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "title"), ("##", "section"), ("###", "phase")],
            strip_headers=False,
        )
        for section in headers.split_text(guide["text"]):
            location = " > ".join(section.metadata.values()) or "Reading guide"
            add_passage(
                "how_to_read_llm_papers.md",
                location,
                section.page_content,
                "study_guide",
            )
    return passages
