
"""Inspect and score PDF section extraction and chunking quality.

Usage:
    # Inspect Attention Is All You Need (default)
    PYTHONPATH=. python tests/inspect_chunks.py

    # Inspect a specific paper
    PYTHONPATH=. python tests/inspect_chunks.py "BERT"

    # Inspect more sample chunks
    PYTHONPATH=. python tests/inspect_chunks.py --samples 10

    # Score all PDFs and export issues to CSV
    PYTHONPATH=. python tests/inspect_chunks.py --all
"""

import argparse
import csv
import re
from pathlib import Path
from statistics import mean, median

from app.document_processing import (
    MAX_CHUNK_CHARACTERS,
    extract_sections,
    chunk_sections,
)


PAPER_DIRECTORY = Path("synced_sources/drive")
DEFAULT_PAPER = "Attention_Is_All_You_Need"
MIN_USEFUL_CHARACTERS = 200

# Heuristic checks, not definitive errors.
NOISE_PATTERNS = [
    r"provided proper attribution is provided",
    r"equal contribution\. listing order is random",
    r"work performed while at",
    r"arxiv:\d{4}\.\d{4,5}v\d+",
]


def find_paper(name: str) -> Path:
    """Find a PDF by partial filename or explicit path."""

    path = Path(name)

    if path.is_file():
        if path.suffix.lower() != ".pdf":
            raise ValueError("Only PDF files are supported")
        return path

    matches = sorted(
        pdf
        for pdf in PAPER_DIRECTORY.rglob("*")
        if pdf.is_file()
        and pdf.suffix.lower() == ".pdf"
        and name.casefold() in pdf.stem.casefold()
    )

    if not matches:
        raise FileNotFoundError(
            f"No PDF found matching '{name}' in {PAPER_DIRECTORY}"
        )

    if len(matches) > 1:
        print("\nMultiple PDFs found:")
        for pdf in matches:
            print(f"  {pdf}")
        raise ValueError("Please provide a more specific paper name")

    return matches[0]


def chunk_issues(chunk) -> list[str]:
    """Identify possible problems with one chunk."""

    issues = []
    content = chunk.page_content
    metadata = chunk.metadata
    length = len(content)

    if length > MAX_CHUNK_CHARACTERS:
        issues.append("Exceeds maximum chunk size")

    if length < MIN_USEFUL_CHARACTERS:
        issues.append("Very short chunk")

    if not metadata.get("location"):
        issues.append("Missing source location")

    if not metadata.get("strategy"):
        issues.append("Missing extraction strategy")

    if metadata.get("section_index") is None:
        issues.append("Missing section index")

    if metadata.get("child_index") is None:
        issues.append("Missing child index")

    for pattern in NOISE_PATTERNS:
        if re.search(pattern, content, flags=re.IGNORECASE):
            issues.append("Possible PDF metadata or attribution noise")
            break

    return issues


def score_quality(sections, chunks) -> dict:
    """Calculate an interpretable, heuristic quality score."""

    if not chunks:
        return {
            "overall": 0.0,
            "scores": {},
            "issues": ["No chunks were generated"],
        }

    total = len(chunks)
    lengths = [len(c.page_content) for c in chunks]

    # 1. Size compliance: 30 points.
    valid_sizes = sum(
        0 < length <= MAX_CHUNK_CHARACTERS
        for length in lengths
    )
    size_score = 30 * valid_sizes / total

    # 2. Metadata completeness: 25 points.
    required_fields = [
        "location",
        "strategy",
        "section_index",
        "child_index",
    ]

    complete_fields = sum(
        chunk.metadata.get(field) is not None
        and chunk.metadata.get(field) != ""
        for chunk in chunks
        for field in required_fields
    )

    metadata_score = (
        25 * complete_fields / (total * len(required_fields))
    )

    # 3. Content substance: 25 points.
    substance_values = [
        min(length / MIN_USEFUL_CHARACTERS, 1.0)
        for length in lengths
    ]
    substance_score = 25 * mean(substance_values)

    # 4. Content cleanliness: 20 points.
    noisy_chunks = sum(
        any(
            re.search(pattern, c.page_content, re.IGNORECASE)
            for pattern in NOISE_PATTERNS
        )
        for c in chunks
    )
    cleanliness_score = 20 * (1 - noisy_chunks / total)

    scores = {
        "Size compliance": size_score,
        "Metadata completeness": metadata_score,
        "Content substance": substance_score,
        "Content cleanliness": cleanliness_score,
    }

    issues = []

    oversized = total - valid_sizes
    short = sum(
        length < MIN_USEFUL_CHARACTERS
        for length in lengths
    )

    if oversized:
        issues.append(
            f"{oversized} chunks violate size limits"
        )

    if short:
        issues.append(
            f"{short} chunks are under 200 characters"
        )

    if noisy_chunks:
        issues.append(
            f"{noisy_chunks} chunks contain possible PDF artifacts"
        )

    # Identify sections with very little content.
    tiny_sections = [
        (i, section)
        for i, section in enumerate(sections)
        if len(section.text.strip()) < 100
    ]

    for index, section in tiny_sections:
        issues.append(
            f"Section {index} ({section.heading or 'no heading'}) "
            f"has only {len(section.text.strip())} characters"
        )

    return {
        "overall": round(sum(scores.values()), 1),
        "scores": scores,
        "issues": issues,
    }


def print_quality_report(report: dict) -> None:
    """Display scoring breakdown and diagnostic warnings."""

    print("\n========== QUALITY SCORE ==========")
    print(f"\nOVERALL SCORE: {report['overall']}/100")

    maximums = {
        "Size compliance": 30,
        "Metadata completeness": 25,
        "Content substance": 25,
        "Content cleanliness": 20,
    }

    for category, value in report["scores"].items():
        print(
            f"{category}: "
            f"{value:.1f}/{maximums[category]}"
        )

    print("\n========== DETECTED ISSUES ==========")

    if report["issues"]:
        for issue in report["issues"]:
            print(f"- {issue}")
    else:
        print("No issues detected by the configured checks.")


def inspect_paper(
    pdf_path: Path,
    sample_count: int = 5,
) -> None:
    """Inspect sections, chunks, and heuristic quality."""

    document_type, sections = extract_sections(pdf_path)
    chunks = chunk_sections(sections, pdf_path.stem)

    print("\n========== PAPER SUMMARY ==========")
    print("PDF:", pdf_path.name)
    print("Document type:", document_type)
    print("Total sections:", len(sections))
    print("Total chunks:", len(chunks))

    print("\n========== EXTRACTED SECTIONS ==========")

    for index, section in enumerate(sections):
        print(
            f"{index}: {section.heading or '(no heading)'} | "
            f"{section.location} | "
            f"{len(section.text)} characters"
        )

    lengths = [len(c.page_content) for c in chunks]

    print("\n========== CHUNK STATISTICS ==========")

    if lengths:
        print("Minimum size:", min(lengths))
        print("Maximum size:", max(lengths))
        print("Average size:", round(mean(lengths), 1))
        print("Median size:", median(lengths))

        print(
            "Chunks under 200 characters:",
            sum(
                length < MIN_USEFUL_CHARACTERS
                for length in lengths
            ),
        )

        print(
            "Chunks exceeding limit:",
            sum(
                length > MAX_CHUNK_CHARACTERS
                for length in lengths
            ),
        )
    else:
        print("No chunks were produced.")

    report = score_quality(sections, chunks)
    print_quality_report(report)

    print("\n========== SAMPLE CHUNKS ==========")

    for index, chunk in enumerate(chunks[:sample_count]):
        issues = chunk_issues(chunk)

        print(f"\n----- CHUNK {index} -----")
        print("Location:", chunk.metadata.get("location"))
        print("Section:", chunk.metadata.get("heading"))
        print("Section index:", chunk.metadata.get("section_index"))
        print("Child index:", chunk.metadata.get("child_index"))
        print("Strategy:", chunk.metadata.get("strategy"))
        print("Length:", len(chunk.page_content))

        print("Inspection:", "CHECK" if issues else "PASS")

        for issue in issues:
            print("  -", issue)

        print("\nCONTENT:")
        print(chunk.page_content)


def inspect_all_papers() -> None:
    """Score all PDFs and save results with diagnostic issues."""

    pdf_files = sorted(
        path
        for path in PAPER_DIRECTORY.rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdf"
    )

    if not pdf_files:
        print(f"No PDFs found in {PAPER_DIRECTORY}")
        return

    results = []

    for pdf_path in pdf_files:
        try:
            document_type, sections = extract_sections(pdf_path)
            chunks = chunk_sections(sections, pdf_path.stem)
            report = score_quality(sections, chunks)

            results.append({
                "paper": pdf_path.name,
                "document_type": document_type,
                "sections": len(sections),
                "chunks": len(chunks),
                "score": report["overall"],
                "issue_count": len(report["issues"]),
                "issues": "; ".join(report["issues"]),
                "status": "success",
            })

        except Exception as error:
            results.append({
                "paper": pdf_path.name,
                "document_type": "",
                "sections": 0,
                "chunks": 0,
                "score": "",
                "issue_count": 1,
                "issues": str(error),
                "status": "error",
            })

            print(f"ERROR processing {pdf_path.name}: {error}")

    # Sort successful papers by score, with errors last.
    results.sort(
        key=lambda row: (
            row["status"] != "success",
            (
                row["score"]
                if row["status"] == "success"
                else float("inf")
            ),
        )
    )

    print("\n========== ALL PAPER SCORES ==========")

    print(
        f"{'Paper':55} {'Sections':>8} "
        f"{'Chunks':>8} {'Score':>8} {'Issues':>8}"
    )
    print("-" * 95)

    for row in results:
        score = (
            f"{row['score']:.1f}"
            if row["status"] == "success"
            else "ERROR"
        )

        print(
            f"{row['paper'][:55]:55} "
            f"{row['sections']:>8} "
            f"{row['chunks']:>8} "
            f"{score:>8} "
            f"{row['issue_count']:>8}"
        )

        # Print actual issue descriptions.
        if row["issues"]:
            for issue in row["issues"].split("; "):
                print(f"    - {issue}")

    # Save results to CSV.
    output_path = Path("data/chunk_inspection_scores.csv")
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "paper",
                "document_type",
                "sections",
                "chunks",
                "score",
                "issue_count",
                "issues",
                "status",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    successful = [
        row
        for row in results
        if row["status"] == "success"
    ]

    print(f"\nPapers processed: {len(results)}")
    print(f"Successfully scored: {len(successful)}")
    print(f"Failed: {len(results) - len(successful)}")

    if successful:
        print(
            "Average score:",
            f"{mean(row['score'] for row in successful):.1f}/100",
        )

    print(f"\nResults saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect and score PDF section and chunk quality."
    )

    parser.add_argument(
        "paper",
        nargs="?",
        default=DEFAULT_PAPER,
        help=f"PDF filename or path (default: {DEFAULT_PAPER})",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of sample chunks to display (default: 5)",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Score all PDFs in synced_sources/drive",
    )

    args = parser.parse_args()

    if args.samples < 0:
        parser.error("--samples must be zero or greater")

    if args.all:
        inspect_all_papers()
    else:
        pdf_path = find_paper(args.paper)
        inspect_paper(
            pdf_path,
            sample_count=args.samples,
        )


if __name__ == "__main__":
    main()