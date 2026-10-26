"""Export and import small human-review CSV files for an evaluation report.

Run:
    python -m app.review_evaluations export report.json review.csv
    python -m app.review_evaluations import report.json completed-review.csv
"""

import argparse
import csv
import json
from pathlib import Path

from .evaluate import summarize

REVIEW_FIELDS = [
    "correctness_review",
    "supported_claims_review",
    "total_claims_review",
    "citation_support_review",
    "review_notes",
]
REVIEW_COLUMNS = [
    "id",
    "question",
    "answer",
    "status",
    "sources",
    *REVIEW_FIELDS,
]
CORRECTNESS = {"correct", "incorrect", "partial"}
CITATION_SUPPORT = {"supported", "unsupported", "partial"}


def load_report(path: Path) -> dict:
    """Read an evaluation JSON report and reject files without saved results."""

    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or not isinstance(report.get("results"), list):
        raise TypeError("Evaluation report needs a results list")
    return report


def source_summary(result: dict) -> str:
    """Make citations readable in a spreadsheet without including passage bodies."""

    return "; ".join(
        f"{row.get('title', 'Untitled')} ({row.get('location', 'unknown location')})"
        for row in result.get("sources", [])
    )


def export_review(report_path: Path, output_path: Path) -> None:
    """Export answers and existing review fields without overwriting another file."""

    if output_path.resolve() == report_path.resolve():
        raise ValueError("The review CSV must be separate from the JSON report")
    report = load_report(report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        for row in report["results"]:
            result = row.get("result", {})
            metrics = row.get("metrics", {})
            writer.writerow(
                {
                    "id": row["id"],
                    "question": row.get("question", ""),
                    "answer": result.get("answer", ""),
                    "status": result.get("status", ""),
                    "sources": source_summary(result),
                    **{field: metrics.get(field, "") for field in REVIEW_FIELDS},
                }
            )


def optional_choice(value: str | None, allowed: set[str], field: str) -> str | None:
    """Validate a non-empty reviewer choice while allowing unfinished rows."""

    cleaned = (value or "").strip().casefold()
    if not cleaned:
        return None
    if cleaned not in allowed:
        raise ValueError(f"{field} must be one of: {', '.join(sorted(allowed))}")
    return cleaned


def optional_count(value: str | None, field: str) -> int | None:
    """Validate a non-negative claim count while allowing unfinished rows."""

    cleaned = (value or "").strip()
    if not cleaned:
        return None
    if not cleaned.isdigit():
        raise ValueError(f"{field} must be a non-negative whole number")
    return int(cleaned)


def reviewed_metrics(review: dict, existing: dict) -> dict:
    """Validate one review and preserve saved values when CSV fields are blank."""

    updates = {
        "correctness_review": optional_choice(
            review.get("correctness_review", ""), CORRECTNESS, "correctness_review"
        ),
        "citation_support_review": optional_choice(
            review.get("citation_support_review", ""),
            CITATION_SUPPORT,
            "citation_support_review",
        ),
        "supported_claims_review": optional_count(
            review.get("supported_claims_review", ""), "supported_claims_review"
        ),
        "total_claims_review": optional_count(
            review.get("total_claims_review", ""), "total_claims_review"
        ),
    }
    metrics = {
        **existing,
        **{key: value for key, value in updates.items() if value is not None},
    }
    supported = metrics.get("supported_claims_review")
    total = metrics.get("total_claims_review")
    if (supported is None) != (total is None):
        raise ValueError("Provide both supported_claims_review and total_claims_review")
    if supported is not None and supported > total:
        raise ValueError("supported_claims_review cannot exceed total_claims_review")
    notes = (review.get("review_notes") or "").strip()
    if notes:
        metrics["review_notes"] = notes
    return metrics


def import_review(report_path: Path, review_path: Path) -> dict:
    """Validate the entire CSV before saving reviewed metrics into the report."""

    report = load_report(report_path)
    saved = {row["id"]: row for row in report["results"]}
    with review_path.open(encoding="utf-8-sig", newline="") as stream:
        reviews = list(csv.DictReader(stream))
    if not reviews or "id" not in reviews[0]:
        raise ValueError("Review CSV needs an id column")
    seen = set()
    for review in reviews:
        case_id = (review.get("id") or "").strip()
        if case_id in seen:
            raise ValueError(f"Duplicate review ID: {case_id}")
        if case_id not in saved:
            raise ValueError(f"Review references unknown evaluation ID: {case_id}")
        seen.add(case_id)
        row = saved[case_id]
        if "result" not in row:
            raise ValueError(
                f"Cannot review failed case {case_id}; retry evaluation first"
            )
        row["metrics"] = reviewed_metrics(review, row.get("metrics", {}))
    report["summary"] = summarize(report["results"])
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("export", "import"))
    parser.add_argument("report", type=Path)
    parser.add_argument("review", type=Path)
    arguments = parser.parse_args()
    if arguments.action == "export":
        export_review(arguments.report, arguments.review)
    else:
        import_review(arguments.report, arguments.review)
