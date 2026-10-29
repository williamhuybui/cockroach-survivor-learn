"""Exchange saved evaluation results with human reviewers through CSV files.

Export questions, answers, sources and review fields, then validate and import
completed reviews to update report metrics. Feedback changes offline evaluation
records, not the agent's answer-generation inputs. Run export or import with python
-m app.review_evaluations.
"""

import argparse
import csv
import json
from pathlib import Path

from .evaluate import summarize_evaluation_results

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


def load_evaluation_report(path: Path) -> dict:
    """Read a saved evaluation JSON file and require a results list.

    Both CSV export and import use this check before interpreting answer or review
    records.
    """

    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or not isinstance(report.get("results"), list):
        raise TypeError("Evaluation report needs a results list")
    return report


def export_review_csv(report_path: Path, output_path: Path) -> None:
    """Create a spreadsheet-friendly file of saved answers and review columns.

    Show source titles/locations and existing feedback without copying full
    passages. Create a new file rather than overwriting another review or the
    original report.
    """

    if output_path.resolve() == report_path.resolve():
        raise ValueError("The review CSV must be separate from the JSON report")
    report = load_evaluation_report(report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        for row in report["results"]:
            result = row.get("result", {})
            metrics = row.get("metrics", {})
            # Show paper titles and locations in the spreadsheet, not full passages.
            sources = "; ".join(
                f"{source.get('title', 'Untitled')} ({source.get('location', 'unknown location')})"
                for source in result.get("sources", [])
            )
            writer.writerow(
                {
                    "id": row["id"],
                    "question": row.get("question", ""),
                    "answer": result.get("answer", ""),
                    "status": result.get("status", ""),
                    "sources": sources,
                    **{field: metrics.get(field, "") for field in REVIEW_FIELDS},
                }
            )


def parse_review_choice(value: str | None, allowed: set[str], field: str) -> str | None:
    """Normalize and validate one reviewer category, such as correct or partial.

    Return None for a blank field so an unfinished spreadsheet cell does not erase a
    previously saved review.
    """

    cleaned = (value or "").strip().casefold()
    if not cleaned:
        return None
    if cleaned not in allowed:
        raise ValueError(f"{field} must be one of: {', '.join(sorted(allowed))}")
    return cleaned


def parse_review_count(value: str | None, field: str) -> int | None:
    """Read a nonnegative whole-number claim count from a review cell.

    Return None for blank cells so partial reviews can be imported without replacing
    existing counts.
    """

    cleaned = (value or "").strip()
    if not cleaned:
        return None
    if not cleaned.isdigit():
        raise ValueError(f"{field} must be a non-negative whole number")
    return int(cleaned)


def validate_review_updates(review: dict, existing: dict) -> dict:
    """Merge one row's feedback with existing metrics after checking allowed values.

    Require supported and total claim counts together and reject a supported count
    larger than the total. Preserve existing values where the reviewer left fields
    blank.
    """

    updates = {
        "correctness_review": parse_review_choice(
            review.get("correctness_review", ""), CORRECTNESS, "correctness_review"
        ),
        "citation_support_review": parse_review_choice(
            review.get("citation_support_review", ""),
            CITATION_SUPPORT,
            "citation_support_review",
        ),
        "supported_claims_review": parse_review_count(
            review.get("supported_claims_review", ""), "supported_claims_review"
        ),
        "total_claims_review": parse_review_count(
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


def import_review_csv(report_path: Path, review_path: Path) -> dict:
    """Validate review rows, merge them by case ID and recalculate report summaries.

    Reject unknown or duplicate IDs and invalid fields before writing the JSON
    report, so a bad spreadsheet does not partially update saved feedback.
    """

    report = load_evaluation_report(report_path)
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
        row["metrics"] = validate_review_updates(review, row.get("metrics", {}))
    report["summary"] = summarize_evaluation_results(report["results"])
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
        export_review_csv(arguments.report, arguments.review)
    else:
        import_review_csv(arguments.report, arguments.review)
