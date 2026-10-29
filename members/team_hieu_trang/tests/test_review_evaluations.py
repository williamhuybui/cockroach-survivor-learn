"""Test CSV export/import of human feedback for saved evaluation answers.

Validate reviewer fields and IDs before updating a report, preserving unfinished
feedback and rejecting inconsistent claim counts.
"""

import csv
import json

import pytest

from app.review_evaluations import (
    export_review_csv,
    import_review_csv,
    validate_review_updates,
)


def test_export_then_import_review_updates_summary(tmp_path):
    """Review fields are separate from model output and update only after import."""

    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "id": "case-1",
                        "question": "What is attention?",
                        "result": {
                            "status": "answered",
                            "answer": "A mechanism.",
                            "sources": [{"title": "Paper", "location": "Page 1"}],
                        },
                        "metrics": {
                            "correctness_review": None,
                            "supported_claims_review": None,
                            "total_claims_review": None,
                            "citation_support_review": None,
                            "review_notes": "",
                        },
                    }
                ]
            }
        )
    )
    review_path = tmp_path / "review.csv"
    export_review_csv(report_path, review_path)

    with review_path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["answer"] == "A mechanism."
    assert rows[0]["sources"] == "Paper (Page 1)"
    rows[0].update(
        {
            "correctness_review": "correct",
            "supported_claims_review": "1",
            "total_claims_review": "1",
            "citation_support_review": "supported",
            "review_notes": "Verified against page 1.",
        }
    )
    with review_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    report = import_review_csv(report_path, review_path)
    assert report["summary"]["all"]["reviewed_answer_accuracy"] == 1
    assert report["summary"]["all"]["reviewed_unsupported_claim_rate"] == 0


def test_partial_count_edit_cannot_make_saved_review_inconsistent():
    """A CSV edit must be valid against the existing count as well as its own fields."""

    existing = {"supported_claims_review": 3, "total_claims_review": 4}
    with pytest.raises(ValueError, match="cannot exceed"):
        validate_review_updates({"total_claims_review": "2"}, existing)
    with pytest.raises(ValueError, match="both"):
        validate_review_updates({"supported_claims_review": "1"}, {})
    assert (
        validate_review_updates({"total_claims_review": "5"}, existing)[
            "total_claims_review"
        ]
        == 5
    )


def test_export_preserves_existing_files(tmp_path):
    """A mistaken export path must not destroy a report or an in-progress review."""

    report = tmp_path / "report.json"
    original = '{"results": []}'
    report.write_text(original)
    with pytest.raises(ValueError, match="separate"):
        export_review_csv(report, report)
    assert report.read_text() == original
    output = tmp_path / "review.csv"
    output.write_text("existing human work")
    with pytest.raises(FileExistsError):
        export_review_csv(report, output)
    assert output.read_text() == "existing human work"


def test_invalid_import_leaves_report_unchanged(tmp_path):
    """Duplicate IDs cannot silently override another review of the same case."""

    report = tmp_path / "report.json"
    original = json.dumps({"results": [{"id": "a", "result": {}, "metrics": {}}]})
    report.write_text(original)
    review = tmp_path / "review.csv"
    review.write_text("id,correctness_review\na,correct\na,incorrect\n")
    with pytest.raises(ValueError, match="Duplicate"):
        import_review_csv(report, review)
    assert report.read_text() == original
