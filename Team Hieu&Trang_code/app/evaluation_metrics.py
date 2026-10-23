"""Offline evidence coverage and latency metrics; labels never enter the agent."""

import json
import math
from collections import defaultdict
from pathlib import Path

from .grounding import normalized


def load_relevance(path: Path | None, case_ids: set[str]) -> dict[str, list[dict]]:
    """Read reviewed evidence spans grouped by question; alternatives share a requirement ID."""
    if path is None:
        return {}
    rows = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list):
        raise TypeError("Relevance file must contain a JSON list")
    grouped = defaultdict(list)
    for row in rows:
        if not isinstance(row, dict) or any(
            not isinstance(row.get(key), str) or not row[key].strip()
            for key in ("question_id", "requirement_id", "paper", "quote")
        ):
            raise ValueError(
                "Each relevance label needs question_id, requirement_id, paper and quote"
            )
        if row["question_id"] not in case_ids:
            raise ValueError("Relevance label references an unknown question ID")
        if len(normalized(row["quote"])) < 8:
            raise ValueError(
                "Reviewed evidence quotes must contain at least eight characters"
            )
        grouped[row["question_id"]].append(row)
    return dict(grouped)


def evidence_coverage(labels: list[dict], passages: list[dict]) -> dict:
    """Measure reviewed evidence requirements found, rather than merely correct paper titles."""
    if not labels:
        return {"coverage": None, "all_covered": None, "missing_requirements": []}
    requirements = {row["requirement_id"] for row in labels}
    # Titles are exact catalog titles; quote normalization only ignores PDF typography.
    found = {
        label["requirement_id"]
        for label in labels
        if any(
            row.get("title") == label["paper"]
            and normalized(label["quote"]) in normalized(row.get("content", ""))
            for row in passages
        )
    }
    return {
        "coverage": len(found) / len(requirements),
        "all_covered": found == requirements,
        "missing_requirements": sorted(requirements - found),
    }


def coverage_metrics(labels: list[dict], result: dict) -> dict:
    """Score the candidate pool and actual budgeted context separately using private labels."""
    runs = result.get("retrieval_trace", {}).get("tasks", [])
    audited = bool(runs) and all("candidate_passages" in run["trace"] for run in runs)
    candidates = [
        row for run in runs for row in run["trace"].get("candidate_passages", [])
    ]
    candidate = (
        evidence_coverage(labels, candidates) if audited else evidence_coverage([], [])
    )
    final = evidence_coverage(labels, result.get("context", []))
    return {
        "candidate_evidence_recall": candidate["coverage"],
        "final_evidence_coverage": final["coverage"],
        "all_requirements_covered": final["all_covered"],
        "missing_requirements": final["missing_requirements"],
    }


def percentile(values: list[float], fraction: float) -> float | None:
    """Interpolate a quantile of sorted observations; empty samples stay unknown."""
    if not values:
        return None
    position = (len(values) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return round(values[low] + (values[high] - values[low]) * (position - low), 4)


def latency_summary(rows: list[dict]) -> dict:
    """Report interpolated P50/P95 and sample counts, keeping failures visible."""
    result = {}
    for stage in ("total", "planning", "retrieval", "generation", "verification"):
        values = sorted(
            row.get("seconds")
            if stage == "total"
            else row.get("result", {}).get("timings", {}).get(stage)
            for row in rows
            if isinstance(
                row.get("seconds")
                if stage == "total"
                else row.get("result", {}).get("timings", {}).get(stage),
                (int, float),
            )
        )

        result[stage] = {
            "samples": len(values),
            "p50_seconds": percentile(values, 0.5),
            "p95_seconds": percentile(values, 0.95),
        }
    return result
