"""Measure evidence retrieval coverage and response times for offline evaluation.

Reviewed quotes identify facts an answer needs, while stage traces show whether
retrieval found them and the final context retained them. These labels are used
after answering and never steer search. Missing labels produce unknown scores, not
zero.
"""

import json
import math
from collections import defaultdict
from pathlib import Path

from .grounding import normalize_quote_text


def load_evidence_labels(
    path: Path | None, case_ids: set[str]
) -> dict[str, list[dict]]:
    """Load reviewed paper/quote requirements and group them by benchmark question ID.

    Validate that questions exist and each quote is usable. Several labels can share
    one requirement ID to represent alternative passages supporting the same
    required fact.
    """
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
        if len(normalize_quote_text(row["quote"])) < 8:
            raise ValueError(
                "Reviewed evidence quotes must contain at least eight characters"
            )
        grouped[row["question_id"]].append(row)
    return dict(grouped)


def measure_evidence_coverage(labels: list[dict], passages: list[dict]) -> dict:
    """Measure how many reviewed evidence requirements occur in supplied passages.

    Match an exact indexed paper title and a normalized supporting quote.
    Alternative quotes with the same requirement ID count once. This measures
    evidence presence, not whether the generated answer uses it correctly.
    """
    if not labels:
        return {"coverage": None, "all_covered": None, "missing_requirements": []}
    requirements = {row["requirement_id"] for row in labels}
    # Titles are exact catalog titles; quote normalization only ignores PDF typography.
    found = {
        label["requirement_id"]
        for label in labels
        if any(
            row.get("title") == label["paper"]
            and normalize_quote_text(label["quote"])
            in normalize_quote_text(row.get("content", ""))
            for row in passages
        )
    }
    return {
        "coverage": len(found) / len(requirements),
        "all_covered": found == requirements,
        "missing_requirements": sorted(requirements - found),
    }


def measure_retrieval_coverage(labels: list[dict], result: dict) -> dict:
    """Compare evidence coverage before reranking and in the final answer context.

    Use candidate traces and returned context to locate where needed evidence was
    lost. Leave candidate recall unknown when the run did not capture candidate
    passages.
    """
    runs = result.get("retrieval_trace", {}).get("tasks", [])
    audited = bool(runs) and all("candidate_passages" in run["trace"] for run in runs)
    candidates = [
        row for run in runs for row in run["trace"].get("candidate_passages", [])
    ]
    candidate = (
        measure_evidence_coverage(labels, candidates)
        if audited
        else measure_evidence_coverage([], [])
    )
    final = measure_evidence_coverage(labels, result.get("context", []))
    return {
        "candidate_evidence_recall": candidate["coverage"],
        "final_evidence_coverage": final["coverage"],
        "all_requirements_covered": final["all_covered"],
        "missing_requirements": final["missing_requirements"],
    }


def calculate_percentile(values: list[float], fraction: float) -> float | None:
    """Estimate a requested percentile from already sorted timing samples.

    Interpolate between neighboring observations and return None for an empty sample
    so missing measurements stay visible.
    """
    if not values:
        return None
    position = (len(values) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return round(values[low] + (values[high] - values[low]) * (position - low), 4)


def summarize_latency(rows: list[dict]) -> dict:
    """Report P50/P95 response and stage times together with sample counts.

    P50 describes the middle measurement; P95 describes the slower end. Include
    available failed-case total times and do not invent missing stage measurements.
    """
    result = {}
    for stage in ("total", "planning", "retrieval", "generation", "verification"):
        values = []
        for row in rows:
            if stage == "total":
                seconds = row.get("seconds")
            else:
                seconds = row.get("result", {}).get("timings", {}).get(stage)
            if isinstance(seconds, (int, float)):
                values.append(seconds)
        values.sort()
        result[stage] = {
            "samples": len(values),
            "p50_seconds": calculate_percentile(values, 0.5),
            "p95_seconds": calculate_percentile(values, 0.95),
        }
    return result
