"""Run benchmark questions through the research agent and save auditable results.

Only each question enters the agent; expected answers, papers and reviewed evidence
remain private scoring data. Save answers, passages, citations, timings and errors
after every case, then export results for human review. Run: python -m app.evaluate
cases.csv --output data/evaluations/run.json.
"""

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from .config import (
    CHAT_MODEL,
    EMBED_MODEL,
    MIN_RERANK_SCORE,
    OLLAMA_BASE_URL,
    RERANK_MODEL,
)
from .evaluation_metrics import (
    load_evidence_labels,
    measure_retrieval_coverage,
    summarize_latency,
)
from .paper_store import build_index_signature, open_chunk_store
from .services import WORKFLOW_VERSION, answer_research_question
from .study_materials import load_study_passages


def build_evaluation_configuration() -> dict:
    """Record model settings plus hashes of pipeline files and current index records.

    Use this configuration to reject a resume that would mix results from different
    code, models or indexes. The source hash includes comments and docstrings as
    well as executable code.
    """
    root = Path(__file__).parent
    files = (
        "services.py",
        "question_planning.py",
        "grounding.py",
        "retrieval.py",
        "llm.py",
        "config.py",
        "document_processing.py",
        "neural_pdf_extraction.py",
        "paper_store.py",
        "study_materials.py",
        "evaluation_metrics.py",
        "evaluate.py",
    )
    digest = hashlib.sha256(
        b"".join((root / name).read_bytes() for name in files)
    ).hexdigest()
    stored = open_chunk_store().get(include=["metadatas"])
    index_rows = sorted(
        (key, meta.get("fingerprint", ""))
        for key, meta in zip(stored["ids"], stored["metadatas"])
    )
    index_hash = hashlib.sha256(json.dumps(index_rows).encode()).hexdigest()
    return {
        "workflow": WORKFLOW_VERSION,
        "code_sha256": digest,
        "index_sha256": index_hash,
        "study_materials_sha256": hashlib.sha256(
            json.dumps(load_study_passages(), sort_keys=True).encode()
        ).hexdigest(),
        "index_signature": build_index_signature(),
        "rerank_threshold": MIN_RERANK_SCORE,
        "ollama_url": OLLAMA_BASE_URL,
        "models": {
            "chat": CHAT_MODEL,
            "embedding": EMBED_MODEL,
            "reranker": RERANK_MODEL,
        },
    }


def load_evaluation_cases(path: Path) -> list[dict]:
    """Read benchmark CSV rows and require a unique ID and nonblank question per row.

    Keep optional category/reference columns available for scoring, without
    restricting the dataset to one benchmark format.
    """
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(
        not row.get("id", "").strip() or not row.get("question", "").strip()
        for row in rows
    ):
        raise ValueError("Dataset needs non-empty id and question columns")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Dataset IDs must be unique")
    return rows


def normalize_paper_title(title: str) -> str:
    """Make paper filenames and human-written titles comparable for source scoring.

    Ignore filename extensions, leading numeric prefixes, case and punctuation; this
    normalization is for evaluation, not retrieval scope selection.
    """
    stem = Path(title.strip()).stem.casefold()
    stem = re.sub(r"^\d+[_ -]*", "", stem)
    return re.sub(r"[^a-z0-9]+", "", stem)


def read_expected_abstention(case: dict) -> bool | None:
    """Read whether a benchmark case expects the agent to decline answering.

    Prefer expected_behavior when supplied, otherwise recognize the unanswerable
    case type. Return None when no expectation is labeled so missing labels are not
    treated as false.
    """
    behavior = case.get("expected_behavior", "").strip().casefold()
    if behavior:
        return behavior in {"abstain", "abstained", "no_answer", "refuse"}
    if case.get("type", "").casefold() == "unanswerable":
        return True
    return None


def score_evaluation_result(
    case: dict, result: dict, relevance: list[dict] | None = None
) -> dict:
    """Compare returned sources and passages with private benchmark labels.

    Measure source/evidence coverage and expected refusal behavior after generation.
    Leave correctness, supported-claim counts and citation support unset for human
    review rather than inferring answer accuracy from retrieval alone.
    """
    # These optional paper labels are read only for scoring, never for retrieval.
    names = case.get("source_docs") or case.get("paper_scope") or ""
    expected = {
        normalize_paper_title(name) for name in names.split(";") if name.strip()
    }
    found = {
        normalize_paper_title(row.get("title", "")) for row in result.get("context", [])
    }
    candidates = {
        normalize_paper_title(title)
        for title in result.get("retrieval_trace", {}).get("candidate_sources", [])
    }
    reranked = {
        normalize_paper_title(title)
        for title in result.get("retrieval_trace", {}).get("reranked_sources", [])
    }
    return {
        **measure_retrieval_coverage(relevance or [], result),
        "source_recall": len(expected & found) / len(expected) if expected else None,
        "all_sources_found": expected <= found if expected else None,
        "candidate_source_recall": len(expected & candidates) / len(expected)
        if expected and "retrieval_trace" in result
        else None,
        "reranked_source_recall": len(expected & reranked) / len(expected)
        if expected and "retrieval_trace" in result
        else None,
        "abstained": result.get("status") == "abstained"
        if "status" in result
        else None,
        "expected_abstention": read_expected_abstention(case),
        "correctness_review": None,
        "supported_claims_review": None,
        "total_claims_review": None,
        "citation_support_review": None,
        "review_notes": "",
    }


def summarize_evaluation_results(rows: list[dict]) -> dict:
    """Aggregate evaluation coverage, timing, refusal behavior and reviewed correctness.

    Group results overall and by available case/route categories. Keep errors and
    scored sample counts visible; missing labels and unreviewed answers do not
    become successful scores.
    """
    groups = defaultdict(list)
    for row in rows:
        groups["all"].append(row)
        route = row.get("result", {}).get("retrieval_trace", {}).get("route", "unknown")
        groups[f"actual_route:{route}"].append(row)
        groups[
            f"runtime_state:{row.get('runtime_state', 'unclassified')}:route:{route}"
        ].append(row)
        for key in (
            "type",
            "difficulty",
            "split",
            "paper_scope",
            "question_type",
            "expected_route",
            "expected_behavior",
        ):
            if row.get(key):
                groups[f"{key}:{row[key]}"].append(row)
    summary = {}
    for name, items in groups.items():
        recalls = [
            item["metrics"]["source_recall"]
            for item in items
            if item.get("metrics", {}).get("source_recall") is not None
        ]
        known = [
            item
            for item in items
            if item.get("metrics", {}).get("abstained") is not None
            and item.get("metrics", {}).get("expected_abstention") is not None
        ]
        refusals = [item for item in known if item["metrics"]["abstained"]]
        expected_refusals = [
            item for item in known if item["metrics"]["expected_abstention"]
        ]
        correct_refusals = sum(
            item["metrics"]["expected_abstention"] for item in refusals
        )
        reviewed = [
            item
            for item in items
            if item.get("metrics", {}).get("correctness_review")
            in {"correct", "partial", "incorrect"}
        ]
        claim_reviews = [
            item["metrics"]
            for item in items
            if isinstance(item.get("metrics", {}).get("total_claims_review"), int)
            and isinstance(item.get("metrics", {}).get("supported_claims_review"), int)
        ]
        total_claims = sum(row["total_claims_review"] for row in claim_reviews)
        supported_claims = sum(row["supported_claims_review"] for row in claim_reviews)
        summary[name] = {
            "latency": summarize_latency(items),
            "cases": len(items),
            "errors": sum("error" in item for item in items),
            "source_recall": sum(recalls) / len(recalls) if recalls else None,
            "source_scored_cases": len(recalls),
            "abstention_precision": correct_refusals / len(refusals)
            if refusals
            else None,
            "abstention_recall": correct_refusals / len(expected_refusals)
            if expected_refusals
            else None,
            "abstention_scored_cases": len(known),
            "reviewed_answer_accuracy": sum(
                item["metrics"]["correctness_review"] == "correct" for item in reviewed
            )
            / len(reviewed)
            if reviewed
            else None,
            "reviewed_cases": len(reviewed),
            "reviewed_unsupported_claim_rate": (total_claims - supported_claims)
            / total_claims
            if total_claims
            else None,
            "human_review_required": sum(
                item.get("metrics", {}).get("correctness_review") is None
                for item in items
            ),
        }
        for metric in (
            "candidate_evidence_recall",
            "final_evidence_coverage",
            "all_requirements_covered",
        ):
            values = [
                item["metrics"][metric]
                for item in items
                if item.get("metrics", {}).get(metric) is not None
            ]
            summary[name][metric] = sum(values) / len(values) if values else None
            summary[name][metric + "_scored_cases"] = len(values)
    return summary


def run_evaluation(
    path: Path,
    output: Path,
    limit: int | None = None,
    resume: bool = False,
    max_cases: int | None = None,
    relevance_path: Path | None = None,
    runtime_state: str = "unclassified",
) -> dict:
    """Run independent benchmark questions and checkpoint each answer or error to JSON.

    On resume, verify dataset, labels, settings and index identity before keeping
    completed cases and retrying failed ones. Pass only question text to the agent,
    then score privately and update the summary after each case.
    """
    dataset_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    cases = load_evaluation_cases(path)
    relevance = load_evidence_labels(relevance_path, {case["id"] for case in cases})
    relevance_hash = (
        hashlib.sha256(relevance_path.read_bytes()).hexdigest()
        if relevance_path
        else None
    )
    if runtime_state not in {"cold", "warm", "unclassified"}:
        raise ValueError("runtime_state must be cold, warm or unclassified")
    if limit is not None:
        cases = cases[:limit]
    if max_cases is not None and max_cases < 1:
        raise ValueError("max_cases must be at least one")
    if output.exists() and not resume:
        raise ValueError(
            "Choose a new output filename or use --resume after an interrupted run"
        )
    if resume:
        report = json.loads(output.read_text(encoding="utf-8"))
        if report.get("dataset_sha256") != dataset_hash:
            raise ValueError("Cannot resume: the dataset file changed")
        if (
            report.get("relevance_sha256") != relevance_hash
            or report.get("runtime_state", "unclassified") != runtime_state
        ):
            raise ValueError(
                "Cannot resume: relevance labels or runtime classification changed"
            )
        if report.get("configuration") != build_evaluation_configuration():
            raise ValueError(
                "Cannot resume: workflow, model settings or code changed; choose a new report"
            )
        if report.get("indexed_chunks") != open_chunk_store().count():
            raise ValueError(
                "Cannot resume: the index chunk count changed; choose a new report"
            )
    else:
        report = {
            "created_at": datetime.now(UTC).isoformat(),
            "configuration": build_evaluation_configuration(),
            "dataset_sha256": dataset_hash,
            "relevance_sha256": relevance_hash,
            "runtime_state": runtime_state,
            "index_signature": build_index_signature(),
            "indexed_chunks": open_chunk_store().count(),
            "models": {
                "chat": CHAT_MODEL,
                "embedding": EMBED_MODEL,
                "reranker": RERANK_MODEL,
            },
            "results": [],
        }
    # Successful rows stay immutable. Failed rows are retried on resume because a
    # local model/service outage should not become a permanent benchmark result.
    completed = {row["id"] for row in report["results"] if "result" in row}
    cases = [case for case in cases if case["id"] not in completed]
    if max_cases is not None:
        cases = cases[:max_cases]
    output.parent.mkdir(parents=True, exist_ok=True)
    for case in cases:
        started = perf_counter()
        row = dict(case)
        # This is an explicit measurement condition, never inferred from case labels.
        row["runtime_state"] = runtime_state
        try:
            # No labels, expected papers, or reference answers enter this call.
            result = answer_research_question(case["question"], [])
            row.update(
                result=result,
                metrics=score_evaluation_result(
                    case, result, relevance.get(case["id"], [])
                ),
            )
        except Exception as exc:  # noqa: BLE001 - an evaluation must record one failed case and continue.
            row["error"] = f"{type(exc).__name__}: {exc}"
        row["seconds"] = round(perf_counter() - started, 2)
        report["results"] = [
            saved for saved in report["results"] if saved["id"] != case["id"]
        ]
        report["results"].append(row)
        report["summary"] = summarize_evaluation_results(report["results"])
        output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(
            f"{case['id']}: {'ERROR' if 'error' in row else 'saved'} ({row['seconds']}s)",
            flush=True,
        )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument(
        "--relevance", type=Path, help="Private reviewed evidence requirements as JSON"
    )
    parser.add_argument(
        "--runtime-state",
        choices=["cold", "warm", "unclassified"],
        default="unclassified",
    )
    args = parser.parse_args()
    run_evaluation(
        args.dataset,
        args.output,
        args.limit,
        args.resume,
        args.max_cases,
        args.relevance,
        args.runtime_state,
    )
