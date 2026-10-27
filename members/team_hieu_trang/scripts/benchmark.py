"""Run questions through the agent and record answers, stage timings and LLM usage.

Usage (from members/team_hieu_trang):
  python scripts/benchmark.py smoke_set.csv
  python scripts/benchmark.py ../../dev_set.csv --ids D01 D04 --output data/evaluations/dev.json

Calls answer_with_context exactly like api_view.team1_agent, without the HTTP layer,
so there is no 120 s gateway timeout; compare "seconds" against 120 yourself.
Reference answers in the CSV are never passed to the agent. Grade with scripts/judge.py.
"""

import argparse
import csv
import json
import resource
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from pathlib import Path

TEAM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEAM_ROOT))

from app import grounding, llm, question_planning, retrieval, services  # noqa: E402

RECORD: dict = {}


@contextmanager
def timed(name):
    start = time.perf_counter()
    try:
        yield
    finally:
        RECORD.setdefault("stages", {}).setdefault(name, []).append(
            round(time.perf_counter() - start, 3)
        )


def wrap(module, attr, name):
    """Time a pipeline function in place; module-level lookups pick up the wrapper."""
    original = getattr(module, attr)

    @wraps(original)
    def inner(*args, **kwargs):
        with timed(name):
            return original(*args, **kwargs)

    setattr(module, attr, inner)


wrap(retrieval, "matching_records", "retrieval.load_chunks")
wrap(retrieval, "keyword_candidates", "retrieval.bm25")
wrap(retrieval, "retrieve_candidates", "retrieval.dense")
wrap(retrieval, "rerank_candidates", "retrieval.rerank")
wrap(retrieval, "expand_parents", "retrieval.expand_parents")
wrap(services, "plan_question", "planning")
wrap(services, "format_answer", "formatting")

_chat_model = llm.chat_model


def structured_request(schema, instruction, payload, max_tokens=None):
    """Same logic as llm.structured_request, plus Ollama token and duration capture."""
    model = (_chat_model(max_tokens) if max_tokens else _chat_model()).with_structured_output(
        schema, method="json_schema", include_raw=True
    )
    messages = [("system", instruction), ("human", json.dumps(payload, ensure_ascii=False))]
    for attempt in range(2):
        start = time.perf_counter()
        response = model.invoke(messages)
        meta = getattr(response.get("raw"), "response_metadata", {}) or {}
        RECORD.setdefault("llm_calls", []).append(
            {
                "schema": schema.__name__,
                "attempt": attempt,
                "seconds": round(time.perf_counter() - start, 2),
                "prompt_tokens": meta.get("prompt_eval_count"),
                "output_tokens": meta.get("eval_count"),
                "prefill_seconds": round((meta.get("prompt_eval_duration") or 0) / 1e9, 2),
                "decode_seconds": round((meta.get("eval_duration") or 0) / 1e9, 2),
                "load_seconds": round((meta.get("load_duration") or 0) / 1e9, 2),
            }
        )
        if response.get("parsing_error") is None and response.get("parsed") is not None:
            return response["parsed"]
        if attempt:
            raise ValueError("Structured output parsing failed")


llm.structured_request = structured_request
question_planning.structured_request = structured_request

_verify = grounding.verify_claims


def verify_and_keep_draft(draft, evidence, tasks, question=""):
    """Keep the raw draft so verifier changes can be replayed offline without Gemma."""
    with timed("verification"):
        result = _verify(draft, evidence, tasks, question=question)
    RECORD["draft"] = draft.model_dump()
    RECORD["rejected_claims"] = result.get("rejection_details", [])
    return result


services.verify_claims = verify_and_keep_draft


def ollama_rss_mb():
    output = subprocess.run(["ps", "-axo", "rss,comm"], capture_output=True, text=True).stdout
    return round(sum(int(line.split()[0]) for line in output.splitlines()[1:] if "ollama" in line) / 1024)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path, help="CSV with id and question columns")
    parser.add_argument("--ids", nargs="*", help="Only these question IDs")
    parser.add_argument("--output", type=Path, help="Default: data/evaluations/<dataset>_<time>.json")
    args = parser.parse_args()
    output = args.output or TEAM_ROOT / "data" / "evaluations" / (
        f"{args.dataset.stem}_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with args.dataset.open(encoding="utf-8-sig") as stream:
        cases = [row for row in csv.DictReader(stream) if not args.ids or row["id"] in args.ids]

    report = []
    for case in cases:
        RECORD.clear()
        start = time.perf_counter()
        try:
            result = services.answer_with_context(
                case["question"], [], capture_evidence=True, allow_clarification=False
            )
            error = None
        except Exception as exc:  # noqa: BLE001 - record the failure and continue.
            result, error = {}, f"{type(exc).__name__}: {exc}"
        trace = result.get("retrieval_trace", {})
        report.append(
            {
                "id": case["id"],
                "seconds": round(time.perf_counter() - start, 2),
                "error": error,
                "status": result.get("status"),
                "answer": result.get("answer"),
                "claims": result.get("claims"),
                "sources": [row.get("title") for row in result.get("sources", [])],
                "tasks": [run["task"] for run in trace.get("tasks", [])],
                "context": [
                    {key: row.get(key) for key in ("id", "title", "location", "task_ids", "rerank_score", "content")}
                    for row in result.get("context", [])
                ],
                **RECORD,
                # macOS reports ru_maxrss in bytes, Linux in kilobytes.
                "python_peak_rss_mb": round(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    / (2**20 if sys.platform == "darwin" else 2**10)
                ),
                "ollama_rss_mb": ollama_rss_mb(),
            }
        )
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        row = report[-1]
        calls = len(row.get("llm_calls", []))
        print(f"{case['id']}: {row['seconds']}s, {calls} LLM calls, status={row['status']}"
              + (f", ERROR {error}" if error else ""), flush=True)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
