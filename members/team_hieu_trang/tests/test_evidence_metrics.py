"""Check private evidence requirements, stage loss, latency and label isolation."""

import json

import pytest

from app import evaluate, retrieval
from app.evaluation_metrics import (
    coverage_metrics,
    evidence_coverage,
    latency_summary,
    load_relevance,
)


def labels():
    """Two separately required pieces of evidence for a comparison."""
    return [
        {
            "question_id": "q",
            "requirement_id": paper,
            "paper": paper,
            "quote": f"{paper} uses eight GPUs",
        }
        for paper in "AB"
    ]


def test_correct_paper_alone_is_not_relevant_evidence():
    assert (
        evidence_coverage(labels(), [{"title": "A", "content": "Unrelated text"}])[
            "coverage"
        ]
        == 0
    )


def test_candidate_success_can_be_lost_during_context_selection():
    passages = [{"title": row["paper"], "content": row["quote"]} for row in labels()]
    result = {
        "retrieval_trace": {"tasks": [{"trace": {"candidate_passages": passages}}]},
        "context": passages[:1],
    }
    metrics = coverage_metrics(labels(), result)
    assert metrics["candidate_evidence_recall"] == 1
    assert metrics["final_evidence_coverage"] == 0.5
    assert not metrics["all_requirements_covered"]
    assert metrics["missing_requirements"] == ["B"]


def test_alternative_spans_do_not_inflate_requirement_count():
    rows = [labels()[0], {**labels()[0], "quote": "Alternative supporting evidence"}]
    assert (
        evidence_coverage(rows, [{"title": "A", "content": rows[1]["quote"]}])[
            "coverage"
        ]
        == 1
    )


def test_missing_labels_and_missing_audit_stay_unscored():
    assert coverage_metrics([], {})["final_evidence_coverage"] is None
    assert coverage_metrics(labels(), {})["candidate_evidence_recall"] is None


def test_latency_includes_failed_total_and_missing_stages():
    result = latency_summary(
        [
            {"seconds": 1, "result": {"timings": {"retrieval": 0.5}}},
            {"seconds": 3, "error": "offline"},
        ]
    )
    assert result["total"] == {"samples": 2, "p50_seconds": 2.0, "p95_seconds": 2.9}
    assert result["retrieval"]["samples"] == 1
    assert result["generation"]["p95_seconds"] is None


def test_unknown_label_question_is_rejected(tmp_path):
    path = tmp_path / "relevance.json"
    path.write_text(json.dumps(labels()))
    with pytest.raises(ValueError, match="unknown"):
        load_relevance(path, {"another"})


def test_labels_never_enter_agent_and_resume_checks_them(tmp_path, monkeypatch):
    dataset = tmp_path / "cases.csv"
    dataset.write_text("id,question\nq,Compare?\n")
    relevance = tmp_path / "relevance.json"
    relevance.write_text(json.dumps(labels()))
    output = tmp_path / "result.json"
    seen = []

    def answer(question, history):
        seen.append((question, history))
        return {"context": [], "status": "abstained"}

    monkeypatch.setattr(evaluate, "answer_with_context", answer)
    report = evaluate.evaluate(dataset, output, relevance_path=relevance)
    assert seen == [("Compare?", [])]
    assert report["summary"]["all"]["final_evidence_coverage"] == 0
    relevance.write_text(json.dumps(labels()[:1]))
    with pytest.raises(ValueError, match="relevance"):
        evaluate.evaluate(dataset, output, resume=True, relevance_path=relevance)


def test_stage_audit_is_opt_in_and_preserves_rank_order(monkeypatch):
    rows = [
        {"id": "a", "title": "A", "content": "A"},
        {"id": "b", "title": "B", "content": "B"},
    ]
    monkeypatch.setattr(retrieval, "hybrid_candidates", lambda *args: rows)
    monkeypatch.setattr(retrieval, "rerank_candidates", lambda *args: rows[::-1])
    monkeypatch.setattr(retrieval, "expand_parents", lambda ranked, limit: ranked[:1])
    audit = {"capture_evidence": True}
    retrieval.search("q", diagnostics=audit)
    assert [row["id"] for row in audit["reranked_passages"]] == ["b", "a"]
    assert len(audit["expanded_passages"]) == 1
    ordinary = {}
    retrieval.search("q", diagnostics=ordinary)
    assert "candidate_passages" not in ordinary
