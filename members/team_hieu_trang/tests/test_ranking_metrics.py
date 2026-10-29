"""Check ranking arithmetic, deduplication, missing labels and offline boundaries."""

import json
import math

import pytest

from app import evaluate
from app.evaluation_metrics import load_ranking_labels, measure_ranking_quality


def result_with_ranks(*groups):
    """Construct captured child rankings for one or several search tasks."""
    return {
        "retrieval_trace": {
            "tasks": [
                {"trace": {"reranked_passages": [{"id": key} for key in group]}}
                for group in groups
            ]
        }
    }


def test_known_ranking_scores():
    """First relevant at rank two gives reciprocal rank one-half."""
    labels = [{"chunk_id": "a", "relevance": 3}, {"chunk_id": "b", "relevance": 1}]
    metrics = measure_ranking_quality(labels, result_with_ranks(["x", "b", "a"]))
    assert metrics["reranked_recall_at_5"] == 1
    assert metrics["reranked_mrr_at_5"] == 0.5
    expected = (1 / math.log2(3) + 7 / math.log2(4)) / (7 + 1 / math.log2(3))
    assert metrics["reranked_ndcg_at_5"] == pytest.approx(expected)


def test_cutoffs_are_applied_before_scoring():
    """A positive at rank six is absent at five but present at ten."""
    scores = measure_ranking_quality(
        [{"chunk_id": "a"}], result_with_ranks(["x", "y", "z", "v", "w", "a"])
    )
    assert scores["reranked_recall_at_5"] == 0
    assert scores["reranked_mrr_at_5"] == 0
    assert scores["reranked_recall_at_10"] == 1
    assert scores["reranked_mrr_at_10"] == pytest.approx(1 / 6)


def test_multitask_rankings_interleave_without_duplicate_credit():
    """Do not compare raw MiniLM scores between different task queries."""
    scores = measure_ranking_quality(
        [{"chunk_id": "a"}], result_with_ranks(["x", "a"], ["x", "a"])
    )
    assert scores["reranked_mrr_at_5"] == 0.5
    assert scores["reranked_recall_at_5"] == 1


@pytest.mark.parametrize(
    "labels,result",
    [
        ([], result_with_ranks(["a"])),
        ([{"chunk_id": "a"}], {}),
        ([{"chunk_id": "a", "relevance": 0}], result_with_ranks([])),
    ],
)
def test_unknown_ranking_quality_is_not_a_zero_or_success(labels, result):
    """Unanswerable questions and missing annotations need separate abstention scoring."""
    assert all(
        value is None for value in measure_ranking_quality(labels, result).values()
    )


def test_empty_audited_ranking_with_positive_labels_scores_zero():
    """Retrieving nothing is a real failure when relevant evidence is known."""
    assert all(
        value == 0
        for value in measure_ranking_quality(
            [{"chunk_id": "a"}], result_with_ranks([])
        ).values()
    )


@pytest.mark.parametrize(
    "row",
    [
        {"question_id": "unknown", "chunk_id": "a"},
        {"question_id": "q", "chunk_id": ""},
        {"question_id": "q", "chunk_id": "a", "relevance": True},
        {"question_id": "q", "chunk_id": "a", "relevance": 4},
    ],
)
def test_invalid_ranking_labels_are_rejected(tmp_path, row):
    """Do not silently accept unusable judgments."""
    path = tmp_path / "ranking.json"
    path.write_text(json.dumps([row]))
    with pytest.raises(ValueError):
        load_ranking_labels(path, {"q"})


def test_ranking_labels_are_scored_only_after_inference(tmp_path, monkeypatch):
    """Neither private grades nor evidence IDs enter the question/planner call."""
    dataset = tmp_path / "questions.csv"
    dataset.write_text("id,question\nq,An independent question?\n")
    judgments = tmp_path / "ranking.json"
    judgments.write_text(
        json.dumps([{"question_id": "q", "chunk_id": "PRIVATE_CHUNK", "relevance": 3}])
    )
    calls = []
    monkeypatch.setattr(
        evaluate,
        "answer_research_question",
        lambda *args, **kwargs: (
            calls.append((args, kwargs)) or result_with_ranks(["PRIVATE_CHUNK"])
        ),
    )
    report = evaluate.run_evaluation(
        dataset, tmp_path / "report.json", ranking_path=judgments
    )
    assert calls == [(("An independent question?", []), {"require_review": False})]
    assert report["summary"]["all"]["reranked_recall_at_5"] == 1
    assert report["ranking_sha256"]
    judgments.write_text("[]")
    with pytest.raises(ValueError, match="Cannot resume"):
        evaluate.run_evaluation(
            dataset, tmp_path / "report.json", ranking_path=judgments, resume=True
        )


def test_false_answer_and_false_abstention_rates_have_separate_denominators():
    """A non-abstention on an unanswerable question is not rewarded as a partial success."""
    rows = [
        {"metrics": {"expected_abstention": expected, "abstained": actual}}
        for expected, actual in [
            (True, True),
            (True, False),
            (False, False),
            (False, True),
        ]
    ]
    summary = evaluate.summarize_evaluation_results(rows)["all"]
    assert summary["false_answer_rate_on_unanswerable"] == 0.5
    assert summary["false_abstention_rate_on_answerable"] == 0.5
    assert summary["answerable_scored_cases"] == 2
    assert summary["unanswerable_scored_cases"] == 2
