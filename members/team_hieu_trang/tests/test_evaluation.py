"""Test dataset loading, private scoring and checkpoint/resume rules.

Keep expected labels outside agent calls and ensure interrupted runs do not mix
incompatible results.
"""

import json

import pytest

from app import evaluate


def test_key_fact_overlap_is_a_diagnostic_not_semantic_accuracy():
    assert evaluate.measure_key_fact_overlap("BART was not used", "BART was used") == 1.0
    assert evaluate.measure_key_fact_overlap("41.0 BLEU", "41.8 BLEU") == 0.5
    assert evaluate.score_evaluation_result(
        {"type": "unanswerable", "ground_truth": "not stated"},
        {"status": "abstained", "answer": "I don't know"},
    )["key_fact_overlap"] is None


def test_generic_dataset_does_not_leak_ground_truth(tmp_path, monkeypatch):
    dataset = tmp_path / "anything.csv"
    dataset.write_text(
        "id,type,question,ground_truth,source_docs\nx,custom,What?,SECRET,HiddenPaper\ny,unanswerable,Missing?,Unknown,\n"
    )
    seen = []

    def answer(question, history, *, require_review):
        assert require_review is False
        seen.append((question, history))
        return {"answer": "Cannot verify", "status": "abstained", "context": []}

    monkeypatch.setattr(evaluate, "answer_research_question", answer)
    output = tmp_path / "report.json"
    report = evaluate.run_evaluation(dataset, output)
    assert seen == [("What?", []), ("Missing?", [])]
    assert report["results"][0]["metrics"]["source_recall"] == 0
    assert report["results"][1]["metrics"]["source_recall"] is None
    assert report["summary"]["all"]["human_review_required"] == 2
    assert json.loads(output.read_text())["dataset_sha256"]


def test_errors_are_saved_and_next_case_runs(tmp_path, monkeypatch):
    dataset = tmp_path / "cases.csv"
    dataset.write_text("id,question\na,fail\nb,pass\n")

    def answer(question, history, *, require_review):
        assert require_review is False
        if question == "fail":
            raise RuntimeError("offline")
        return {"answer": "ok", "context": []}

    monkeypatch.setattr(evaluate, "answer_research_question", answer)
    report = evaluate.run_evaluation(dataset, tmp_path / "report.json")
    assert len(report["results"]) == 2
    assert report["summary"]["all"]["errors"] == 1


def test_duplicate_ids_are_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("id,question\na,first\na,second\n")
    with pytest.raises(ValueError, match="unique"):
        evaluate.load_evaluation_cases(path)


def test_human_paper_scope_and_expected_behavior_are_scoring_only(
    tmp_path, monkeypatch
):
    """Accept common benchmark columns without sending them to answer generation."""
    dataset = tmp_path / "benchmark.csv"
    dataset.write_text(
        "id,paper_scope,question,expected_answer,expected_behavior\n"
        "a,Attention Is All You Need,Question?,SECRET,answer\n"
    )
    seen = []

    def answer(question, history, *, require_review):
        assert require_review is False
        seen.append((question, history))
        return {
            "answer": "answer",
            "status": "answered",
            "context": [{"title": "03_Attention_Is_All_You_Need.pdf"}],
        }

    monkeypatch.setattr(evaluate, "answer_research_question", answer)
    report = evaluate.run_evaluation(dataset, tmp_path / "report.json")
    assert seen == [("Question?", [])]
    assert report["results"][0]["metrics"]["source_recall"] == 1
    assert report["results"][0]["metrics"]["expected_abstention"] is False


def test_resume_skips_completed_cases(tmp_path, monkeypatch):
    """An interrupted long evaluation continues without rerunning saved questions."""
    dataset = tmp_path / "cases.csv"
    dataset.write_text("id,question\na,first\nb,second\n")
    output = tmp_path / "report.json"
    calls = []
    monkeypatch.setattr(
        evaluate,
        "answer_research_question",
        lambda question, history, **kwargs: (
            calls.append(question)
            or {"answer": "ok", "status": "answered", "context": []}
        ),
    )
    evaluate.run_evaluation(dataset, output, limit=1)
    evaluate.run_evaluation(dataset, output, resume=True)
    assert calls == ["first", "second"]


def test_resume_retries_a_previous_error(tmp_path, monkeypatch):
    """Transient failures are replaced by a later successful result, not duplicated."""
    dataset = tmp_path / "cases.csv"
    dataset.write_text("id,question\na,retry\n")
    output = tmp_path / "report.json"
    failing = True

    def answer(question, history, *, require_review):
        assert require_review is False
        nonlocal failing
        if failing:
            failing = False
            raise RuntimeError("temporary failure")
        return {"answer": "ok", "status": "answered", "context": []}

    monkeypatch.setattr(evaluate, "answer_research_question", answer)
    evaluate.run_evaluation(dataset, output)
    report = evaluate.run_evaluation(dataset, output, resume=True)
    assert len(report["results"]) == 1
    assert "result" in report["results"][0]


def test_max_cases_makes_resumable_batches(tmp_path, monkeypatch):
    """A large local evaluation can intentionally run one checkpoint batch at a time."""
    dataset = tmp_path / "cases.csv"
    dataset.write_text("id,question\na,first\nb,second\n")
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        evaluate,
        "answer_research_question",
        lambda question, history, **kwargs: {
            "answer": "ok",
            "status": "answered",
            "context": [],
        },
    )
    first = evaluate.run_evaluation(dataset, output, max_cases=1)
    second = evaluate.run_evaluation(dataset, output, resume=True, max_cases=1)
    assert len(first["results"]) == 1
    assert len(second["results"]) == 2


def test_resume_rejects_changed_configuration(tmp_path, monkeypatch):
    """Never mix old workflow outputs with new model or code settings."""
    dataset = tmp_path / "cases.csv"
    dataset.write_text("id,question\na,first\n")
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        evaluate,
        "answer_research_question",
        lambda *args, **kwargs: {"answer": "ok", "context": []},
    )
    evaluate.run_evaluation(dataset, output)
    monkeypatch.setattr(evaluate, "WORKFLOW_VERSION", "changed")
    with pytest.raises(ValueError, match="workflow"):
        evaluate.run_evaluation(dataset, output, resume=True)
