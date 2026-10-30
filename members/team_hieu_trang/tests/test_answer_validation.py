"""Check per-detail completeness, strict source association and one-call generation."""

import pytest
from pydantic import ValidationError

from app import grounding, services
from app.grounding import DetailAnswer, DraftAnswer
from app.question_planning import QuestionPlan, SearchTask


def test_citation_recovery_uses_exact_supporting_quote_from_same_task(monkeypatch):
    tasks = [{"question": "How many tokens?", "paper": "A.pdf"}]
    evidence = [
        {"id": "S1", "title": "A.pdf", "content": "No token count stated.", "task_ids": [0]},
        {"id": "S2", "title": "A.pdf", "content": "The model used 15 tokens.", "task_ids": [0]},
        {"id": "S3", "title": "B.pdf", "content": "The model used 15 tokens.", "task_ids": [0]},
    ]
    draft = DraftAnswer(answers=[DetailAnswer(availability="available", answer="15 tokens", source_ids=["S1"])])
    checked = grounding.verify_answer_citations(draft, evidence, tasks)
    def select(schema, instruction, candidates, request):
        assert [row["id"] for row in candidates] == ["S2"]
        return grounding.RecoveredCitation(source_id="S2", quote="The model used 15 tokens.")
    monkeypatch.setattr(grounding, "generate_from_evidence", select)
    result = grounding.recover_rejected_citations(draft, evidence, tasks, "How many tokens?", checked)
    assert result["accepted"][0]["source_ids"] == ["S2"]
    assert draft.answers[0].source_ids == ["S1"]


@pytest.mark.parametrize("quote", ["The model used", "It used 20 tokens."])
def test_citation_recovery_rejects_invented_or_incomplete_quotes(monkeypatch, quote):
    tasks = [{"question": "How many tokens?", "paper": "A.pdf"}]
    evidence = [{"id": "S1", "title": "A.pdf", "content": "No token count.", "task_ids": [0]},
                {"id": "S2", "title": "A.pdf", "content": "The model used 20 tokens.", "task_ids": [0]}]
    draft = DraftAnswer(answers=[DetailAnswer(availability="available", answer="20 tokens", source_ids=["S1"])])
    checked = grounding.verify_answer_citations(draft, evidence, tasks)
    monkeypatch.setattr(grounding, "generate_from_evidence", lambda *args: grounding.RecoveredCitation(source_id="S2", quote=quote))
    assert grounding.recover_rejected_citations(draft, evidence, tasks, "How many?", checked)["accepted"] == []


def test_citation_recovery_preserves_rejection_on_model_error(monkeypatch):
    tasks = [{"question": "How many tokens?", "paper": "A.pdf"}]
    evidence = [{"id": "S1", "title": "A.pdf", "content": "No token count.", "task_ids": [0]},
                {"id": "S2", "title": "A.pdf", "content": "The model used 20 tokens.", "task_ids": [0]}]
    draft = DraftAnswer(answers=[DetailAnswer(availability="available", answer="20 tokens", source_ids=["S1"])])
    checked = grounding.verify_answer_citations(draft, evidence, tasks)
    def fail(*args):
        raise RuntimeError("model offline")
    monkeypatch.setattr(grounding, "generate_from_evidence", fail)
    result = grounding.recover_rejected_citations(draft, evidence, tasks, "How many?", checked)
    assert result["rejections"] == checked["rejections"]
    assert "model offline" in result["recovery_errors"][0]["error"]


@pytest.fixture
def evidence():
    """Provide sources for two independently requested details."""
    return [
        {
            "id": "S1",
            "title": "A.pdf",
            "content": "In 2022, Alpha scored 37.6.",
            "location": "page 4",
            "task_ids": [0],
        },
        {
            "id": "S2",
            "title": "B.pdf",
            "content": "Beta trains small adapters.",
            "location": "page 7",
            "task_ids": [1],
        },
    ]


def evaluate(answers, evidence, tasks=None):
    """Run validation and rendering with no live model or storage."""
    tasks = tasks or [
        {"question": "Alpha score?", "paper": "A.pdf"},
        {"question": "Beta method?", "paper": "B.pdf"},
    ]
    # Deliberately bypass schema validation here to test the verifier's own defenses.
    draft = DraftAnswer(
        answers=[
            DetailAnswer.model_construct(
                **{"source_ids": [], "availability": "available", **row}
            )
            if row.get("answer")
            else DetailAnswer(availability="unavailable", answer="", source_ids=[])
            for row in answers
        ]
    )
    checked = grounding.verify_answer_citations(draft, evidence, tasks)
    return grounding.format_cited_answer(checked, evidence, tasks), checked


def test_all_supported_details_are_answered(evidence):
    """Each requested detail needs its own supported record."""
    result, _ = evaluate(
        [
            {"answer": "Alpha scored 37.6 in 2022.", "source_ids": ["S1"]},
            {"answer": "Beta trains small adapters.", "source_ids": ["S2"]},
        ],
        evidence,
    )
    assert result["status"] == "answered"
    assert result["missing_tasks"] == []
    assert len(result["detail_answers"]) == 2
    assert "[S1]" in result["answer"] and "[S2]" in result["answer"]


@pytest.mark.parametrize(
    "answers", [[], [{}, {}], [{"answer": "", "source_ids": ["S1"]}, {}]]
)
def test_no_answered_details_means_abstention_even_with_relevant_sources(
    evidence, answers
):
    """Finding a relevant paper does not itself supply the requested facts."""
    result, _ = evaluate(answers, evidence)
    assert result["status"] == "abstained"
    assert result["sources"] == []
    assert len(result["missing_tasks"]) == 2


@pytest.mark.parametrize(
    "answers",
    [
        [{"answer": "Alpha scored 37.6.", "source_ids": ["S1"]}],
        [{"answer": "Alpha scored 37.6.", "source_ids": ["S1"]}, {}],
    ],
)
def test_some_answered_details_mean_partial_without_model_status(evidence, answers):
    """Omitted and explicit empty records both leave their detail unanswered."""
    result, _ = evaluate(answers, evidence)
    assert result["status"] == "partial"
    assert result["missing_tasks"] == ["Beta method?"]
    assert result["detail_answers"][1]["answer"] is None
    assert [row["title"] for row in result["sources"]] == ["A.pdf"]


@pytest.mark.parametrize(
    "bad",
    [
        {"answer": "Alpha scored 99.9.", "source_ids": ["S1"]},
        {"answer": "Alpha scored 37.6.", "source_ids": ["S99"]},
        {"answer": "Alpha scored 37.6.", "source_ids": []},
        {"answer": "Alpha scored 37.6.", "source_ids": ["S2"]},
        {"answer": "Alpha scored 37.6. [S99]", "source_ids": ["S1"]},
    ],
)
def test_invalid_detail_cannot_make_answer_complete(evidence, bad):
    """Unknown citations, wrong scope and unsupported values are rejected independently."""
    result, checked = evaluate([bad], evidence)
    assert result["status"] == "abstained"
    assert result["sources"] == []
    assert checked["rejected_count"] == 1


def test_shared_source_does_not_automatically_cover_every_detail(evidence):
    """Even a passage linked to both tasks needs a separate answer for each."""
    evidence[0]["task_ids"] = [0, 1]
    tasks = [
        {"question": "Score?", "paper": "A.pdf"},
        {"question": "Batch size?", "paper": "A.pdf"},
    ]
    result, _ = evaluate(
        [{"answer": "The score was 37.6.", "source_ids": ["S1"]}, {}], evidence, tasks
    )
    assert result["status"] == "partial"
    assert result["missing_tasks"] == ["Batch size?"]


def test_literal_list_recovery_stays_within_task_and_paper():
    """Restore a missing list citation from a passage in the same paper and task."""
    rows = [
        {"id": "S1", "title": "A.pdf", "task_ids": [0], "content": "Results are discussed."},
        {"id": "S2", "title": "A.pdf", "task_ids": [0], "content": "We evaluated Alpha-13B and Beta-30B."},
    ]
    tasks = [{"question": "Which models were evaluated?", "paper": "A.pdf"}]
    result, _ = evaluate(
        [{"answer": "Alpha-13B, Beta-30B", "source_ids": ["S1"]}], rows, tasks
    )
    assert result["status"] == "answered"
    assert result["detail_answers"][0]["source_ids"] == ["S2"]


def test_citation_must_belong_to_this_detail_even_when_paper_is_unscoped(evidence):
    """A source retrieved for another task cannot silently support this one."""
    tasks = [
        {"question": "Alpha?", "paper": None},
        {"question": "Beta?", "paper": None},
    ]
    result, _ = evaluate(
        [{"answer": "Beta trains small adapters.", "source_ids": ["S2"]}],
        evidence,
        tasks,
    )
    assert result["status"] == "abstained"


def test_mixed_task_citations_keep_only_matching_sources(evidence):
    """Extra citations are removed when a source for the correct task remains."""
    result, checked = evaluate(
        [
            {"answer": "Alpha scored 37.6.", "source_ids": ["S1", "S2"]},
            {"answer": "Beta trains small adapters.", "source_ids": ["S2"]},
        ],
        evidence,
    )
    assert result["status"] == "answered"
    assert result["missing_tasks"] == []
    assert checked["rejected_count"] == 0
    assert result["detail_answers"][0]["source_ids"] == ["S1"]
    assert "Beta trains small adapters." in result["answer"]


def test_exact_repeated_paragraphs_collapse_without_losing_citation(evidence):
    """Keep one identical paragraph and its validated source."""
    result, _ = evaluate(
        [{"answer": "Alpha scored 37.6.\n\nAlpha scored 37.6.", "source_ids": ["S1"]}],
        evidence,
        [{"question": "Alpha score?", "paper": "A.pdf"}],
    )
    assert result["answer"] == "Alpha scored 37.6. [S1]"
    assert result["sources"][0]["id"] == "S1"


def test_same_answer_for_distinct_tasks_is_not_discarded(evidence):
    """Identical values in two tasks can refer to distinct facts."""
    evidence[0]["task_ids"] = [0, 1]
    tasks = [{"question": "First year?", "paper": "A.pdf"}, {"question": "Second year?", "paper": "A.pdf"}]
    result, _ = evaluate(
        [{"answer": "2022", "source_ids": ["S1"]}, {"answer": "2022", "source_ids": ["S1"]}],
        evidence,
        tasks,
    )
    assert result["answer"].count("2022 [S1]") == 2
    assert result["status"] == "answered"


def test_duplicate_labels_and_extra_records_do_not_invent_coverage(evidence):
    """Deduplicate sources and ignore answers to unrequested details."""
    tasks = [{"question": "Score?", "paper": "A.pdf"}]
    result, checked = evaluate(
        [
            {"answer": "37.6", "source_ids": ["S1", "S1"]},
            {"answer": "Extra", "source_ids": ["S2"]},
        ],
        evidence,
        tasks,
    )
    assert result["status"] == "answered"
    assert len(result["sources"]) == 1
    assert checked["rejected_count"] == 1
    assert "Extra" not in result["answer"]


def test_years_and_thousands_can_be_supported_by_cited_context(evidence):
    """Validate against surrounding source text without requiring copied quotations."""
    evidence[0]["content"] = "Training in 2022 used 10,000 examples."
    result, _ = evaluate(
        [{"answer": "Training used 10000 examples in 2022.", "source_ids": ["S1"]}],
        evidence,
        [{"question": "Examples?", "paper": "A.pdf"}],
    )
    assert result["status"] == "answered"


def test_generation_requests_fixed_count_without_global_status(monkeypatch, evidence):
    """One model call returns only answers and source labels in task order."""
    captured = []

    def generate(schema, instruction, passages, request):
        """Inspect the real schema and return a controlled missing answer."""
        captured.append((schema, request))
        with pytest.raises(ValidationError):
            schema(answers=[])
        return schema(
            answers=[{"availability": "unavailable", "answer": "", "source_ids": []}]
            * 2
        )

    monkeypatch.setattr(grounding, "generate_from_evidence", generate)
    tasks = [
        {"question": "First?", "paper": None},
        {"question": "Second?", "paper": None},
    ]
    draft = grounding.generate_grounded_draft("Two details?", evidence, tasks)
    assert len(captured) == 1
    assert set(captured[0][0].model_fields) == {"answers"}
    assert set(DetailAnswer.model_fields) == {"availability", "answer", "source_ids"}
    assert [row["position"] for row in captured[0][1]["requested_details"]] == [0, 1]
    result = grounding.format_cited_answer(
        grounding.verify_answer_citations(draft, evidence, tasks), evidence, tasks
    )
    assert result["status"] == "abstained"


def test_graph_has_one_generation_stage_and_python_status(monkeypatch, evidence):
    """The graph never invokes a synthesis/repair call after its detail answers."""
    monkeypatch.setattr(services, "list_indexed_papers", list)
    monkeypatch.setattr(
        services,
        "plan_question_tasks",
        lambda *args: QuestionPlan(tasks=[SearchTask(question="Score?", paper=None)]),
    )
    monkeypatch.setattr(
        services,
        "retrieve_relevant_passages",
        lambda *args, **kwargs: [
            {"id": "child", "title": "A.pdf", "content": evidence[0]["content"]}
        ],
    )
    calls = []
    monkeypatch.setattr(
        services,
        "generate_grounded_draft",
        lambda *args: (
            calls.append(1)
            or DraftAnswer(
                answers=[
                    DetailAnswer(
                        availability="available", answer="37.6", source_ids=["S1"]
                    )
                ]
            )
        ),
    )
    result = services.answer_research_question("Score?", require_review=False)
    assert calls == [1]
    assert result["status"] == "answered"
    assert "repair" not in services.chat_graph.nodes
