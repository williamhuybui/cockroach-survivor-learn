"""Test decomposition and paper-scope resolution for questions and follow-ups.

Controlled planner outputs check direct, multipart and comparison routes, including
ambiguous or missing paper references.
"""

import pytest

from app import grounding, services
from app import question_planning as planning
from app.question_planning import QuestionPlan, SearchTask


def test_simple_question_skips_planner_model(monkeypatch):
    monkeypatch.setattr(
        planning,
        "request_structured_response",
        lambda *args: pytest.fail("Unexpected model"),
    )
    result = planning.plan_question_tasks("What is attention?", [], [{"title": "A"}])
    assert len(result.tasks) == 1


def test_unknown_comparison_paper_clarifies(monkeypatch):
    monkeypatch.setattr(
        planning,
        "request_structured_response",
        lambda *args: QuestionPlan(
            route="comparison",
            tasks=[
                SearchTask(question="A?", paper="Unknown"),
                SearchTask(question="B?", paper="B"),
            ],
        ),
    )
    assert planning.plan_question_tasks(
        "Compare A and B", [], [{"title": "B"}]
    ).clarification


def test_clarification_skips_search(monkeypatch):
    monkeypatch.setattr(
        services,
        "plan_question_tasks",
        lambda *args: QuestionPlan(clarification="Which paper?"),
    )
    monkeypatch.setattr(
        services,
        "retrieve_relevant_passages",
        lambda *args, **kwargs: pytest.fail("Must skip search"),
    )
    result = services.answer_research_question("Compare these papers")
    assert result["status"] == "clarification"
    assert not result["context"]


def test_comparison_scopes_tasks_and_marks_missing_coverage(monkeypatch):
    monkeypatch.setattr(
        services,
        "plan_question_tasks",
        lambda *args: QuestionPlan(
            route="comparison",
            tasks=[
                SearchTask(question="Method A?", paper="A"),
                SearchTask(question="Method B?", paper="B"),
            ],
        ),
    )
    calls = []

    def search(query, paper=None, diagnostics=None):
        calls.append((query, paper))
        diagnostics.update(
            candidate_sources=[paper], reranked_sources=[paper], candidate_count=1
        )
        return [{"id": paper, "title": paper, "content": "This uses 8 GPUs."}]

    monkeypatch.setattr(services, "retrieve_relevant_passages", search)
    monkeypatch.setattr(
        grounding,
        "generate_from_evidence",
        lambda *args: grounding.DraftAnswer(
            status="answered",
            claims=[
                grounding.SupportedClaim(
                    text="Uses 8 GPUs.",
                    source_id="S1",
                    quote="uses 8 GPUs",
                    task_ids=[0],
                )
            ],
        ),
    )
    result = services.answer_research_question("Compare A and B")
    assert calls == [("Method A?", "A"), ("Method B?", "B")]
    assert result["retrieval_trace"]["candidate_sources"] == ["A", "B"]
    assert result["status"] == "partial"
    assert result["missing_tasks"] == ["Method B?"]
    assert result["context"][0]["id"] == result["sources"][0]["id"]


def test_shared_budget_and_provenance(monkeypatch):
    def search(query, **kwargs):
        return [{"id": "shared", "content": "x" * 2000}] + [
            {"id": f"{query}{n}", "content": "x" * 2400} for n in range(5)
        ]

    monkeypatch.setattr(services, "retrieve_relevant_passages", search)
    tasks = [{"question": q, "paper": None} for q in "abc"]
    result = services.retrieve_task_evidence({"tasks": tasks, "retrieval_trace": {}})
    assert (
        sum(len(row["content"]) for row in result["evidence"])
        <= services.CONTEXT_CHARACTERS
    )
    assert result["evidence"][0]["task_ids"] == [0, 1, 2]


def test_claim_cannot_cover_another_paper():
    claim = grounding.SupportedClaim(
        text="Uses 8 GPUs.", source_id="S1", quote="uses 8 GPUs", task_ids=[1]
    )
    checked = grounding.verify_answer_claims(
        grounding.DraftAnswer(status="answered", claims=[claim]),
        [{"id": "S1", "title": "A", "content": "uses 8 GPUs", "task_ids": [0]}],
        [{"question": "A?", "paper": "A"}, {"question": "B?", "paper": "B"}],
    )
    assert not checked["accepted"]
    assert checked["missing_tasks"] == [0, 1]


def test_multipart_plan_keeps_subquestions(monkeypatch):
    """The planner receives user context and catalog only, and preserves distinct tasks."""
    captured = []

    def structured(schema, instruction, payload):
        captured.append(payload)
        return QuestionPlan(
            route="multipart",
            tasks=[
                SearchTask(question="What is the method?"),
                SearchTask(question="What are its limitations?"),
            ],
        )

    monkeypatch.setattr(planning, "request_structured_response", structured)
    result = planning.plan_question_tasks(
        "Explain the method and discuss its limitations", [], [{"title": "A"}]
    )
    assert len(result.tasks) == 2
    assert set(captured[0]) == {
        "question",
        "recent_user_questions",
        "recent_conversation",
        "indexed_titles",
        "indexed_papers",
    }


def test_all_task_claims_can_form_complete_answer():
    """A fully supported comparison remains answered rather than always partial."""
    tasks = [{"question": "A?", "paper": "A"}, {"question": "B?", "paper": "B"}]
    rows = [
        {"id": f"S{i + 1}", "title": title, "content": "uses 8 GPUs", "task_ids": [i]}
        for i, title in enumerate("AB")
    ]
    draft = grounding.DraftAnswer(
        status="answered",
        claims=[
            grounding.SupportedClaim(
                text="Uses 8 GPUs.",
                source_id=row["id"],
                quote="uses 8 GPUs",
                task_ids=[i],
            )
            for i, row in enumerate(rows)
        ],
    )
    checked = grounding.verify_answer_claims(draft, rows, tasks)
    assert (
        grounding.format_verified_answer(draft, checked, rows, tasks)["status"]
        == "answered"
    )


def test_plan_has_no_fixed_task_count_limit():
    """Preserve larger requests instead of rejecting the fourth task."""
    assert len(QuestionPlan(tasks=[SearchTask(question="Task?")] * 20).tasks) == 20
