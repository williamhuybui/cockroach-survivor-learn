"""Check bounded conversation input to the unified planner, not pronoun regexes."""

import pytest

from app import question_planning


@pytest.mark.parametrize(
    "question",
    ["How does it work?", "What about training time?", "Explain another model."],
)
def test_planner_receives_bounded_history_for_context_resolution(monkeypatch, question):
    """Implicit follow-ups and independent topics use the same understanding stage."""
    captured = []

    def plan(schema, instruction, payload):
        """Only the planner decides whether previous topics resolve this request."""
        captured.append(payload)
        assert "ignore unrelated prior topics" in instruction
        return schema(tasks=[{"question": question, "paper": None, "scope": "broad"}])

    monkeypatch.setattr(question_planning, "request_structured_response", plan)
    history = [{"role": "human", "content": str(i)} for i in range(10)]
    question_planning.plan_question_tasks(question, history, [{"title": "A.pdf"}])
    assert [row["content"] for row in captured[0]["recent_conversation"]] == [
        str(i) for i in range(4, 10)
    ]


def test_independent_question_has_no_previous_context(monkeypatch):
    """The independent endpoint's empty history stays empty in planning."""

    def plan(schema, instruction, payload):
        """Verify that planning does not create hidden conversation state."""
        assert payload["recent_conversation"] == []
        return schema(
            tasks=[{"question": payload["question"], "paper": None, "scope": "broad"}]
        )

    monkeypatch.setattr(question_planning, "request_structured_response", plan)
    question_planning.plan_question_tasks("A new question?", [], [{"title": "A.pdf"}])
