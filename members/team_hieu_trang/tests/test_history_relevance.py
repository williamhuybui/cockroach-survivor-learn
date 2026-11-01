"""Check bounded conversation input to the unified planner, not pronoun regexes."""

from app import question_planning


def test_planner_receives_bounded_history_for_context_resolution(monkeypatch):
    """A follow-up reaches the planner with only the last six conversation turns."""
    question = "How does it work?"
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
