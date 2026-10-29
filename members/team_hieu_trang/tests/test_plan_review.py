"""Test the browser approval/revision workflow for complex search plans.

A plan ID identifies exact saved content; stale or duplicate actions must not
execute an unintended proposal.
"""

import pytest
from fastapi.testclient import TestClient

from app import main, services
from app.db import open_database
from app.question_planning import QuestionPlan, SearchTask

client = TestClient(main.app)


def proposal(token="first", count=4):
    """Build a review response without any model or retrieval calls."""
    return {
        "kind": "answer",
        "status": "awaiting_approval",
        "answer": "Review tasks",
        "sources": [],
        "plan": {
            "plan_id": token,
            "question": "Original request",
            "route": "multipart",
            "tasks": [{"question": f"Task {i}?", "paper": None} for i in range(count)],
        },
    }


def test_proposal_persists_and_approval_uses_saved_tasks(monkeypatch):
    monkeypatch.setattr(main, "handle_chat_message", lambda *args: proposal())
    first = client.post(
        "/api/chat", json={"message": "Multipart", "session_id": "s"}
    ).json()
    assert first["pending_plan_id"] == "first"
    restored = client.get("/api/sessions/s").json()
    assert len(restored["messages"][-1]["plan"]["tasks"]) == 4
    seen = []

    def execute(question, history, **kwargs):
        seen.append((question, kwargs["approved_plan"]))
        return {
            "kind": "answer",
            "status": "answered",
            "answer": "Done",
            "sources": [],
            "context": [],
        }

    monkeypatch.setattr(main, "answer_research_question", execute)
    assert (
        client.post(
            "/api/chat",
            json={"session_id": "s", "action": "approve", "plan_id": "first"},
        ).status_code
        == 200
    )
    assert len(seen[0][1]["tasks"]) == 4
    assert seen[0][0] == "Original request"
    assert (
        client.post(
            "/api/chat",
            json={"session_id": "s", "action": "approve", "plan_id": "first"},
        ).status_code
        == 409
    )


def test_revision_invalidates_old_approval(monkeypatch):
    calls = []

    def respond(question, history, *args):
        calls.append((question, args))
        return proposal("first" if len(calls) == 1 else "revised")

    monkeypatch.setattr(main, "handle_chat_message", respond)
    client.post("/api/chat", json={"message": "Multipart", "session_id": "s"})
    result = client.post(
        "/api/chat",
        json={
            "message": "Add limitations",
            "session_id": "s",
            "action": "revise",
            "plan_id": "first",
        },
    )
    assert result.json()["pending_plan_id"] == "revised"
    assert "Original request" in calls[-1][0] and "Add limitations" in calls[-1][0]
    assert calls[-1][1] == (True,)
    assert (
        client.post(
            "/api/chat",
            json={"session_id": "s", "action": "approve", "plan_id": "first"},
        ).status_code
        == 409
    )


def test_review_node_skips_search_and_model(monkeypatch):
    monkeypatch.setattr(
        services,
        "plan_question_tasks",
        lambda *args: QuestionPlan(
            route="multipart",
            tasks=[SearchTask(question=f"Task {i}?") for i in range(4)],
        ),
    )
    monkeypatch.setattr(
        services,
        "retrieve_relevant_passages",
        lambda *args, **kwargs: pytest.fail("Must wait for approval"),
    )
    result = services.handle_chat_message("Multipart", [])
    assert result["status"] == "awaiting_approval"
    assert len(result["plan"]["tasks"]) == 4


def test_approved_plan_is_not_replanned_and_larger_plan_is_not_lost(monkeypatch):
    monkeypatch.setattr(
        services,
        "plan_question_tasks",
        lambda *args: pytest.fail("Never replan after approval"),
    )
    monkeypatch.setattr(
        services, "retrieve_relevant_passages", lambda *args, **kwargs: []
    )
    saved = proposal(count=9)["plan"]
    result = services.answer_research_question(
        "Original request", approved_plan=saved, require_review=True
    )
    assert len(result["plan"]["tasks"]) == 4
    assert "4 search tasks remain" in result["answer"]


def test_failed_approved_execution_releases_plan(monkeypatch):
    monkeypatch.setattr(main, "handle_chat_message", lambda *args: proposal())
    client.post("/api/chat", json={"message": "Multipart", "session_id": "s"})
    monkeypatch.setattr(
        main,
        "answer_research_question",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("Temporary model error")
        ),
    )
    assert (
        client.post(
            "/api/chat",
            json={"session_id": "s", "action": "approve", "plan_id": "first"},
        ).status_code
        == 422
    )
    assert client.get("/api/sessions/s").json()["pending_plan_id"] == "first"


def test_pending_plan_is_scoped_to_session_and_deleted_with_it(monkeypatch):
    monkeypatch.setattr(main, "handle_chat_message", lambda *args: proposal())
    client.post("/api/chat", json={"message": "Multipart", "session_id": "s"})
    assert (
        client.post(
            "/api/chat",
            json={"session_id": "other", "action": "approve", "plan_id": "first"},
        ).status_code
        == 409
    )
    client.delete("/api/sessions/s")
    with open_database() as database:
        assert (
            database.execute(
                "SELECT * FROM pending_plans WHERE session_id = 's'"
            ).fetchone()
            is None
        )
