"""Test browser/API contracts using temporary application storage.

Exercise chat restoration, saved plans, status and quiz routes through the FastAPI
test client; external model work is replaced where needed.
"""

from fastapi.testclient import TestClient

from app import main

client = TestClient(main.app)


def test_ui_serves_chat_and_static_assets():
    response = client.get("/")
    assert response.status_code == 200
    assert "Sign in" not in response.text
    assert client.get("/static/app.js").status_code == 200


def test_chat_survives_refresh_and_can_be_deleted(monkeypatch):
    monkeypatch.setattr(
        main,
        "handle_chat_message",
        lambda *args: {"kind": "answer", "answer": "Ready", "sources": []},
    )
    body = {"message": "Question", "session_id": "browser-id"}
    assert client.post("/api/chat", json=body).status_code == 200
    assert (
        client.post("/api/chat", json={**body, "message": "Follow-up"}).status_code
        == 200
    )
    assert len(client.get("/api/sessions").json()) == 1
    assert len(client.get("/api/sessions/browser-id").json()["messages"]) == 4
    assert client.delete("/api/sessions/browser-id").status_code == 204
    assert client.get("/api/sessions/browser-id").status_code == 404


def test_obsolete_endpoints_are_removed():
    """Keep the old typo and separate quiz-creation endpoint unavailable."""
    for path in ("/team1/rag-aget", "/api/quizzes"):
        response = client.post(path, json={"query": "Explain attention"})
        assert response.status_code == 404


def test_quiz_storage_and_grading_use_sqlite(monkeypatch):
    from app import services
    from app.quiz_validation import QuizContent

    monkeypatch.setattr(
        services,
        "retrieve_relevant_passages",
        lambda query: [{"id": "paper", "content": "Evidence", "title": "Paper"}],
    )
    quiz = QuizContent(
        questions=[
            {
                "question": "Where is evidence stored?",
                "options": ["Chroma", "CSS", "HTML", "None"],
                "correct_index": 0,
                "explanation": "Chroma stores the paper passages.",
                "source_ids": ["S1"],
            }
        ]
    )
    monkeypatch.setattr(services, "generate_from_evidence", lambda *args: quiz)
    response = client.post("/api/chat", json={"message": '/quiz "RAG" medium 1'})
    assert response.status_code == 200
    assert "correct_index" not in response.json()["questions"][0]
    quiz_id = response.json()["quiz_id"]
    result = client.post(f"/api/quizzes/{quiz_id}/submit", json={"answers": [0]})
    assert result.status_code == 200
    assert result.json()["score"] == 1
