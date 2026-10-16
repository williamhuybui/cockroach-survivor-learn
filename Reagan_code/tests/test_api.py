from fastapi.testclient import TestClient
from app.main import app

# No DB/model connection: check API boundaries before dependencies run.
client = TestClient(app)


def test_ui_is_served():
    response = client.get("/")
    assert response.status_code == 200
    assert "Team Knowledge" in response.text
    assert "Sign in" not in response.text


def test_static_javascript_exists():
    assert client.get("/static/app.js").status_code == 200


def test_quiz_answers_reject_boolean_and_fraction():
    from pydantic import ValidationError
    from app.schemas import SubmitRequest
    import pytest
    for answer in (True, 1.5, "1"):
        with pytest.raises(ValidationError):
            SubmitRequest(answers=[answer])


def test_shared_workspace_can_open_a_saved_chat(monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace
    from app import main

    @contextmanager
    def session():
        yield SimpleNamespace(get=lambda *args: SimpleNamespace(id="shared-chat", messages=[]))

    monkeypatch.setattr(main, "Session", session)
    response = client.get("/api/sessions/shared-chat")
    assert response.status_code == 200
    assert response.json()["session_id"] == "shared-chat"


def test_shared_workspace_can_delete_a_chat(monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace
    from app import main

    deleted = []

    class DB:
        def get(self, *args):
            return SimpleNamespace(id="shared-chat")

        def delete(self, row):
            deleted.append(row.id)

    @contextmanager
    def begin():
        yield DB()

    monkeypatch.setattr(main, "Session", SimpleNamespace(begin=begin))
    response = client.delete("/api/sessions/shared-chat")
    assert response.status_code == 204
    assert deleted == ["shared-chat"]


def test_chat_uses_browser_session_id_before_generation(monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace
    from app import main
    from app.schemas import ChatRequest

    records = {}

    class DB:
        def get(self, model, key):
            return records.get(key)

        def add(self, row):
            if row.messages is None:
                row.messages = []
            records[row.id] = row

        def flush(self):
            pass

    @contextmanager
    def begin():
        yield DB()

    monkeypatch.setattr(main, "Session", SimpleNamespace(begin=begin))
    monkeypatch.setattr(main, "respond", lambda *args: {"kind": "answer", "answer": "Ready", "sources": []})
    result = main.chat(ChatRequest(message="Question", session_id="browser-created-id"))
    assert result["session_id"] == "browser-created-id"
    assert records["browser-created-id"].messages[0]["content"] == "Question"
