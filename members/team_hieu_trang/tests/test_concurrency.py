"""Check overlapping requests without downloading models or contacting Ollama.

Barriers require both calls to reach the expensive step together. A global request
or inference lock would prevent this and fail the test instead of going unnoticed.
"""

import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app import db, llm, main, retrieval
from app.grounding import DraftAnswer


def test_simultaneous_first_requests_load_one_reranker(monkeypatch):
    """All first callers reuse one model instead of loading several weight copies."""
    loading, finish_loading = Event(), Event()
    calls = []
    model = object()

    def load(*args, **kwargs):
        """Hold construction open so other threads also request the unloaded model."""
        calls.append(1)
        loading.set()
        assert finish_loading.wait(5)
        return model

    monkeypatch.setattr(retrieval, "_reranker_model", None)
    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=load)
    )
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(retrieval.get_reranker_model) for _ in range(4)]
        try:
            assert loading.wait(5)
        finally:
            finish_loading.set()
        assert all(future.result(timeout=5) is model for future in futures)
    assert len(calls) == 1


def test_reranker_predictions_can_overlap(monkeypatch):
    """Only initialization is protected; two predictions can reach the model together."""
    together = Barrier(2)

    def predict(pairs):
        """Return relevance only after both requests enter prediction."""
        together.wait(timeout=5)
        return [10.0] * len(pairs)

    monkeypatch.setattr(retrieval, "_reranker_model", SimpleNamespace(predict=predict))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                retrieval.rerank_passage_candidates,
                question,
                [{"id": question, "content": question}],
            )
            for question in ("training", "evaluation")
        ]
        assert [future.result(timeout=10)[0]["id"] for future in futures] == [
            "training",
            "evaluation",
        ]


def test_cached_chat_client_does_not_serialize_model_calls(monkeypatch):
    """The actual cached getter returns a shared client while invokes overlap."""
    together = Barrier(2)

    def invoke(messages):
        """Keep each response tied to its own request while both calls are active."""
        together.wait(timeout=5)
        return messages[-1][1]

    client = SimpleNamespace(
        with_structured_output=lambda *args, **kwargs: SimpleNamespace(invoke=invoke)
    )
    monkeypatch.setattr(llm, "ChatOllama", lambda **kwargs: client)
    llm.get_chat_model.cache_clear()
    try:
        assert llm.get_chat_model() is client
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(
                    llm.request_structured_response,
                    DraftAnswer,
                    "Use evidence",
                    {"question": question},
                )
                for question in ("training", "evaluation")
            ]
            assert "training" in futures[0].result(timeout=10)
            assert "evaluation" in futures[1].result(timeout=10)
    finally:
        llm.get_chat_model.cache_clear()


def post_question(session_id, message):
    """Send one browser request using its own HTTP test client."""
    with TestClient(main.app) as client:
        return client.post(
            "/api/chat", json={"session_id": session_id, "message": message}
        )


def test_independent_chat_requests_overlap(monkeypatch):
    """Both routes release their database transaction before doing model work."""
    together = Barrier(2)

    def answer(message, history):
        """Fail if another chat is blocked from reaching its own answer step."""
        together.wait(timeout=5)
        return {"kind": "answer", "answer": message, "sources": []}

    monkeypatch.setattr(main, "handle_chat_message", answer)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(post_question, session, session)
            for session in ("chat-a", "chat-b")
        ]
        responses = [future.result(timeout=10) for future in futures]
    assert [response.status_code for response in responses] == [200, 200]
    assert [response.json()["answer"] for response in responses] == ["chat-a", "chat-b"]


def test_same_chat_cannot_silently_overwrite_another_answer(monkeypatch):
    """Concurrent changes return a conflict instead of losing a saved message pair."""
    together = Barrier(2)

    def answer(message, history):
        """Make both requests read the same original history before either saves."""
        together.wait(timeout=5)
        return {"kind": "answer", "answer": message, "sources": []}

    monkeypatch.setattr(main, "handle_chat_message", answer)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(post_question, "shared-chat", message)
            for message in ("First", "Second")
        ]
        responses = [future.result(timeout=10) for future in futures]
    assert sorted(response.status_code for response in responses) == [200, 409]
    with TestClient(main.app) as client:
        saved = client.get("/api/sessions/shared-chat").json()["messages"]
    assert len(saved) == 2
    assert saved[0]["content"] == saved[1]["content"]


def test_readers_can_read_history_during_a_database_write():
    """WAL keeps an existing committed conversation readable during another write."""
    with db.open_database() as database:
        database.execute("INSERT INTO chat_sessions (id) VALUES ('readable')")
    with db.open_database() as writer:
        writer.execute(
            "UPDATE chat_sessions SET messages = '[1]' WHERE id = 'readable'"
        )
        with db.open_database() as reader:
            row = reader.execute(
                "SELECT messages FROM chat_sessions WHERE id = 'readable'"
            ).fetchone()
            assert row["messages"] == "[]"
