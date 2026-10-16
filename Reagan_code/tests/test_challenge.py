from contextlib import contextmanager
from datetime import date
from types import SimpleNamespace
from app import challenge
from app.db import DailyAttempt, DailyChallenge


def payload():
    return {"topic": "RAG", "difficulty": "medium", "questions": [{
        "question": "Where is evidence found?", "options": ["Index", "CSS", "JWT", "HTML"],
        "correct_index": 0, "explanation": "The index retrieves evidence.", "source_ids": ["s1"]
    }], "sources": [{"id": "s1", "title": "Notes", "source_url": "https://example.com"}]}


def test_daily_answer_is_hidden_until_submission():
    record = SimpleNamespace(day=date(2026, 10, 11), payload=payload())
    hidden = challenge.daily_result(record, None)
    assert set(hidden["questions"][0]) == {"question", "options"}
    assert "sources" not in hidden
    shown = challenge.daily_result(record, SimpleNamespace(answer=0))
    assert shown["score"] == 1
    assert shown["questions"][0]["explanation"]


def test_cached_daily_question_skips_generation(monkeypatch):
    day = date(2026, 10, 11)
    record = SimpleNamespace(day=day, payload=payload())
    class DB:
        def execute(self, statement):
            pass
        def get(self, model, key):
            return record if model is DailyChallenge else None
    @contextmanager
    def begin():
        yield DB()
    monkeypatch.setattr(challenge, "Session", SimpleNamespace(begin=begin))
    monkeypatch.setattr(challenge, "team_day", lambda: day)
    monkeypatch.setattr(challenge, "quiz_payload", lambda *args: (_ for _ in ()).throw(AssertionError("Regenerated cached quiz")))
    assert challenge.get_daily("user")["day"] == day.isoformat()


def test_daily_first_attempt_is_preserved_per_member(monkeypatch):
    day = date(2026, 10, 11)
    record = SimpleNamespace(day=day, payload=payload())
    attempts = {}
    class DB:
        def execute(self, statement):
            pass
        def get(self, model, key):
            return record if model is DailyChallenge else attempts.get(key)
        def add(self, row):
            assert isinstance(row, DailyAttempt)
            attempts[(row.day, row.user_id)] = row
    @contextmanager
    def begin():
        yield DB()
    monkeypatch.setattr(challenge, "Session", SimpleNamespace(begin=begin))
    assert challenge.submit_daily(day, "alice", [0])["score"] == 1
    assert challenge.submit_daily(day, "alice", [1])["score"] == 1
    assert challenge.submit_daily(day, "bob", [1])["score"] == 0
    assert len(attempts) == 2

