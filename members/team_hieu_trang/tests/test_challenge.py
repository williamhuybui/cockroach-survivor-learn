"""Test daily-quiz creation, reuse and grading against temporary SQLite records.

Ensure refreshes and repeated submissions keep the saved question and first answer.
"""

import json
from datetime import date

from app import challenge
from app.db import open_database


def payload():
    """Build a fixed question to test storage without running a language model."""

    return {
        "topic": "RAG",
        "difficulty": "medium",
        "sources": [],
        "questions": [
            {
                "question": "Where is evidence found?",
                "options": ["Index", "CSS", "JWT", "HTML"],
                "correct_index": 0,
                "explanation": "The index retrieves evidence.",
                "source_ids": ["S1"],
            }
        ],
    }


def test_cached_daily_answer_is_hidden_and_first_submission_wins(monkeypatch):
    day = date(2026, 10, 19)
    monkeypatch.setattr(challenge, "get_team_date", lambda: day)
    with open_database() as database:
        database.execute(
            "INSERT INTO daily_challenges (day, payload) VALUES (?, ?)",
            (day.isoformat(), json.dumps(payload())),
        )
    hidden = challenge.get_or_create_daily_challenge()
    assert set(hidden["questions"][0]) == {"question", "options"}
    assert "sources" not in hidden
    assert challenge.submit_daily_challenge(day, [0])["score"] == 1
    assert challenge.submit_daily_challenge(day, [1])["score"] == 1


def test_sqlite_rolls_back_failed_transaction():
    try:
        with open_database() as database:
            database.execute("INSERT INTO chat_sessions (id) VALUES ('rollback-test')")
            raise RuntimeError("abort")
    except RuntimeError:
        pass
    with open_database() as database:
        assert database.execute("SELECT * FROM chat_sessions").fetchall() == []
