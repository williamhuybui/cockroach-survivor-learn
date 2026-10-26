"""One daily paper question in the shared single-user workspace."""

import json
from datetime import date, datetime
from zoneinfo import ZoneInfo

from .config import TEAM_TIMEZONE
from .db import open_database
from .quiz_validation import QuizRequest, grade_quiz, public_quiz
from .retrieval import paper_for_daily
from .services import quiz_payload


def team_day() -> date:
    """Use the team's calendar day rather than the server's local timezone."""

    return datetime.now(ZoneInfo(TEAM_TIMEZONE)).date()


def daily_result(challenge: dict) -> dict:
    """Hide the answer until this workspace submits its first attempt."""

    payload = challenge["payload"]
    result = {"day": challenge["day"], "kind": "daily", **public_quiz(payload)}
    if challenge["answer"] is not None:
        chosen = challenge["answer"]
        result.update(
            {
                "submitted": True,
                "answers": [chosen],
                "total": 1,
                "score": grade_quiz(payload["questions"], [chosen]),
                "questions": payload["questions"],
                "sources": payload["sources"],
            }
        )
    return result


def get_daily() -> dict:
    """Reuse today's saved question or create one from an indexed paper."""

    day = team_day().isoformat()
    with open_database() as database:
        row = database.execute(
            "SELECT * FROM daily_challenges WHERE day = ?", (day,)
        ).fetchone()
    if row:
        return daily_result(
            {"day": day, "payload": json.loads(row["payload"]), "answer": row["answer"]}
        )
    evidence = paper_for_daily(date.fromisoformat(day).toordinal())
    if not evidence:
        raise ValueError("Sync research PDFs before starting the daily challenge")
    payload = quiz_payload(
        QuizRequest(topic=evidence[0]["title"], difficulty="medium", count=1), evidence
    )
    with open_database() as database:
        database.execute(
            "INSERT OR IGNORE INTO daily_challenges (day, payload) VALUES (?, ?)",
            (day, json.dumps(payload)),
        )
        row = database.execute(
            "SELECT * FROM daily_challenges WHERE day = ?", (day,)
        ).fetchone()
    return daily_result(
        {"day": day, "payload": json.loads(row["payload"]), "answer": row["answer"]}
    )


def submit_daily(day: date, answers: list[int]) -> dict:
    """Store the first answer; repeated submissions return the saved result."""

    with open_database() as database:
        row = database.execute(
            "SELECT * FROM daily_challenges WHERE day = ?", (day.isoformat(),)
        ).fetchone()
        if not row:
            raise ValueError("Open the daily challenge before answering")
        payload = json.loads(row["payload"])
        grade_quiz(payload["questions"], answers)
        database.execute(
            "UPDATE daily_challenges SET answer = ? WHERE day = ? AND answer IS NULL",
            (answers[0], day.isoformat()),
        )
        saved = database.execute(
            "SELECT answer FROM daily_challenges WHERE day = ?", (day.isoformat(),)
        ).fetchone()
    return daily_result(
        {"day": day.isoformat(), "payload": payload, "answer": saved["answer"]}
    )
