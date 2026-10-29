"""Create and grade one research-paper quiz question per team calendar day.

Reuse the retrieval and quiz-generation code, but save the selected question and
first submitted answer in SQLite so refreshes do not generate a different challenge.
"""

import json
from datetime import date, datetime
from sqlite3 import Row
from zoneinfo import ZoneInfo

from .config import TEAM_TIMEZONE
from .db import open_database
from .quiz_validation import QuizRequest, grade_quiz, hide_quiz_answer_key
from .retrieval import select_daily_paper_passages
from .services import generate_quiz_content


def get_team_date() -> date:
    """Return today's date in the configured team timezone.

    Use one shared calendar boundary even when the server runs in another timezone.
    """

    return datetime.now(ZoneInfo(TEAM_TIMEZONE)).date()


def build_daily_challenge_response(challenge: Row) -> dict:
    """Convert a saved daily-challenge row into the browser response.

    Hide the answer key before submission; afterward include the saved choice,
    score, explanations and sources.
    """

    payload = json.loads(challenge["payload"])
    result = {"day": challenge["day"], "kind": "daily", **hide_quiz_answer_key(payload)}
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


def get_or_create_daily_challenge() -> dict:
    """Return today's saved question, generating it from indexed evidence if absent.

    Save the question before returning it so subsequent requests see the same
    challenge.
    """

    today = get_team_date()
    day = today.isoformat()
    with open_database() as database:
        row = database.execute(
            "SELECT * FROM daily_challenges WHERE day = ?", (day,)
        ).fetchone()
    if row:
        return build_daily_challenge_response(row)
    evidence = select_daily_paper_passages(today.toordinal())
    if not evidence:
        raise ValueError("Sync research PDFs before starting the daily challenge")
    payload = generate_quiz_content(
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
    return build_daily_challenge_response(row)


def submit_daily_challenge(day: date, answers: list[int]) -> dict:
    """Validate and save the first answer for the displayed challenge date.

    Repeated submissions return the saved result rather than changing the recorded
    first answer.
    """

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
            "SELECT * FROM daily_challenges WHERE day = ?", (day.isoformat(),)
        ).fetchone()
    return build_daily_challenge_response(saved)
