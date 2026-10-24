"""One shared daily question, generated lazily and stored in PostgreSQL."""
from datetime import date, datetime
from zoneinfo import ZoneInfo
from sqlalchemy import select, text
from .config import TEAM_TIMEZONE
from .db import Chunk, DailyAttempt, DailyChallenge, Document, Session
from .retrieval import evidence_row
from .schemas import QuizRequest, grade_quiz, public_quiz
from .services import quiz_payload


def team_day():
    """Choose the challenge date using the configured team timezone."""
    return datetime.now(ZoneInfo(TEAM_TIMEZONE)).date()


def daily_result(challenge, attempt):
    """Reveal the answer and explanation only after the team submits."""
    payload = challenge.payload
    result = {"day": challenge.day.isoformat(), "kind": "daily", **public_quiz(payload)}
    if attempt:
        result.update({"submitted": True, "answers": [attempt.answer], "total": 1,
                       "score": grade_quiz(payload["questions"], [attempt.answer]),
                       "questions": payload["questions"], "sources": payload["sources"]})
    return result


def get_daily(user_id: str):
    """Generate at most once per day, including concurrent requests across workers."""
    day = team_day()
    with Session.begin() as db:
        # All daily writes use the same transaction lock to avoid duplicate generation.
        db.execute(text("SELECT pg_advisory_xact_lock(713242)"))
        challenge = db.get(DailyChallenge, day)
        if challenge is None:
            documents = list(db.scalars(select(Document).where(Document.active.is_(True)).order_by(Document.id)))
            if not documents:
                raise ValueError("Sync your Google Drive documents before starting a challenge.")
            document = documents[day.toordinal() % len(documents)]
            chunks = db.scalars(select(Chunk).where(Chunk.document_id == document.id).order_by(Chunk.id).limit(6))
            evidence = [evidence_row(chunk, document) for chunk in chunks]
            request = QuizRequest(topic=document.topic or document.title[:200], difficulty="medium", count=1)
            challenge = DailyChallenge(day=day, payload=quiz_payload(request, evidence))
            db.add(challenge)
            db.flush()
        return daily_result(challenge, db.get(DailyAttempt, (day, user_id)))


def submit_daily(day: date, user_id: str, answers: list[int]):
    """Keep the first workspace answer for this date; repeat submissions are harmless."""
    with Session.begin() as db:
        db.execute(text("SELECT pg_advisory_xact_lock(713242)"))
        challenge = db.get(DailyChallenge, day)
        if not challenge:
            raise ValueError("Open the daily challenge before answering.")
        grade_quiz(challenge.payload["questions"], answers)
        attempt = db.get(DailyAttempt, (day, user_id))
        if attempt is None:
            attempt = DailyAttempt(day=day, user_id=user_id, answer=answers[0])
            db.add(attempt)
        return daily_result(challenge, attempt)
