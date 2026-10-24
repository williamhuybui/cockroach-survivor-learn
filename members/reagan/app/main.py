import logging
from datetime import date
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select

from .db import Attempt, ChatSession, Document, Quiz, SHARED_USER_ID, Session, Setting
from .challenge import get_daily, submit_daily
from .schemas import ChatRequest, QuizRequest, SubmitRequest, grade_quiz
from .services import make_quiz, respond

STATIC = Path(__file__).parent / "static"


app = FastAPI(title="Team Knowledge Agent")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.middleware("http")
async def disable_frontend_cache(request: Request, call_next):
    """Prevent an old HTML or JavaScript file from surviving interface updates."""
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


@app.exception_handler(ValueError)
async def bad_value(request: Request, exc: ValueError):
    """Convert application validation errors to readable HTTP responses."""
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def service_failure(request: Request, exc: Exception):
    # Do not expose connection strings, SQL bind values, or source text to clients/logs.
    """Report failure without exposing credentials, SQL values, or document text."""
    logging.error("Request %s failed: %s", request.url.path, type(exc).__name__)
    return JSONResponse(status_code=503, content={"detail": "A model or database operation failed. Check service health and the setup guide."})


@app.get("/")
def index():
    """Serve the chat interface from the static folder."""
    return FileResponse(STATIC / "index.html")


@app.get("/api/sources")
def sources():
    """List synchronized document metadata for the shared workspace."""
    with Session() as db:
        return [{"id": d.id, "title": d.title, "source_type": d.source_type, "url": d.source_url,
                 "session_date": d.session_date, "indexed_at": d.indexed_at, "active": d.active}
                for d in db.scalars(select(Document).order_by(Document.title))]


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    """Restore a saved conversation from the shared workspace."""
    with Session() as db:
        row = db.get(ChatSession, session_id)
        if not row:
            raise HTTPException(404, "Chat session not found")
        return {"session_id": row.id, "messages": row.messages}


@app.delete("/api/sessions/{session_id}", status_code=204)
def delete_session(session_id: str):
    """Delete one conversation from the shared recent-chat list."""
    with Session.begin() as db:
        row = db.get(ChatSession, session_id)
        if not row:
            raise HTTPException(404, "Chat session not found")
        db.delete(row)
    return Response(status_code=204)


@app.post("/api/chat")
def chat(body: ChatRequest):
    """Run retrieval and save the answer, citations, and quiz in the shared chat history."""
    # Save the browser-provided ID before the slow model call. A refresh can then
    # return to this same conversation instead of creating a duplicate.
    with Session.begin() as db:
        row = db.get(ChatSession, body.session_id) if body.session_id else None
        if row is None:
            row = ChatSession(id=body.session_id, user_id=SHARED_USER_ID) if body.session_id else ChatSession(user_id=SHARED_USER_ID)
            db.add(row)
            db.flush()
        session_id = row.id
        history = list(row.messages)
    result = respond(body.message, history, SHARED_USER_ID)
    summary = result.get("answer", f"Quiz created: {result.get('topic', '')}")
    assistant_message = {"role": "assistant", "content": summary, "sources": result.get("sources", [])}
    if result["kind"] == "quiz":
        assistant_message["quiz"] = dict(result)
    updated = history + [{"role": "human", "content": body.message}, assistant_message]
    with Session.begin() as db:
        row = db.get(ChatSession, session_id)
        if row is None:
            raise HTTPException(409, "This chat was deleted while the answer was being generated")
        row.messages = updated
        result["session_id"] = session_id
    return result


@app.get("/api/sessions")
def sessions():
    """List shared chats, using each first question as its title."""
    with Session() as db:
        rows = db.scalars(select(ChatSession))
        return [{"session_id": row.id, "title": row.messages[0]["content"][:70]}
                for row in rows if row.messages]


@app.get("/api/status")
def status():
    """Return readiness and the last successful ingestion time for the sidebar footer."""
    with Session() as db:
        ready = db.scalar(select(Document.id).where(Document.active.is_(True)).limit(1)) is not None
        synced = db.get(Setting, "last_ingested_at")
        return {"ready": ready, "synced_at": synced.value if synced else None}


@app.get("/api/daily-challenge")
def daily():
    """Load today's shared question and saved team result, if any."""
    return get_daily(SHARED_USER_ID)


@app.post("/api/daily-challenge/{day}/submit")
def daily_submit(day: date, body: SubmitRequest):
    """Grade the displayed day's question, even if midnight passed while it was open."""
    return submit_daily(day, SHARED_USER_ID, body.answers)


@app.post("/api/quizzes")
def quizzes(body: QuizRequest):
    """Generate a quiz with validated topic, difficulty, and question count."""
    return make_quiz(body, SHARED_USER_ID)


@app.post("/api/quizzes/{quiz_id}/submit")
def submit(quiz_id: str, body: SubmitRequest):
    """Grade a shared quiz on the server and then reveal its explanation."""
    with Session.begin() as db:
        quiz = db.get(Quiz, quiz_id)
        if not quiz:
            raise HTTPException(404, "Quiz not found")
        questions = quiz.payload["questions"]
        score = grade_quiz(questions, body.answers)
        db.add(Attempt(quiz_id=quiz.id, user_id=SHARED_USER_ID, answers=body.answers, score=score))
        return {"score": score, "total": len(questions), "questions": questions, "sources": quiz.payload["sources"]}
