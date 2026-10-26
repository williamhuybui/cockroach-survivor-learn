"""FastAPI routes for the research-paper chat and existing quiz interface."""

import json
import logging
from datetime import date
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from .challenge import get_daily, submit_daily
from .db import new_id, open_database
from .paper_store import collection, source_rows
from .quiz_validation import SubmitRequest, grade_quiz
from .services import answer_with_context, respond

STATIC = Path(__file__).parent / "static"
logger = logging.getLogger(__name__)
app = FastAPI(title="Team Knowledge Agent")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.exception_handler(ValueError)
def bad_value(request: Request, exc: ValueError):
    """Return validation errors without turning them into server failures."""

    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(Exception)
def service_failure(request: Request, exc: Exception):
    """Keep secrets and paper text out of unexpected error responses."""

    logger.exception("Request failed: %s", request.url.path)
    return JSONResponse(
        status_code=503,
        content={
            "detail": "A local model or database operation failed. Check the server log."
        },
    )


@app.get("/")
def index():
    """Serve the plain HTML chat interface."""

    return FileResponse(STATIC / "index.html")


@app.get("/api/sources")
def sources():
    """Show the research PDFs currently indexed in Chroma."""

    return source_rows()


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    """Restore one saved browser conversation."""

    with open_database() as database:
        row = database.execute(
            "SELECT messages FROM chat_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        pending = database.execute(
            "SELECT plan_id FROM pending_plans WHERE session_id = ? AND status = 'pending'",
            (session_id,),
        ).fetchone()
    if not row:
        raise HTTPException(404, "Chat session not found")
    return {
        "session_id": session_id,
        "messages": json.loads(row["messages"]),
        "pending_plan_id": pending["plan_id"] if pending else None,
    }


@app.delete("/api/sessions/{session_id}", status_code=204)
def delete_session(session_id: str):
    """Delete a conversation from the recent-chat list."""

    with open_database() as database:
        deleted = database.execute(
            "DELETE FROM chat_sessions WHERE id = ?", (session_id,)
        )
    if deleted.rowcount == 0:
        raise HTTPException(404, "Chat session not found")
    return Response(status_code=204)


class ChatRequest(BaseModel):
    """Validate a browser message and its optional conversation identifier."""

    message: str = Field(default="", max_length=2000)
    session_id: str | None = None
    action: Literal["message", "approve", "revise"] = "message"
    plan_id: str | None = None

    @model_validator(mode="after")
    def valid_action(self):
        """Require text for requests and exact saved-plan tokens for review actions."""
        if self.action != "approve" and not self.message.strip():
            raise ValueError("A non-empty message is required")
        if self.action != "message" and (not self.plan_id or not self.session_id):
            raise ValueError("Plan review requires a session ID and plan ID")
        return self


@app.post("/api/chat")
def chat(body: ChatRequest):
    """Save a stable chat ID before the slow model call, then append its answer."""

    session_id = body.session_id or new_id()
    with open_database() as database:
        database.execute(
            "INSERT OR IGNORE INTO chat_sessions (id) VALUES (?)", (session_id,)
        )
        row = database.execute(
            "SELECT messages FROM chat_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        history = json.loads(row["messages"])
        pending = database.execute(
            "SELECT * FROM pending_plans WHERE session_id = ?", (session_id,)
        ).fetchone()
        if pending and pending["status"] == "running":
            raise HTTPException(409, "This plan is already running or being revised")
        if body.action != "message" and (
            not pending or body.plan_id != pending["plan_id"]
        ):
            raise HTTPException(
                409, "This plan is no longer current. Reopen the conversation."
            )
        if pending:
            claimed = database.execute(
                "UPDATE pending_plans SET status = 'running' WHERE session_id = ? AND plan_id = ? AND status = 'pending'",
                (session_id, pending["plan_id"]),
            )
            if claimed.rowcount != 1:
                raise HTTPException(
                    409, "The plan changed while your request was submitted"
                )
    try:
        if body.action == "approve":
            saved_plan = json.loads(pending["payload"])
            result = answer_with_context(
                saved_plan["question"],
                history,
                capture_evidence=False,
                approved_plan=saved_plan,
                require_review=True,
            )
            result.pop("context", None)
        else:
            question = body.message
            if pending and not body.message.strip().startswith("/"):
                saved_plan = json.loads(pending["payload"])
                question = (
                    saved_plan["question"]
                    + "\nRequested plan revision: "
                    + body.message
                )
            result = (
                respond(question, history, True)
                if pending
                else respond(question, history)
            )
    except Exception:
        # Failed generation must not leave the saved proposal permanently locked.
        if pending:
            with open_database() as database:
                database.execute(
                    "UPDATE pending_plans SET status = 'pending' WHERE session_id = ? AND plan_id = ?",
                    (session_id, pending["plan_id"]),
                )
        raise
    assistant_message = {
        "role": "assistant",
        "content": result.get("answer", "Quiz created"),
        "sources": result.get("sources", []),
    }
    if result["kind"] == "quiz":
        assistant_message["quiz"] = dict(result)
    if result.get("plan"):
        assistant_message["plan"] = result["plan"]
    human_text = "Approve plan" if body.action == "approve" else body.message
    updated = history + [{"role": "human", "content": human_text}, assistant_message]
    with open_database() as database:
        saved = database.execute(
            "UPDATE chat_sessions SET messages = ? WHERE id = ?",
            (json.dumps(updated), session_id),
        )
        if saved.rowcount:
            database.execute(
                "DELETE FROM pending_plans WHERE session_id = ?", (session_id,)
            )
            if result.get("plan"):
                database.execute(
                    "INSERT INTO pending_plans (session_id, plan_id, payload) VALUES (?, ?, ?)",
                    (session_id, result["plan"]["plan_id"], json.dumps(result["plan"])),
                )
    if saved.rowcount == 0:
        raise HTTPException(409, "This chat was deleted while the answer was generated")
    return {
        **result,
        "session_id": session_id,
        "pending_plan_id": result.get("plan", {}).get("plan_id"),
    }


@app.get("/api/sessions")
def sessions():
    """Use the first user question as each recent chat's title."""

    with open_database() as database:
        rows = database.execute(
            "SELECT id, messages FROM chat_sessions ORDER BY rowid DESC"
        ).fetchall()
    chats = []
    for row in rows:
        messages = json.loads(row["messages"])
        if messages:
            chats.append(
                {"session_id": row["id"], "title": messages[0]["content"][:70]}
            )
    return chats


@app.get("/api/status")
def status():
    """Show whether any papers are indexed and when ingestion last completed."""

    ready = collection().count() > 0
    with open_database() as database:
        row = database.execute(
            "SELECT value FROM settings WHERE key = ?", ("last_ingested_at",)
        ).fetchone()
    return {"ready": ready, "synced_at": row["value"] if row else None}


@app.get("/api/daily-challenge")
def daily():
    """Return today's quiz question, creating it if needed."""

    return get_daily()


@app.post("/api/daily-challenge/{day}/submit")
def daily_submit(day: date, body: SubmitRequest):
    """Grade the displayed day even when midnight has passed."""

    return submit_daily(day, body.answers)


@app.post("/api/quizzes/{quiz_id}/submit")
def submit(quiz_id: str, body: SubmitRequest):
    """Grade a saved quiz and reveal its answer explanations."""

    with open_database() as database:
        row = database.execute(
            "SELECT payload FROM quizzes WHERE id = ?", (quiz_id,)
        ).fetchone()
        if not row:
            raise HTTPException(404, "Quiz not found")
        payload = json.loads(row["payload"])
        questions = payload["questions"]
        score = grade_quiz(questions, body.answers)
        database.execute(
            "INSERT INTO quiz_attempts (id, quiz_id, answers, score) VALUES (?, ?, ?, ?)",
            (new_id(), quiz_id, json.dumps(body.answers), score),
        )
    return {
        "score": score,
        "total": len(questions),
        "questions": questions,
        "sources": payload["sources"],
    }
