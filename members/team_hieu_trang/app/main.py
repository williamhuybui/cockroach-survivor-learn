"""Expose the browser chat, saved conversations and quiz routes through FastAPI.

Validate incoming browser actions, load/save SQLite state and delegate research work
to services.py. The browser submits questions or approves a saved plan; these routes
do not download or rebuild the paper index.
"""

import json
import logging
from datetime import date
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from .challenge import get_or_create_daily_challenge, submit_daily_challenge
from .db import create_record_id, open_database
from .paper_store import list_indexed_papers, open_chunk_store
from .quiz_validation import SubmitRequest, grade_quiz
from .services import answer_research_question, handle_chat_message

STATIC = Path(__file__).parent / "static"
logger = logging.getLogger(__name__)
app = FastAPI(title="Team Knowledge Agent")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.exception_handler(ValueError)
def handle_validation_error(request: Request, exc: ValueError):
    """Return HTTP 422 for invalid requests or service input.

    Give the browser a readable message so it can show a validation problem rather
    than treating it as a server crash.
    """

    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(Exception)
def handle_service_error(request: Request, exc: Exception):
    """Log unexpected failures and return a generic HTTP 503 response.

    The server log retains diagnostic details while the browser receives no secrets
    or paper text from exception messages.
    """

    logger.exception("Request failed: %s", request.url.path)
    return JSONResponse(
        status_code=503,
        content={
            "detail": "A local model or database operation failed. Check the server log."
        },
    )


@app.get("/")
def serve_chat_page():
    """Return the HTML entry page whose script calls the chat and quiz APIs."""

    return FileResponse(STATIC / "index.html")


@app.get("/api/sources")
def get_indexed_paper_sources():
    """Return the current indexed-paper catalog for the browser's source display."""

    return list_indexed_papers()


@app.get("/api/sessions/{session_id}")
def get_chat_session(session_id: str):
    """Load a saved conversation and its pending plan ID from SQLite.

    The browser uses these to restore messages and enable only the approval controls
    belonging to the current proposal.
    """

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
def delete_chat_session(session_id: str):
    """Delete one saved conversation and report a missing ID as HTTP 404.

    SQLite removes its pending proposal through the foreign-key relationship.
    """

    with open_database() as database:
        deleted = database.execute(
            "DELETE FROM chat_sessions WHERE id = ?", (session_id,)
        )
    if deleted.rowcount == 0:
        raise HTTPException(404, "Chat session not found")
    return Response(status_code=204)


# The browser sends either new text or an approve/revise action for a saved plan.
# Pydantic checks the shape here before the route loads history or calls the graph.
class ChatRequest(BaseModel):
    """Validate a browser message and its optional conversation identifier."""

    message: str = Field(default="", max_length=2000)
    session_id: str | None = None
    action: Literal["message", "approve", "revise"] = "message"
    plan_id: str | None = None

    @model_validator(mode="after")
    def validate_chat_action(self):
        """Check that messages contain text and review actions identify a saved proposal.

        Approval needs a session ID and plan ID; ordinary and revision messages also
        need nonblank text.
        """
        if self.action != "approve" and not self.message.strip():
            raise ValueError("A non-empty message is required")
        if self.action != "message" and (not self.plan_id or not self.session_id):
            raise ValueError("Plan review requires a session ID and plan ID")
        return self


@app.post("/api/chat")
def handle_chat_request(body: ChatRequest):
    """Process a message, approval or revision and save the resulting conversation.

    Load history and claim any pending plan before work so duplicate actions cannot
    run it twice. Execute approved content or request a revision, release the plan
    after failures, then save the assistant response and any next proposal.
    """

    session_id = body.session_id or create_record_id()
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
            result = answer_research_question(
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
                handle_chat_message(question, history, True)
                if pending
                else handle_chat_message(question, history)
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
def list_chat_sessions():
    """Return recent conversation IDs and titles derived from their first message.

    The sidebar uses this list to reopen chats without storing a second history in
    the browser.
    """

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
def get_index_status():
    """Report whether Chroma contains chunks and when ingestion last completed.

    This is an index-readiness indicator, not a live check of every model service.
    """

    ready = open_chunk_store().count() > 0
    with open_database() as database:
        row = database.execute(
            "SELECT value FROM settings WHERE key = ?", ("last_ingested_at",)
        ).fetchone()
    return {"ready": ready, "synced_at": row["value"] if row else None}


@app.get("/api/daily-challenge")
def get_daily_challenge():
    """Return the saved daily question or generate today's question on first access."""

    return get_or_create_daily_challenge()


@app.post("/api/daily-challenge/{day}/submit")
def handle_daily_challenge_submission(day: date, body: SubmitRequest):
    """Grade the question for the date the browser displayed.

    Using that date keeps a submission tied to its original daily challenge even if
    midnight has passed.
    """

    return submit_daily_challenge(day, body.answers)


@app.post("/api/quizzes/{quiz_id}/submit")
def handle_quiz_submission(quiz_id: str, body: SubmitRequest):
    """Grade submitted option indexes against the saved quiz key and record the attempt.

    Return the score, explanations and source links after submission; grading
    requires no model call.
    """

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
            (create_record_id(), quiz_id, json.dumps(body.answers), score),
        )
    return {
        "score": score,
        "total": len(questions),
        "questions": questions,
        "sources": payload["sources"],
    }
