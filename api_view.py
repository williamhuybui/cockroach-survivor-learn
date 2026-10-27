import sys
import time
from typing import Literal, Optional

import requests
from fastapi import FastAPI, HTTPException, Path as PathParam, Query
from pydantic import BaseModel, Field

app = FastAPI(
    title="Cockroach Survivor API",
    description="Chat demos (/chat, /chat-v2) and the evaluation gateway (/{team}/rag-agent, /{team}/evaluate, /{team}/health).\n\n"
    "Teams: team1 = Hieu (Captain), Trang; team2 = Ha (Captain), Tri; team3 = Huy.",
)


class Message(BaseModel):
    text: str = Field(description="The user's message.")


@app.post("/chat", summary="Plain chat with gemma3")
def send_message(message: Message):
    """Send `text` straight to the local Ollama gemma3 model and return its reply as `message`. No documents are used."""
    payload = {
        "model": "gemma3",
        "messages": [
            {
                "role": "user",
                "content": message.text
            }
        ],
        "stream": False
    }
    response = requests.post("http://localhost:11434/api/chat", json=payload, timeout=120)
    response.raise_for_status()
    return {"message": response.json()["message"]["content"]}

# ---------------------------------------------------------------------------
# /chat-v2: LangGraph flow
#   classify -> list_documents                        (what documents do you have)
#   classify -> chat                                  (general chat)
#   classify -> find_document -> not_found            (no matching document)
#                             -> answer_document      (answer any question about it)
# ---------------------------------------------------------------------------
import csv
import sys
from pathlib import Path
from typing import Optional, TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so members.<folder> imports work from anywhere

from langgraph.graph import StateGraph, START, END

csv.field_size_limit(sys.maxsize)  # the content column is huge
MAX_CHARS = 8000  # how much of the document the model sees (keeps it inside Ollama's context)
TMP_DIR = Path(__file__).resolve().parent / "tmp"


def ask_llm(prompt: str) -> str:
    payload = {"model": "gemma3", "messages": [{"role": "user", "content": prompt}], "stream": False}
    response = requests.post("http://localhost:11434/api/chat", json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["message"]["content"].strip()


def load_documents() -> list[dict]:
    """Every document row (file_name, file_link, content) from the csv files in tmp/."""
    rows = []
    for csv_path in TMP_DIR.glob("*.csv"):
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows.extend(r for r in csv.DictReader(f) if r.get("file_name") and r.get("content"))  # skip tables like question_set.csv
    return rows


class State(TypedDict):
    text: str
    intent: str
    document: Optional[dict]
    answer: str


def classify(state: State):
    reply = ask_llm(
        "Classify the user's message. Answer with one word:\n"
        "list - the user asks what documents/files exist or are available, without naming one "
        "(e.g. 'what documents do you have', 'list your files', 'what can I ask about')\n"
        "document - the user names or describes ONE specific document and wants to find it, read it, or ask about its content "
        "(e.g. 'where is the BERT paper', 'summarize the Word2Vec paper')\n"
        "chat - anything else\n\n"
        f"User: {state['text']}"
    ).lower()
    intent = next((i for i in ["list", "document"] if i in reply), "chat")
    return {"intent": intent}


def list_documents(state: State):
    names = [d["file_name"] for d in load_documents()]
    return {"answer": f"I have {len(names)} documents:\n" + "\n".join(names)}


def chat(state: State):
    return {"answer": ask_llm(state["text"])}


def find_document(state: State):
    documents = load_documents()
    names = "\n".join(d["file_name"] for d in documents)
    reply = ask_llm(
        "Which file from this list is the user asking for? "
        "Answer with the exact file name only, or NONE if it is not in the list.\n\n"
        f"Files:\n{names}\n\nUser: {state['text']}"
    )
    match = next((d for d in documents if d["file_name"] == reply.strip(" \"'`")), None)
    return {"document": match}


def not_found(state: State):
    return {"answer": "I couldn't find that document in the file list."}


def answer_document(state: State):
    doc = state["document"]
    return {"answer": ask_llm(
        "Answer the user's question using this document. "
        "You can also use the file name and link if the user asks where it is.\n\n"
        f"File name: {doc['file_name']}\nFile link: {doc['file_link']}\n"
        f"Content (first {MAX_CHARS} characters):\n{doc['content'][:MAX_CHARS]}\n\n"
        f"User: {state['text']}"
    )}


graph = StateGraph(State)
for node in [classify, chat, list_documents, find_document, not_found, answer_document]:
    graph.add_node(node.__name__, node)

graph.add_edge(START, "classify")
graph.add_conditional_edges("classify", lambda s: s["intent"], {"chat": "chat", "list": "list_documents", "document": "find_document"})
graph.add_conditional_edges(
    "find_document", lambda s: "found" if s["document"] else "missing",
    {"found": "answer_document", "missing": "not_found"},
)
for node in ["chat", "list_documents", "not_found", "answer_document"]:
    graph.add_edge(node, END)

chat_graph = graph.compile()


@app.post("/chat-v2", summary="LangGraph chat over the documents in tmp/")
def send_message_v2(message: Message):
    """Classify `text` as list / document / chat, then answer: list the documents in tmp/*.csv, answer about one
    named document, or chat normally. Returns the detected `intent` and the reply as `message`."""
    result = chat_graph.invoke({"text": message.text})
    return {"intent": result["intent"], "message": result["answer"]}


# ---------------------------------------------------------------------------
# Evaluation gateway: POST /{team}/rag-agent, POST /{team}/evaluate, GET /{team}/health
#
# Every team implements ONE function below, `teamN_agent(question) -> {"answer": str, "sources": [str, ...]}`,
# in its own marked section. Rules:
#   - answer: the final answer as plain text; if the documents do not contain it, say so explicitly
#     ("I don't know - this is not in the documents."). Wrong answers cost -0.5, abstaining costs 0.
#   - sources: document names (strings) used for the answer, [] if none.
#   - every question is independent (no session or chat history).
#   - use gemma3 through ask_llm() above; read the documents with load_documents() (tmp/*.csv) or your own index.
# Put helpers and imports for your team inside your section. Do not edit other teams' sections.
# ---------------------------------------------------------------------------
def not_implemented(team: str):
    """Placeholder for a team's agent. Replace `teamN_agent = not_implemented(...)` with a real `def`."""
    def agent(question: str) -> dict:
        raise NotImplementedError(f"{team} has not implemented its agent yet")
    agent.placeholder = True
    return agent


# ============================== TEAM 1: Hieu & Trang ==============================
# team1_agent = not_implemented("team1")
def team1_agent(question: str) -> dict:
    """Connect Team 1's existing LangGraph RAG agent to the shared API."""

    from members.team_hieu_trang.app.services import answer_with_context

    result = answer_with_context(
        question,
        history=[],
        capture_evidence=False,
        require_review=False,
        allow_clarification=False,
    )

    # Extract the answer
    answer = str(result.get("answer") or "").strip()

    if not answer:
        answer = "I don't know - this is not in the documents."

    # Convert source metadata to document titles
    sources = []

    for source in result.get("sources") or []:
        if isinstance(source, dict):
            title = source.get("title") or source.get("source_url")
        else:
            title = source

        if title:
            sources.append(str(title))

    return {
        "answer": answer,
        "sources": list(dict.fromkeys(sources)),
    }

# ============================== TEAM 2: Ha & Tri ==================================
def team2_agent(question: str) -> dict:
    from members.team_tri_ha.gateway_agent import answer as t2_answer  # imported on first use; adds our folder to sys.path

    return t2_answer(question)


# ============================== TEAM 3: Huy (code in members/team_huy/agent.py) ===
def team3_agent(question: str) -> dict:
    from members.team_huy.agent import answer  # imported on first use so a bug here cannot break other teams

    return answer(question, ask_llm, load_documents)


# ============================== gateway (do not edit per team) ====================
AGENTS = {"team1": team1_agent, "team2": team2_agent, "team3": team3_agent}
Team = Literal["team1", "team2", "team3"]
TEAM_MEMBERS = "team1 = Hieu (Captain), Trang; team2 = Ha (Captain), Tri; team3 = Huy"
TeamParam = PathParam(description=f"Which team's agent to use. {TEAM_MEMBERS}.")


class Question(BaseModel):
    question: str = Field(description="One question for the team's RAG agent.")


def source_names(sources: list) -> list[str]:
    return [str(s.get("title") or s.get("source_url") or s) if isinstance(s, dict) else str(s) for s in sources or []]


def ask_agent(team: str, question: str) -> dict:
    if getattr(AGENTS[team], "placeholder", False):
        raise HTTPException(501, f"{team} has not implemented its agent yet (see its section in api_view.py)")
    result = AGENTS[team](question)
    return {"answer": str(result.get("answer", "")), "sources": source_names(result.get("sources"))}


@app.post("/{team}/rag-agent", summary="Ask one question to a team's agent")
def rag_agent(body: Question, team: Team = TeamParam):
    """Run the chosen team's `teamN_agent(question)` and return `{answer, sources}`, where `sources` is a list of
    document names or titles. Returns 501 if the team has not implemented its agent yet."""
    return ask_agent(team, body.question)


@app.post("/{team}/evaluate", summary="Run a team's agent over a whole question set")
def evaluate_team(
    team: Team = TeamParam,
    set: Literal["test", "dev"] = Query("test", description="Question set: `test` = hidden evaluation/test_set.csv, `dev` = public dev_set.csv."),
    limit: Optional[int] = Query(None, ge=1, description="Only run the first N questions (quick check). Limited runs never update the leaderboard."),
    workers: int = Query(4, ge=1, le=16, description="How many questions to ask in parallel. 1 = one at a time. Your agent must be thread-safe; Ollama may still serve gemma3 calls one at a time, so the speed-up depends on its OLLAMA_NUM_PARALLEL setting."),
    judge_model: str = Query("gpt-4o-mini", description="OpenAI model that grades each answer as correct / partial / idk / hallucinated (needs OPENAI_API_KEY)."),
):
    """No question needed: asks the team's agent every question in the set, grades each answer, and returns the score,
    verdict counts, per-type scores, retrieval hit rate and error count. A full run of the test set also updates
    one row (time, team, mode, score) is appended to docs/scores.csv, which the leaderboard page reads. Per-question answers go to evaluation/eval_results/,
    not into the response. Blocks until done (about 5 seconds per question). Returns 429 if the practice limit applies,
    501 if the team has not implemented its agent."""
    if getattr(AGENTS[team], "placeholder", False):
        raise HTTPException(501, f"{team} has not implemented its agent yet (see its section in api_view.py)")
    sys.path.insert(0, str(Path(__file__).resolve().parent / "evaluation"))
    import evaluate

    def ask(question):
        t0 = time.time()
        try:
            r = ask_agent(team, question)
            return r["answer"].strip(), r["sources"], time.time() - t0, ""
        except Exception as e:  # a crashed or slow answer scores as "idk" and is flagged
            return "", [], time.time() - t0, f"{type(e).__name__}: {e}"

    path = evaluate.TEST_SET if set == "test" else Path(__file__).resolve().parent / "dev_set.csv"
    try:
        run, out, updated = evaluate.run_evaluation(team, ask, "practice", path, limit, judge_model, workers=workers)
    except RuntimeError as e:
        raise HTTPException(429, str(e))
    return {"team": team, "set": set, "leaderboard_updated": updated, "results_file": out.name, **run}


@app.get("/{team}/health", summary="Check that a team has an agent plugged in")
def team_health(team: Team = TeamParam):
    """Return `{ok: true}` if the team has implemented its agent, otherwise 501. `evaluate.py` calls this before a run."""
    if getattr(AGENTS[team], "placeholder", False):
        raise HTTPException(501, f"{team} has not implemented its agent yet (see its section in api_view.py)")
    return {"ok": True}
