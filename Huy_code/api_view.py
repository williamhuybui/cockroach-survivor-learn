import requests
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Message(BaseModel):
    text: str


@app.post("/chat")
def send_message(message: Message):
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

from langgraph.graph import StateGraph, START, END

csv.field_size_limit(sys.maxsize)  # the content column is huge
MAX_CHARS = 8000  # how much of the document the model sees (keeps it inside Ollama's context)
TMP_DIR = Path(__file__).resolve().parent.parent / "tmp"


def ask_llm(prompt: str) -> str:
    payload = {"model": "gemma3", "messages": [{"role": "user", "content": prompt}], "stream": False}
    response = requests.post("http://localhost:11434/api/chat", json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["message"]["content"].strip()


def load_documents() -> list[dict]:
    """Every row (file_name, file_link, content) from every csv in tmp/."""
    rows = []
    for csv_path in TMP_DIR.glob("*.csv"):
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
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


@app.post("/chat-v2")
def send_message_v2(message: Message):
    result = chat_graph.invoke({"text": message.text})
    return {"intent": result["intent"], "message": result["answer"]}
