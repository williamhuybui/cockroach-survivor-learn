"""Huy's agent (team3). Currently a placeholder: keyword (TF-IDF style) retrieval over the documents, answered by gemma3.

api_view.py calls answer(question, ask_llm, load_documents):
  ask_llm(prompt) -> str          gemma3 through Ollama
  load_documents() -> list[dict]  rows with file_name, file_link, content (from tmp/)
and expects {"answer": str, "sources": [str, ...]} back. Develop here; api_view.py only forwards to answer().
"""
import math
import re
import threading
from collections import Counter

CHUNK, OVERLAP, TOP_K = 1500, 200, 4
STOPWORDS = set("the a an of in on for to and or is are was were what which who how does do did with by as at from that this it its be".split())
_index = None
_index_lock = threading.Lock()  # the evaluation asks several questions at once


def tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


def get_index(load_documents):
    global _index
    with _index_lock:
        if _index is None:
            chunks = []
            for doc in load_documents():
                text = doc["content"]
                for start in range(0, max(len(text), 1), CHUNK - OVERLAP):
                    piece = text[start:start + CHUNK]
                    chunks.append({"file": doc["file_name"], "text": piece, "tf": Counter(tokens(piece))})
            _index = (chunks, Counter(w for c in chunks for w in c["tf"]))
        return _index


def answer(question: str, ask_llm, load_documents) -> dict:
    chunks, df = get_index(load_documents)
    q = set(tokens(question))

    def score(c):
        return sum((1 + math.log(c["tf"][w])) * math.log(1 + len(chunks) / df[w]) for w in q if w in c["tf"])

    hits = [c for c in sorted(chunks, key=score, reverse=True)[:TOP_K] if score(c) > 0]
    if not hits:
        return {"answer": "I don't know - this is not in the documents.", "sources": []}
    context = "\n\n".join(f"[{c['file']}]\n{c['text']}" for c in hits)
    answer = ask_llm(
        "Answer the question using ONLY the excerpts below. Be concise and specific.\n"
        "If the excerpts do not contain the answer, reply exactly: I don't know - this is not in the documents.\n\n"
        f"{context}\n\nQuestion: {question}")
    return {"answer": answer, "sources": list(dict.fromkeys(c["file"] for c in hits))}
