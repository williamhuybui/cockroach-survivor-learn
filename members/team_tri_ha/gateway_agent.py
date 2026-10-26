"""Adapter between api_view.py (team2_agent) and the rag.agent.Agent of team Ha & Tri."""
import re
import sys
import threading
from pathlib import Path

HERE = str(Path(__file__).resolve().parent)
DONT_KNOW = "I don't know - this is not in the documents."

_init_lock = threading.Lock()
_ask_lock = threading.Lock()  # one question at a time: the local LLM and the Chroma client are shared
_agent = None


def _get_agent():
    global _agent
    with _init_lock:
        if _agent is None:
            # Our code does `from rag import config` and `from phase3_ask import ...`, so this folder must lead sys.path.
            if HERE in sys.path:
                sys.path.remove(HERE)
            sys.path.insert(0, HERE)
            for name in [m for m in sys.modules if m == "rag" or m.startswith("rag.") or m == "phase3_ask"]:
                mod_file = getattr(sys.modules[name], "__file__", None) or ""
                if not str(Path(mod_file).resolve()).startswith(HERE):
                    del sys.modules[name]  # a different team's module with the same name
            import rag.models  # noqa: F401  (loaded now: phase3_ask imports it lazily later)
            from rag.agent import Agent

            _agent = Agent()
    return _agent


def answer(question: str) -> dict:
    try:
        agent = _get_agent()
        with _ask_lock:
            result = agent.ask(question)
    except Exception as e:
        return {"answer": f"Sorry, the agent failed to answer ({type(e).__name__}: {str(e)[:200]}).", "sources": []}

    text = (result.get("answer") or "").strip()
    if not text or text.lower().startswith("i could not find this in the documents") or "not stated in the documents" in text.lower():
        return {"answer": DONT_KNOW, "sources": []}

    chunks = result.get("trace", {}).get("chunks", [])
    all_sources = list(dict.fromkeys(c["source"] for c in chunks))
    cited = [s for s in all_sources if s in text]
    return {"answer": text, "sources": cited or all_sources}
