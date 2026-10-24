"""Settings shared by the paper index, local models, and web app."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

CHROMA_PATH = Path(os.getenv("CHROMA_PATH", str(ROOT / "data" / "chroma")))
SQLITE_PATH = Path(os.getenv("SQLITE_PATH", str(ROOT / "data" / "app.sqlite3")))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemma3:4b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "embeddinggemma")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")
# A raw relevance score, not a probability; tune using reviewed paper questions.
MIN_RERANK_SCORE = float(os.getenv("MIN_RERANK_SCORE", "0"))
MAX_CHUNK_CHARACTERS = 1_200
CHUNK_OVERLAP_CHARACTERS = 120
TEAM_TIMEZONE = os.getenv("TEAM_TIMEZONE", "America/Chicago")
