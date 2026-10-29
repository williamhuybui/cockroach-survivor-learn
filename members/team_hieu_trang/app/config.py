"""Read shared model settings and local storage paths from environment variables.

Load the project .env first, then apply defaults. Indexing, retrieval and the web
app import these constants to agree on vector dimensions, models and storage
locations. The PDF warning interval reports slow work without cancelling it.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# Chroma keeps paper text/vectors; SQLite keeps conversations, quizzes and run history.
CHROMA_PATH = Path(os.getenv("CHROMA_PATH", str(ROOT / "data" / "chroma")))
SQLITE_PATH = Path(os.getenv("SQLITE_PATH", str(ROOT / "data" / "app.sqlite3")))
# The models serve different jobs: generate text, embed meaning and score relevance.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemma3:4b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "embeddinggemma")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")
# A raw relevance score, not a probability; tune using reviewed paper questions.
MIN_RERANK_SCORE = float(os.getenv("MIN_RERANK_SCORE", "0"))
# Chunk size includes its repeated title/heading prefix; overlap repeats nearby text.
MAX_CHUNK_CHARACTERS = 1_200
CHUNK_OVERLAP_CHARACTERS = 120
TEAM_TIMEZONE = os.getenv("TEAM_TIMEZONE", "America/Chicago")

# Progress warnings never stop neural extraction or trigger fallback.
PDF_LAYOUT_WARN_SECONDS = float(os.getenv("PDF_LAYOUT_WARN_SECONDS", "30"))
PDF_EXTRACTION_REPORT_PATH = ROOT / "data" / "extraction_reports"
# The same manifest identifies PDFs and the two small learning-reference files.
SOURCE_MANIFEST_PATH = Path(
    os.getenv("SOURCE_MANIFEST_PATH", str(ROOT / "sources.local.json"))
)
