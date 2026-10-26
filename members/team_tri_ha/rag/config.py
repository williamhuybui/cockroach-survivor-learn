
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"                     
DOCS_FILE = ROOT / "data" / "processed" / "docs.jsonl"  

DRIVE_FOLDER_URL = os.getenv("DRIVE_FOLDER_URL", "")
DRIVE_ONLY = os.getenv("DRIVE_ONLY", "")
GOOGLE_CREDENTIALS = os.getenv("GOOGLE_CREDENTIALS", str(ROOT / "credentials" / "sa.json"))

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma3:4b")
LLM_NUM_CTX = int(os.getenv("LLM_NUM_CTX", "8192")) 
EMBED_MODEL = os.getenv("EMBED_MODEL", "embeddinggemma")

CHUNK_STRATEGY = os.getenv("CHUNK_STRATEGY", "recursive")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "4"))


def chroma_dir(fake: bool = False) -> Path:
    """Vector index folder. Fake embeddings get their own folder so they never mix."""
    return ROOT / "data" / ("chroma_fake" if fake else "chroma")
