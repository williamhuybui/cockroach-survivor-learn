import os
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://team:team@localhost:5432/team_knowledge")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemma3:4b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "embeddinggemma")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))
TEAM_TIMEZONE = os.getenv("TEAM_TIMEZONE", "America/Chicago")
