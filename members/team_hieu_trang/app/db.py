"""Store application history and small bookkeeping records in local SQLite.

Chat messages, pending plans, quizzes, attempts, settings and index-run summaries
use schema.sql. Research passages and embedding vectors live in Chroma instead. Use
open_database() so transactions and connections are cleaned up consistently.
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from .config import SQLITE_PATH

SCHEMA_FILE = Path(__file__).with_name("schema.sql")


def create_record_id() -> str:
    """Create a new unique ID (UUID) for a chat, proposal, quiz, attempt or index run.

    Callers save this ID with the record so later requests can refer to the same
    item.
    """

    return str(uuid4())


@contextmanager
def open_database():
    """Provide a SQLite connection for a with block and always close it afterward.

    Commit successful writes and roll back failures. Enable foreign keys and named-
    column access so callers can safely share the same storage pattern.
    """

    SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    database = sqlite3.connect(SQLITE_PATH, timeout=15)
    database.row_factory = sqlite3.Row
    database.execute("PRAGMA foreign_keys = ON")
    try:
        with database:
            yield database
    finally:
        database.close()


def initialize_database() -> None:
    """Create any missing tables from schema.sql without clearing saved records.

    Called during application/setup flows so chat, quiz and indexing bookkeeping
    have the required tables.
    """

    with open_database() as database:
        database.executescript(SCHEMA_FILE.read_text(encoding="utf-8"))
