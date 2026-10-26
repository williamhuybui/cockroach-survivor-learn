"""SQLite stores chats, quizzes, settings and index-run summaries; passages live in Chroma."""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from .config import SQLITE_PATH

SCHEMA_FILE = Path(__file__).with_name("schema.sql")


def new_id() -> str:
    """Give each chat, quiz, or attempt a stable identifier for the browser."""

    return str(uuid4())


@contextmanager
def open_database():
    """Commit successful writes, roll back failures, and always close SQLite.

    A plain ``with sqlite3.connect(...)`` handles the transaction but does not
    close the connection. This small wrapper avoids repeating that cleanup.
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
    """Create the small local tables used by the existing chat and quiz UI."""

    with open_database() as database:
        database.executescript(SCHEMA_FILE.read_text(encoding="utf-8"))
