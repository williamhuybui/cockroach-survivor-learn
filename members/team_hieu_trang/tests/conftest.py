"""Keep every automated test away from the real paper index and chat history."""

import pytest

from app import db, paper_store, retrieval


@pytest.fixture(autouse=True)
def local_storage(tmp_path, monkeypatch):
    """Use disposable SQLite and Chroma paths for each test."""

    monkeypatch.setattr(db, "SQLITE_PATH", tmp_path / "app.sqlite3")
    monkeypatch.setattr(paper_store, "CHROMA_PATH", tmp_path / "chroma")
    db.initialize_database()
    retrieval.all_records.cache_clear()
