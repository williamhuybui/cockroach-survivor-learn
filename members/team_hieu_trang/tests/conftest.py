"""Isolate automated tests from the real index and application history.

Redirect SQLite, Chroma and extraction reports into temporary folders for every
test. Test runs must never clear or overwrite the papers being indexed by the user.
"""

import pytest

from app import db, document_processing, paper_store, study_materials


@pytest.fixture(autouse=True)
def local_storage(tmp_path, monkeypatch):
    """Use disposable SQLite and Chroma paths for each test."""

    monkeypatch.setattr(db, "SQLITE_PATH", tmp_path / "app.sqlite3")
    monkeypatch.setattr(paper_store, "CHROMA_PATH", tmp_path / "chroma")
    monkeypatch.setattr(
        study_materials, "SOURCE_MANIFEST_PATH", tmp_path / "sources.json"
    )
    monkeypatch.setattr(
        document_processing,
        "PDF_EXTRACTION_REPORT_PATH",
        tmp_path / "extraction_reports",
    )
    db.initialize_database()
