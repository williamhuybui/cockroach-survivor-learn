"""Verify the schema keeps a compact history of indexing attempts."""

import json

from app import db, ingest


def test_save_index_run_records_pipeline_identity():
    """An index history row preserves source, configuration, and outcome."""

    ingest.save_index_run(
        "run-1",
        "2026-10-20T00:00:00+00:00",
        "manifest-hash",
        False,
        2,
        "completed",
        {"indexed": 2, "chunks": 10},
    )
    with db.open_database() as database:
        row = database.execute(
            "SELECT manifest_sha256, status, counts FROM index_runs WHERE id = ?",
            ("run-1",),
        ).fetchone()
    assert row["manifest_sha256"] == "manifest-hash"
    assert row["status"] == "completed"
    assert json.loads(row["counts"])["chunks"] == 10
