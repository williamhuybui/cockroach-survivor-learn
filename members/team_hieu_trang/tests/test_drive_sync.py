"""Test Drive source selection and PDF manifest behavior with fake Drive objects.

Validate folder scope and file handling without authenticating to or modifying real
Drive files.
"""

from app.drive_sync import find_drive_folder_ids


def test_synced_manifest_keeps_source_identity_for_ingestion(tmp_path, monkeypatch):
    """Keep citation and paper identity without carrying unused Drive folder metadata."""
    import json
    from pathlib import Path

    from app import drive_sync, ingest

    class DriveFile(dict):
        def GetContentFile(self, path):
            Path(path).write_bytes(b"downloaded PDF fixture")

    item = DriveFile(
        id="paper-1",
        title="Paper.pdf",
        mimeType="application/pdf",
        parents=[{"id": "folder-1"}],
        alternateLink="https://drive.google.com/open?id=paper-1",
    )
    monkeypatch.setattr(drive_sync, "authenticate_google_drive", lambda *args: object())
    monkeypatch.setattr(drive_sync, "walk_drive_files", lambda *args: [item])
    manifest = tmp_path / "sources.json"
    drive_sync.sync_research_folder(
        ["folder-1"],
        [],
        tmp_path / "downloads",
        manifest,
        tmp_path / "client.json",
        tmp_path / "credentials.json",
    )

    saved = json.loads(manifest.read_text())[0]
    assert saved["id"] == "gdrive-paper-1"
    assert saved["source_url"] == item["alternateLink"]
    assert "parent_ids" not in saved
    paper = ingest.load_source_manifest(manifest)[0]
    assert paper["id"] == saved["id"]
    assert paper["_path"].read_bytes() == b"downloaded PDF fixture"


def test_non_pdf_is_skipped(tmp_path):
    """Research-only sync must not download native Docs or other file types."""
    from app.drive_sync import download_research_source

    class DriveFile(dict):
        def GetContentFile(self, path, mimetype):
            self["export_mime"] = mimetype

    item = DriveFile(
        id="123", title="Notes", mimeType="application/vnd.google-apps.document"
    )
    result = download_research_source(item, tmp_path)
    assert result is None
    assert "export_mime" not in item


class FakeQuery:
    def __init__(self, rows):
        self.rows = rows

    def GetList(self):
        return self.rows


class FakeDrive:
    def __init__(self, folders):
        self.folders = folders

    def ListFile(self, request):
        query = request["q"]
        rows = [
            {"id": folder_id, "title": name}
            for name, folder_id in self.folders.items()
            if f"title='{name}'" in query
        ]
        return FakeQuery(rows)


def test_find_root_folders_resolves_both_names():
    drive = FakeDrive({"Meeting notes": "meeting-id", "RAG-data": "rag-id"})

    assert find_drive_folder_ids(drive, ["Meeting notes", "RAG-data"]) == [
        "meeting-id",
        "rag-id",
    ]
