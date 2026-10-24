from app.drive_sync import find_root_folders


def test_non_pdf_is_skipped(tmp_path):
    """Research-only sync must not download native Docs or other file types."""
    from app.drive_sync import download_supported

    class DriveFile(dict):
        def GetContentFile(self, path, mimetype):
            self["export_mime"] = mimetype

    item = DriveFile(
        id="123", title="Notes", mimeType="application/vnd.google-apps.document"
    )
    result = download_supported(item, tmp_path)
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

    assert find_root_folders(drive, ["Meeting notes", "RAG-data"]) == [
        "meeting-id",
        "rag-id",
    ]
