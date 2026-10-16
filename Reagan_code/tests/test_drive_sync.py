from app.drive_sync import find_root_folders, split_values


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


def test_split_values_removes_blanks_and_duplicates():
    assert split_values("Meeting notes, RAG-data, Meeting notes, ") == [
        "Meeting notes",
        "RAG-data",
    ]


def test_find_root_folders_resolves_both_names():
    drive = FakeDrive({"Meeting notes": "meeting-id", "RAG-data": "rag-id"})

    assert find_root_folders(drive, ["Meeting notes", "RAG-data"]) == [
        "meeting-id",
        "rag-id",
    ]
