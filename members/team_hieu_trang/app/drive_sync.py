"""Download research PDFs plus the catalog/guide and write a local source manifest.

Authentication and downloads happen only when this command runs. Each manifest row
keeps the Drive ID, local file path, title and source URL needed by app.ingest for
indexing and later citations.
"""

import argparse
import json
import os
import re
from pathlib import Path

from pydrive2.auth import GoogleAuth, RefreshError
from pydrive2.drive import GoogleDrive

from .config import ROOT, SOURCE_MANIFEST_PATH
from .study_materials import STUDY_FILENAMES

FOLDER_MIME = "application/vnd.google-apps.folder"


def find_drive_folder_ids(drive: GoogleDrive, names: list[str]) -> list[str]:
    """Resolve exact folder names in My Drive to unique folder IDs.

    Reject missing or duplicate names so synchronization cannot silently choose the
    wrong research collection.
    """
    folder_ids = []
    for name in names:
        # Escape quotes and backslashes so a folder name stays inside the query value.
        escaped = name.replace("\\", "\\\\").replace("'", "\\'")
        query = (
            "'root' in parents and trashed=false "
            f"and mimeType='{FOLDER_MIME}' and title='{escaped}'"
        )
        matches = drive.ListFile({"q": query}).GetList()
        if not matches:
            raise ValueError(f"Google Drive folder not found in My Drive: {name}")
        if len(matches) > 1:
            raise ValueError(
                f"More than one My Drive folder is named {name}. Configure folder IDs instead."
            )
        folder_ids.append(matches[0]["id"])
    return folder_ids


def authenticate_google_drive(client_secrets: Path, credentials: Path) -> GoogleDrive:
    """Create an authenticated Drive client using local OAuth configuration.

    Reuse or refresh saved credentials when possible; otherwise open Google's
    browser consent flow and save the new credentials for the next sync.
    """
    if not client_secrets.is_file():
        raise FileNotFoundError(
            f"Missing {client_secrets}. Download the OAuth JSON, rename it client_secrets.json, "
            "and place it in the starter root."
        )
    gauth = GoogleAuth()
    gauth.settings["client_config_file"] = str(client_secrets)
    gauth.settings["get_refresh_token"] = True
    gauth.LoadCredentialsFile(str(credentials))
    if gauth.credentials is None:
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        try:
            gauth.Refresh()
        except RefreshError:
            # A revoked token cannot be repaired. Remove the local cache and
            # ask Google for fresh consent in the browser.
            credentials.unlink(missing_ok=True)
            gauth = GoogleAuth()
            gauth.settings["client_config_file"] = str(client_secrets)
            gauth.settings["get_refresh_token"] = True
            gauth.LocalWebserverAuth()
    else:
        gauth.Authorize()
    gauth.SaveCredentialsFile(str(credentials))
    return GoogleDrive(gauth)


def walk_drive_files(drive: GoogleDrive, folder_id: str):
    """Yield non-folder files from the selected folder and all its subfolders.

    Keep a list of folders still to visit and exclude trashed items from each Drive
    query.
    """
    pending = [folder_id]
    while pending:
        parent = pending.pop()
        children = drive.ListFile(
            {"q": f"'{parent}' in parents and trashed=false"}
        ).GetList()
        for item in children:
            if item["mimeType"] == FOLDER_MIME:
                pending.append(item["id"])
            else:
                yield item


def download_research_source(item, output_dir: Path):
    """Save a PDF or one of the two named catalog/guide files from Drive.

    Keep unrelated file types excluded. A native Google Doc named after the guide
    is exported as text; its headings are still available as reference content.
    Return the local path for inclusion in the source manifest.
    """
    # Replace characters Windows forbids in filenames before saving the PDF.
    original = (
        re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", item["title"]).strip(" .") or "untitled"
    )
    extension = Path(original).suffix.lower()
    native_guide = item.get(
        "mimeType"
    ) == "application/vnd.google-apps.document" and original.casefold() in {
        "how_to_read_llm_papers",
        "how_to_read_llm_papers.md",
    }
    if native_guide:
        original = "how_to_read_llm_papers.md"
    if extension != ".pdf" and original.casefold() not in STUDY_FILENAMES:
        print(f"SKIP unrelated Drive file: {item['title']}")
        return None
    if original.casefold() in STUDY_FILENAMES:
        original = original.casefold()
    target = output_dir / f"{item['id']}_{original}"
    if native_guide:
        item.GetContentFile(str(target), mimetype="text/plain")
    else:
        item.GetContentFile(str(target))
    return target


def sync_research_folder(
    folder_ids: list[str],
    folder_names: list[str],
    output_dir: Path,
    manifest: Path,
    client_secrets: Path,
    credentials: Path,
):
    """Download PDFs and the named learning references from the selected folder.

    Require one root folder, deduplicate Drive file IDs and keep citation links.
    Reject an empty result rather than replacing the manifest with an accidental
    empty collection.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    drive = authenticate_google_drive(client_secrets, credentials)
    selected_folder_ids = list(folder_ids)
    if folder_names:
        selected_folder_ids.extend(find_drive_folder_ids(drive, folder_names))
    selected_folder_ids = list(dict.fromkeys(selected_folder_ids))
    if len(selected_folder_ids) != 1:
        raise ValueError("Configure exactly one Google Drive research-paper folder")

    drive_rows = []
    seen_file_ids = set()
    for folder_id in selected_folder_ids:
        for item in walk_drive_files(drive, folder_id):
            if item["id"] in seen_file_ids:
                continue
            seen_file_ids.add(item["id"])
            target = download_research_source(item, output_dir)
            if target is None:
                continue
            try:
                local_path = target.relative_to(manifest.parent)
            except ValueError:
                local_path = target
            source_id = f"gdrive-{item['id']}"
            row = {
                "id": source_id,
                "path": local_path.as_posix(),
                "title": next(
                    (
                        name
                        for name in STUDY_FILENAMES
                        if target.name.endswith("_" + name)
                    ),
                    item["title"],
                ),
                "source_type": "google_drive",
                "source_url": item.get("alternateLink")
                or f"https://drive.google.com/open?id={item['id']}",
            }
            drive_rows.append(row)
            print(f"SYNC {item['title']} -> {local_path}")

    combined = sorted(drive_rows, key=lambda row: row["title"].casefold())
    if not any(Path(row["path"]).suffix.lower() == ".pdf" for row in combined):
        raise ValueError("No PDFs were found in the research-paper folder")
    manifest.write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "folders": len(selected_folder_ids),
                "google_drive_sources": len(drive_rows),
                "manifest": str(manifest),
            }
        )
    )


def main():
    """Read sync paths and folder selection from command arguments or .env.

    Pass the resolved settings to sync_research_folder(); embedding and index
    updates happen in the separate ingestion command.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder-id", action="append", dest="folder_ids")
    parser.add_argument("--folder-name", action="append", dest="folder_names")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "synced_sources" / "drive"
    )
    parser.add_argument("--manifest", type=Path, default=SOURCE_MANIFEST_PATH)
    parser.add_argument(
        "--client-secrets", type=Path, default=ROOT / "client_secrets.json"
    )
    parser.add_argument("--credentials", type=Path, default=ROOT / "credentials.json")
    args = parser.parse_args()
    research_id = os.getenv("RESEARCH_FOLDER_ID", "").strip()
    folder_ids = args.folder_ids or ([research_id] if research_id else [])
    folder_names = args.folder_names or []
    if not folder_ids and not folder_names:
        parser.error(
            "Set RESEARCH_FOLDER_ID in .env, or pass --folder-id or --folder-name"
        )
    sync_research_folder(
        folder_ids,
        folder_names,
        args.output_dir.resolve(),
        args.manifest.resolve(),
        args.client_secrets.resolve(),
        args.credentials.resolve(),
    )


if __name__ == "__main__":
    main()
