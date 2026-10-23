"""Download PDFs from one Google Drive research folder for local indexing."""

import argparse
import json
import os
import re
from pathlib import Path

from pydrive2.auth import GoogleAuth, RefreshError
from pydrive2.drive import GoogleDrive

from .config import ROOT

FOLDER_MIME = "application/vnd.google-apps.folder"


def escape_drive_query(value: str) -> str:
    """Escape a folder name before placing it in a Google Drive query."""
    return value.replace("\\", "\\\\").replace("'", "\\'")


def find_root_folders(drive: GoogleDrive, names: list[str]) -> list[str]:
    """Find each configured folder by its exact name in My Drive."""
    folder_ids = []
    for name in names:
        escaped = escape_drive_query(name)
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


def safe_name(value: str) -> str:
    """Replace characters Windows does not allow in downloaded filenames."""
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value).strip(" .")
    return cleaned or "untitled"


def authenticate(client_secrets: Path, credentials: Path) -> GoogleDrive:
    """Reuse saved Google OAuth credentials or open the first-time consent flow."""
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


def list_tree(drive: GoogleDrive, folder_id: str):
    """Walk the selected Drive folder and subfolders, excluding trashed files."""
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


def download_supported(item, output_dir: Path):
    """Download a PDF; other Drive files are outside this research phase."""
    original = safe_name(item["title"])
    extension = Path(original).suffix.lower()
    if extension != ".pdf":
        print(f"SKIP non-PDF Drive file: {item['title']}")
        return None
    target = output_dir / f"{item['id']}_{original}"
    item.GetContentFile(str(target))
    return target


def sync(
    folder_ids: list[str],
    folder_names: list[str],
    output_dir: Path,
    manifest: Path,
    client_secrets: Path,
    credentials: Path,
):
    """Refresh one research folder and write a manifest of its PDFs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    drive = authenticate(client_secrets, credentials)
    selected_folder_ids = list(folder_ids)
    if folder_names:
        selected_folder_ids.extend(find_root_folders(drive, folder_names))
    selected_folder_ids = list(dict.fromkeys(selected_folder_ids))
    if len(selected_folder_ids) != 1:
        raise ValueError("Configure exactly one Google Drive research-paper folder")

    drive_rows = []
    seen_file_ids = set()
    for folder_id in selected_folder_ids:
        for item in list_tree(drive, folder_id):
            if item["id"] in seen_file_ids:
                continue
            seen_file_ids.add(item["id"])
            target = download_supported(item, output_dir)
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
                "title": item["title"],
                "source_type": "google_drive",
                "parent_ids": [parent["id"] for parent in item.get("parents", [])],
                "source_url": item.get("alternateLink")
                or f"https://drive.google.com/open?id={item['id']}",
            }
            drive_rows.append(row)
            print(f"SYNC {item['title']} -> {local_path}")

    combined = sorted(drive_rows, key=lambda row: row["title"].casefold())
    if not combined:
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
    """Parse arguments to synchronize Drive."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder-id", action="append", dest="folder_ids")
    parser.add_argument("--folder-name", action="append", dest="folder_names")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "synced_sources" / "drive"
    )
    parser.add_argument("--manifest", type=Path, default=ROOT / "sources.local.json")
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
    sync(
        folder_ids,
        folder_names,
        args.output_dir.resolve(),
        args.manifest.resolve(),
        args.client_secrets.resolve(),
        args.credentials.resolve(),
    )


if __name__ == "__main__":
    main()
