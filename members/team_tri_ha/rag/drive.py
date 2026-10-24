import io
import json
import re
from pathlib import Path

from rag import config
from rag.loaders import BINARY_EXTS, MAX_SNIFF_BYTES, SUPPORTED, SkipFile, load_any

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
FOLDER_TYPE = "application/vnd.google-apps.folder"

EXPORTS = {
    "application/vnd.google-apps.document": (
        ".docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    "application/vnd.google-apps.spreadsheet": (
        ".xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    "application/vnd.google-apps.presentation": (
        ".pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation"),
}


MIME_EXT = {
    "application/pdf": ".pdf",
    "text/plain": ".txt",
    "text/markdown": ".md",
    "text/csv": ".csv",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
}


BINARY_MIMES = {"application/zip", "application/x-zip-compressed", "application/gzip", "application/x-tar",
                "application/x-sqlite3", "application/vnd.sqlite3", "application/x-7z-compressed"}


def folder_id_from(url_or_id: str) -> str:
    """Accept a full folder link or a bare folder id."""
    m = re.search(r"/folders/([A-Za-z0-9_-]+)", url_or_id)
    return m.group(1) if m else url_or_id.strip()


def service_account_email() -> str:
    try:
        return json.loads(Path(config.GOOGLE_CREDENTIALS).read_text())["client_email"]
    except Exception:
        return "(unknown)"


def get_service():
    key = Path(config.GOOGLE_CREDENTIALS)
    if not key.exists():
        raise SystemExit(
            f"Service-account key not found: {key}\n"
            "Create one (README: 'Reading Drive with the API') and save it there,\n"
            "or set GOOGLE_CREDENTIALS in .env to its path."
        )
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    creds = service_account.Credentials.from_service_account_file(str(key), scopes=SCOPES)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _wanted(path: str, only: str) -> bool:
    """With only='Research Paper/': keep that folder, the folders leading to it, and what is inside it."""
    return not only or path.startswith(only) or only.startswith(path)


def list_files(service, folder_id: str, prefix: str = "", only: str = ""):
    """Yield one dict per file in the folder and all its sub-folders: id, name, mimeType, path.
    Hidden files and folders (.git, .DS_Store...) are skipped; `only` limits which sub-folder is read."""
    page_token = None
    while True:  
        resp = service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            fields="nextPageToken, files(id, name, mimeType, size)",
            pageSize=100,
            pageToken=page_token,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        for f in resp.get("files", []):
            if f["name"].startswith("."):
                continue
            if f["mimeType"] == FOLDER_TYPE:
                if _wanted(prefix + f["name"] + "/", only):
                    yield from list_files(service, f["id"], prefix + f["name"] + "/", only)
            elif _wanted(prefix + f["name"], only) and (prefix + f["name"]).startswith(only):
                f["path"] = prefix + f["name"]
                yield f
        page_token = resp.get("nextPageToken")
        if not page_token:
            break


def download_bytes(service, f: dict) -> bytes:
    """Fetch one file's content into memory (export it first if it is a Google-native file)."""
    from googleapiclient.http import MediaIoBaseDownload

    if f["mimeType"] in EXPORTS:
        request = service.files().export_media(fileId=f["id"], mimeType=EXPORTS[f["mimeType"]][1])
    else:
        request = service.files().get_media(fileId=f["id"], supportsAllDrives=True)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buf.getvalue()


def load_drive_folder(url_or_id: str, service=None, only: str = ""):
    """Same return shape as loaders.load_folder: (documents, report).
    only="Research Paper" reads just that sub-folder of the Drive folder."""
    service = service or get_service()
    folder_id = folder_id_from(url_or_id)
    only = (only.strip().strip("/") + "/") if only.strip() else ""
    print(f"Reading Drive folder {folder_id}" + (f" (only '{only}')" if only else "")
          + f" as {service_account_email()}")

    try:
        files = list(list_files(service, folder_id, only=only))
    except Exception as e:
        raise SystemExit(
            f"Could not list the Drive folder: {type(e).__name__}: {e}\n"
            "Check that the Drive API is enabled for your Google Cloud project."
        )
    if not files:
        raise SystemExit(
            "No files found.\n"
            f"- Is the folder shared (Viewer) with: {service_account_email()} ?\n"
            + (f"- Does a sub-folder named '{only.rstrip('/')}' exist (check spelling and capitals)?" if only else "")
        )

    def display_name(f):
        name = f["path"] + (EXPORTS[f["mimeType"]][0] if f["mimeType"] in EXPORTS else "")
        if Path(name).suffix.lower() not in SUPPORTED and f["mimeType"] in MIME_EXT:
            name += MIME_EXT[f["mimeType"]]  # no usable extension in the name: trust the Drive file type
        return name

    def skip_reason(f):
        """Why a file is not even downloaded (decided from Drive's metadata); None = try to read it."""
        mime = f["mimeType"]
        if mime.startswith("application/vnd.google-apps.") and mime not in EXPORTS:
            return "unsupported Google file type"
        if Path(display_name(f)).suffix.lower() in SUPPORTED:
            return None
        if mime.startswith(("image/", "audio/", "video/")) or mime in BINARY_MIMES \
                or Path(display_name(f)).suffix.lower() in BINARY_EXTS:
            return "binary file"
        if int(f.get("size") or 0) > MAX_SNIFF_BYTES:
            return "too large to treat as text"
        return None  

    def readable(f):
        return skip_reason(f) is None

    files = sorted(files, key=lambda x: x["path"])
    total = sum(1 for f in files if readable(f))
    print(f"Found {len(files)} files, {total} readable. Downloading and reading (big PDFs take a while)...")

    docs, report, done = [], [], 0
    for f in files:
        name = display_name(f)
        path = Path(name)  # only used for the name and extension, never opened
        row = {"file": name, "status": "ok", "parts": 0, "chars": 0}
        report.append(row)

        reason = skip_reason(f)
        if reason:
            row["status"] = f"SKIPPED ({reason})"
            continue
        done += 1
        print(f"[{done}/{total}] {name}", flush=True)
        try:
            file_docs = load_any(path, download_bytes(service, f))
            if not file_docs:
                row["status"] = "EMPTY (scanned PDF? needs OCR)"
        except SkipFile as e:  # unknown type that turned out to be binary
            file_docs, row["status"] = [], f"SKIPPED ({e})"
        except Exception as e:  # one broken file must not stop the whole run
            file_docs, row["status"] = [], f"ERROR: {type(e).__name__}: {e}"
        docs.extend(file_docs)
        row["parts"] = len(file_docs)
        row["chars"] = sum(len(d.page_content) for d in file_docs)
    return docs, report
