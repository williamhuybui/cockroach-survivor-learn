
import os
import tempfile
from pathlib import Path

import fitz
from dotenv import load_dotenv
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive


# --------------------------------------------------
# Project paths
# --------------------------------------------------

# drive.py is in learner-rag/app/
# BASE_DIR points to learner-rag/
BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = BASE_DIR / "app"
SECRETS_DIR = BASE_DIR / "secrets"

CLIENT_SECRETS_FILE = SECRETS_DIR / "client_secrets.json"
CREDENTIALS_FILE = SECRETS_DIR / "credentials.json"
ENV_FILE = BASE_DIR / ".env"
DATA_DIR = BASE_DIR / "data"

load_dotenv(ENV_FILE)
DATA_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# PDF configuration
# --------------------------------------------------

PDF_MIME = "application/pdf"
SUPPORTED_MIME_TYPES = {PDF_MIME}

TARGET_DOCUMENTS = [
    "attention is all you need",
    "harnessing the universal geometry of embeddings",
]


# --------------------------------------------------
# Google Drive authentication
# --------------------------------------------------

def get_drive_service():
    """Authenticate with Google Drive using PyDrive2."""

    if not CLIENT_SECRETS_FILE.exists():
        raise FileNotFoundError(
            f"OAuth client secrets file not found: "
            f"{CLIENT_SECRETS_FILE}"
        )

    SECRETS_DIR.mkdir(parents=True, exist_ok=True)

    gauth = GoogleAuth()
    gauth.settings["client_config_file"] = str(
        CLIENT_SECRETS_FILE
    )

    if CREDENTIALS_FILE.exists():
        gauth.LoadCredentialsFile(str(CREDENTIALS_FILE))

    if gauth.credentials is None:
        gauth.LocalWebserverAuth()

    elif gauth.access_token_expired:
        if getattr(gauth.credentials, "refresh_token", None):
            gauth.Refresh()
        else:
            gauth.LocalWebserverAuth()

    else:
        gauth.Authorize()

    gauth.SaveCredentialsFile(str(CREDENTIALS_FILE))

    return GoogleDrive(gauth)


# --------------------------------------------------
# Find target PDF files
# --------------------------------------------------

def normalize_title(title):
    """Normalize a filename for matching."""
    title = Path(title).stem
    title = title.replace("_", " ").replace("-", " ")
    return " ".join(title.lower().strip().split())


def list_drive_files(drive, folder_id=None):
    """
    Find the target PDF papers in Google Drive.

    Searches all accessible files unless DRIVE_FOLDER_ID
    is configured. Folder search is not recursive.
    """

    query = "trashed=false"

    if folder_id:
        # Escape single quotes in a Drive folder ID, if present.
        safe_folder_id = folder_id.replace("\\", "\\\\").replace(
            "'", "\\'"
        )
        query += f" and '{safe_folder_id}' in parents"

    file_list = drive.ListFile(
        {
            "q": query,
            "maxResults": 1000,
        }
    ).GetList()

    results = []

    for item in file_list:
        title = item.get("title", item.get("name", ""))
        mime_type = item.get("mimeType", "")

        # Only process PDFs.
        if mime_type not in SUPPORTED_MIME_TYPES:
            continue

        normalized_title = normalize_title(title)

        # Match either target paper.
        if not any(
            target in normalized_title
            for target in TARGET_DOCUMENTS
        ):
            continue

        results.append(
            {
                "id": item["id"],
                "name": title,
                "mimeType": mime_type,
                "webViewLink": item.get(
                    "alternateLink",
                    item.get("webViewLink", ""),
                ),
            }
        )

    return results


# --------------------------------------------------
# Download PDF to a local temporary file
# --------------------------------------------------

def download_file(drive, file):
    """
    Download a PDF from Google Drive.

    Returns the local path as a pathlib.Path object.
    The caller is responsible for deleting the temporary file.
    """

    if file["mimeType"] != PDF_MIME:
        raise ValueError(
            f"Only PDF files are supported: {file['name']}"
        )

    suffix = Path(file["name"]).suffix or ".pdf"

    temp_file = tempfile.NamedTemporaryFile(
        suffix=suffix,
        delete=False,
    )
    temp_path = Path(temp_file.name)
    temp_file.close()

    try:
        drive_file = drive.CreateFile({"id": file["id"]})
        drive_file.GetContentFile(str(temp_path))
        return temp_path

    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


# --------------------------------------------------
# Extract text from PDF using PyMuPDF
# --------------------------------------------------

def extract_text(file, local_path):
    """
    Extract text from a PDF using PyMuPDF.

    Returns a list of dictionaries, one per non-empty page.
    Each page retains its original page number.
    """

    if file["mimeType"] != PDF_MIME:
        raise ValueError(
            f"Only PDF files are supported: {file['name']}"
        )

    pages = []

    with fitz.open(str(local_path)) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            page_text = page.get_text("text").strip()

            if page_text:
                pages.append(
                    {
                        "page": page_number,
                        "text": page_text,
                    }
                )

    return pages


# --------------------------------------------------
# Run this module directly
# --------------------------------------------------

if __name__ == "__main__":
    print(f"Project root: {BASE_DIR}")
    print(f"OAuth client file: {CLIENT_SECRETS_FILE}")
    print("Connecting to Google Drive...")

    drive_service = get_drive_service()

    folder_id = os.getenv("DRIVE_FOLDER_ID") or None

    files = list_drive_files(
        drive_service,
        folder_id=folder_id,
    )

    if not files:
        print("No matching PDF papers found.")
    else:
        print(f"\nFound {len(files)} matching PDF(s).")

        for file in files:
            print(f"\nTitle: {file['name']}")
            print(f"ID: {file['id']}")
            print(f"Link: {file['webViewLink']}")

            local_path = None

            try:
                local_path = download_file(drive_service, file)
                pages = extract_text(file, local_path)

                total_chars = sum(
                    len(page["text"]) for page in pages
                )

                print(f"Pages with text: {len(pages)}")
                print(f"Extracted characters: {total_chars}")

                # Preview the first page of extracted text.
                if pages:
                    print("\nFirst page preview:")
                    print(pages[0]["text"][:500])

            except Exception as exc:
                print(f"Error processing PDF: {exc}")

            finally:
                if local_path is not None:
                    local_path.unlink(missing_ok=True)