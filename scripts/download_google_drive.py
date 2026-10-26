"""Download every file from Google Drive into the repo's tmp/ folder."""
from pathlib import Path
from pydrive2.auth import GoogleAuth, RefreshError
from pydrive2.drive import GoogleDrive

repo = Path(__file__).resolve().parents[1]
out_dir = repo / "tmp"

# Google-native files can't be downloaded as-is, so export them to a normal format
EXPORTS = {
    "application/vnd.google-apps.document": (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document", ".docx"),
    "application/vnd.google-apps.spreadsheet": (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", ".xlsx"),
    "application/vnd.google-apps.presentation": (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation", ".pptx"),
}

# Login (browser opens the first time, then credentials.json is reused)
gauth = GoogleAuth()
gauth.settings["client_config_file"] = str(repo / "client_secrets.json")
gauth.settings["get_refresh_token"] = True
creds_file = str(repo / "credentials.json")
gauth.LoadCredentialsFile(creds_file)
if gauth.credentials is None:
    gauth.LocalWebserverAuth()
elif gauth.access_token_expired:
    try:
        gauth.Refresh()
    except RefreshError:  # saved login expired or revoked -> log in again
        gauth.credentials = None
        gauth.LocalWebserverAuth()
else:
    gauth.Authorize()
gauth.SaveCredentialsFile(creds_file)
drive = GoogleDrive(gauth)


def download_folder(folder_id, folder_path):
    folder_path.mkdir(parents=True, exist_ok=True)
    files = drive.ListFile({"q": f"'{folder_id}' in parents and trashed=false"}).GetList()
    for f in files:
        name = f["title"].replace("/", "_")
        if f["mimeType"] == "application/vnd.google-apps.folder":
            download_folder(f["id"], folder_path / name)
        elif f["mimeType"] in EXPORTS:
            mimetype, ext = EXPORTS[f["mimeType"]]
            path = folder_path / (name + ext)
            if path.exists():
                print("Already exists, skipping", path)
                continue
            print("Downloading", path)
            f.GetContentFile(str(path), mimetype=mimetype)
        elif f["mimeType"].startswith("application/vnd.google-apps."):
            print("Skipping (can't export):", name)
        else:
            path = folder_path / name
            if path.exists():
                print("Already exists, skipping", path)
                continue
            print("Downloading", path)
            f.GetContentFile(str(path))


download_folder("root", out_dir)
print("Done. Files saved in", out_dir)
