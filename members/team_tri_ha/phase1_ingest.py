import argparse
import shutil
from pathlib import Path

from rag import config
from rag.loaders import load_folder, save_docs


def download_from_drive(url: str, out_dir: Path) -> None:
    import gdown

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Drive folder -> {out_dir}")
    try:
        gdown.download_folder(url=url, output=str(out_dir), quiet=False)
    except Exception as e:
        raise SystemExit(
            f"Drive download failed: {type(e).__name__}: {e}\n"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true", help="do not download, use data/raw as is")
    ap.add_argument("--local-dir", help="copy files from this folder instead of using Drive")
    ap.add_argument("--drive-api", action="store_true",
                    help="read the Drive folder with the Drive API (service account), no files saved")
    ap.add_argument("--only", default=config.DRIVE_ONLY,
                    help='with --drive-api: read only this sub-folder, e.g. --only "Research Paper"')
    args = ap.parse_args()

    if args.drive_api:
        from rag.drive import load_drive_folder

        docs, report = load_drive_folder(config.DRIVE_FOLDER_URL, only=args.only)
    else:
        if args.local_dir:
            config.RAW_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copytree(args.local_dir, config.RAW_DIR, dirs_exist_ok=True)
        elif not args.skip_download:
            download_from_drive(config.DRIVE_FOLDER_URL, config.RAW_DIR)

        if not config.RAW_DIR.exists() or not any(config.RAW_DIR.iterdir()):
            raise SystemExit(f"{config.RAW_DIR} is empty. Run without --skip-download, or use --local-dir.")

        docs, report = load_folder(config.RAW_DIR)

    def short(name, width=60):  
        return name if len(name) <= width else "…" + name[-(width - 1):]

    print(f"\n{'FILE':60} {'PARTS':>6} {'CHARS':>10}  STATUS")
    print("-" * 100)
    for r in report:
        print(f"{short(r['file']):60} {r['parts']:>6} {r['chars']:>10}  {r['status']}")

    save_docs(docs, config.DOCS_FILE)
    ok_files = sum(1 for r in report if r["status"] == "ok")
    print(f"\nFiles read OK: {ok_files}/{len(report)}   Documents (pages/sheets/slides): {len(docs)}")
    print(f"Saved -> {config.DOCS_FILE}")

    if docs:
        d = docs[0]
        print(f"\nPreview of first document  [{d.metadata}]")
        print(d.page_content[:400].replace("\n", " ") + " ...")


if __name__ == "__main__":
    main()
