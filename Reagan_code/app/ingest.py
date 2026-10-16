"""Embed changed Google Drive files and store their searchable chunks.

Document parsing and chunking live in document_processing.py so this file can
focus on one workflow: validate the manifest, embed changed content, and update
PostgreSQL atomically. If any changed document fails, the previous working index
remains available.
"""
import argparse
import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from sqlalchemy import delete, select, text

from .config import EMBED_DIM, EMBED_MODEL
from .db import ActionItem, Chunk, Document, Session, Setting
from .document_processing import chunk_sections, extract_sections
from .llm import embed_texts

PIPELINE_VERSION = "2:adaptive-structure:1800chars:150overlap:pymupdf"


def parse_actions(markdown):
    """Exact Markdown table, no LLM extraction or inferred owner/status."""
    import re
    active = False
    actions = []
    for number, line in enumerate(markdown.splitlines(), 1):
        if not line.strip().startswith("|"):
            active = False
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if [c.casefold() for c in cells] == ["action", "owner", "status", "due"]:
            active = True
            continue
        if not active or all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
            continue
        if len(cells) != 4 or cells[2].lower() not in {"open", "in_progress", "done"}:
            raise ValueError(f"Invalid action table at line {number}")
        description, owner, status, due = cells
        if not description or not owner:
            raise ValueError(f"Missing action description/owner at line {number}")
        actions.append(dict(description=description, owner=owner, status=status.lower(),
                            due_date=date.fromisoformat(due) if due else None, location=f"line {number}"))
    return actions


def load_manifest(manifest):
    """Validate a Google Drive manifest and resolve each downloaded file path."""
    rows = json.loads(manifest.read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manifest must be a non-empty JSON list")
    ids = set()
    for row in rows:
        for key in ("id", "path", "title", "source_type", "source_url"):
            if not row.get(key):
                raise ValueError(f"Manifest record needs {key}")
        if row["id"] in ids:
            raise ValueError("Duplicate source ID")
        ids.add(row["id"])
        if row["source_type"] != "google_drive":
            raise ValueError("source_type must be google_drive")
        if urlparse(row["source_url"]).scheme != "https":
            raise ValueError("source_url must be an HTTPS source link")
        path = Path(row["path"])
        row["_path"] = path if path.is_absolute() else manifest.parent / path
        if not row["_path"].is_file():
            raise ValueError(f"Source is missing or not locally available: {row['path']}")
    return rows


def run(manifest, rebuild=False):
    """Replace changed documents atomically; unchanged files avoid another embedding call."""
    rows = load_manifest(manifest)
    signature = f"{EMBED_MODEL}:{EMBED_DIM}:{PIPELINE_VERSION}"
    counts = {"indexed": 0, "unchanged": 0, "skipped": 0, "chunks": 0}
    with Session.begin() as db:
        # One ingestion writer. Full transaction rolls back if any source fails.
        db.execute(text("SELECT pg_advisory_xact_lock(713241)"))
        setting = db.get(Setting, "embedding_signature")
        if setting and setting.value != signature and not rebuild:
            raise ValueError(
                "Chunking or embedding settings changed. Rerun with --rebuild to replace old chunks."
            )
        if setting and rebuild:
            setting.value = signature
        if not setting:
            db.add(Setting(key="embedding_signature", value=signature))
        incoming = {row["id"] for row in rows}
        for row in rows:
            path = row["_path"]
            data = path.read_bytes()
            metadata = {k: v for k, v in row.items() if k != "_path"}
            fingerprint = hashlib.sha256(
                data + json.dumps(metadata, sort_keys=True).encode() + signature.encode()
            ).hexdigest()
            existing = db.get(Document, row["id"])
            if existing and existing.fingerprint == fingerprint:
                existing.active = True
                counts["unchanged"] += 1
                continue
            sections = extract_sections(path)
            missing = [section.location for section in sections if not section.text]
            if missing and path.suffix.lower() in {".pdf", ".pptx"}:
                print(f"WARNING {row['id']}: no extractable text at {', '.join(missing)}; images are not OCR'd")
            chunks = chunk_sections(sections)
            if not chunks:
                print(f"SKIP no extractable text: {path.name}")
                if existing:
                    existing.active = False
                counts["skipped"] += 1
                continue
            vectors = []
            for offset in range(0, len(chunks), 16):
                vectors.extend(embed_texts([c.page_content for c in chunks[offset:offset+16]]))
            actions = (
                parse_actions(path.read_text(encoding="utf-8", errors="replace"))
                if path.suffix.lower() == ".md"
                else []
            )
            if existing:
                db.execute(delete(Chunk).where(Chunk.document_id == row["id"]))
                db.execute(delete(ActionItem).where(ActionItem.document_id == row["id"]))
            else:
                existing = Document(id=row["id"])
                db.add(existing)
            for field in ("title", "source_type", "source_url"):
                setattr(existing, field, row[field])
            existing.author = row.get("author", "")
            existing.topic = row.get("topic", "")
            existing.session_date = date.fromisoformat(row["session_date"]) if row.get("session_date") else None
            existing.fingerprint, existing.active = fingerprint, True
            existing.indexed_at = datetime.now(timezone.utc)
            db.flush()
            for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
                db.add(Chunk(id=f"{row['id']}:{fingerprint[:12]}:c{index}", document_id=row["id"],
                             content=chunk.page_content, location=chunk.metadata["location"], embedding=vector))
            for index, action in enumerate(actions):
                db.add(ActionItem(id=f"{row['id']}:{fingerprint[:12]}:a{index}", document_id=row["id"], **action))
            counts["indexed"] += 1
            counts["chunks"] += len(chunks)
            strategies = sorted({chunk.metadata["strategy"] for chunk in chunks})
            print(f"INDEX {row['title']}: {len(chunks)} chunks using {', '.join(strategies)}")
        # Omitted sources are hidden from future searches, preserved for recovery.
        for doc in db.scalars(select(Document)):
            if doc.id not in incoming:
                doc.active = False
        synced = db.get(Setting, "last_ingested_at")
        if synced is None:
            synced = Setting(key="last_ingested_at", value="")
            db.add(synced)
        synced.value = datetime.now(timezone.utc).isoformat()
    print(json.dumps(counts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Replace existing chunks after changing the chunking or embedding configuration.",
    )
    arguments = parser.parse_args()
    run(arguments.manifest.resolve(), rebuild=arguments.rebuild)
