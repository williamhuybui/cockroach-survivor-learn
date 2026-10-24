"""Index only the research PDFs in a Google Drive sync manifest."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from .db import initialize_database, new_id, open_database
from .document_processing import chunk_sections, classify_format, extract_sections
from .llm import embed_texts
from .paper_store import (
    collection,
    embedding_signature,
    parent_collection,
    reset_collection,
)

EMBED_BATCH_SIZE = 16


def file_sha256(path: Path) -> str:
    """Return the content hash that identifies one exact source manifest."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_index_run(
    run_id: str,
    started_at: str,
    manifest_hash: str,
    rebuild: bool,
    source_count: int,
    status: str,
    counts: dict,
    error: str = "",
) -> None:
    """Save a small, queryable record of a completed or failed index attempt."""

    with open_database() as database:
        database.execute(
            """
            INSERT INTO index_runs (
                id, started_at, completed_at, manifest_sha256, index_signature,
                rebuild, source_count, status, counts, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                started_at,
                datetime.now(UTC).isoformat(),
                manifest_hash,
                embedding_signature(),
                int(rebuild),
                source_count,
                status,
                json.dumps(counts),
                error,
            ),
        )


def load_manifest(path: Path) -> list[dict]:
    """Validate the downloaded Drive files and ignore non-PDF items."""

    rows = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list):
        raise TypeError("The Drive manifest must contain a JSON list")
    papers = []
    seen_ids = set()
    for row in rows:
        if not isinstance(row, dict) or not all(
            row.get(key) for key in ("id", "path", "title", "source_url")
        ):
            raise ValueError("Each manifest entry needs id, path, title and source_url")
        if row["id"] in seen_ids:
            raise ValueError(f"Duplicate Drive source ID: {row['id']}")
        seen_ids.add(row["id"])
        if classify_format(Path(row["path"])) != "pdf":
            continue
        if (
            row.get("source_type") != "google_drive"
            or urlparse(row["source_url"]).scheme != "https"
        ):
            raise ValueError("Research PDFs need a Google Drive HTTPS source link")
        local_path = Path(row["path"])
        local_path = (
            local_path if local_path.is_absolute() else path.parent / local_path
        )
        if not local_path.is_file():
            raise ValueError(f"PDF is missing: {local_path}")
        papers.append({**row, "_path": local_path})
    if not papers:
        raise ValueError("No research PDFs were found in the selected Drive folder")
    return papers


def fingerprint(row: dict) -> str:
    """Detect changed PDF bytes or source metadata before re-embedding."""

    identity = f"{row['title']}|{row['source_url']}|{embedding_signature()}".encode()
    return hashlib.sha256(row["_path"].read_bytes() + identity).hexdigest()


def index_paper(store, row: dict) -> tuple[str, int]:
    """Extract, chunk, embed, and replace one PDF in Chroma."""

    source_id = row["id"]
    existing = store.get(where={"document_id": source_id}, include=["metadatas"])
    digest = fingerprint(row)
    parents = parent_collection()
    existing_parents = parents.get(where={"document_id": source_id})
    if (
        existing["ids"]
        and all(
            meta["fingerprint"] == digest
            and meta.get("chunk_count") == len(existing["ids"])
            for meta in existing["metadatas"]
        )
        and existing_parents["ids"]
    ):
        return "unchanged", 0

    document_type, sections = extract_sections(row["_path"])
    chunks = chunk_sections(sections, row["title"])
    if not chunks:
        # Keep the old index intact if extraction fails (for example a scanned PDF).
        print(f"SKIP {row['title']}: no extractable text; OCR is not configured")
        return "skipped", 0
    # Finish embedding before writing so a model failure leaves the old paper intact.
    vectors = []
    for offset in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[offset : offset + EMBED_BATCH_SIZE]
        vectors.extend(embed_texts([chunk.page_content for chunk in batch]))
    indexed_at = datetime.now(UTC).isoformat()
    new_ids = [f"{source_id}:{digest[:12]}:{index}" for index in range(len(chunks))]
    parent_ids = {}
    section_vectors = {}
    metadata = []
    for chunk, vector in zip(chunks, vectors):
        section_index = chunk.metadata["section_index"]
        section = sections[section_index]
        parent_id = f"{source_id}:{digest[:12]}:section:{section_index}"
        parent_ids[section_index] = parent_id
        section_vectors.setdefault(section_index, []).append(vector)
        # Extraction stores the first and last page numbers in its location label.
        page_parts = section.location.split()
        metadata.append(
            {
                "document_id": source_id,
                "title": row["title"],
                "source_url": row["source_url"],
                "location": chunk.metadata["location"],
                "heading": chunk.metadata["heading"],
                "document_type": document_type,
                "fingerprint": digest,
                "indexed_at": indexed_at,
                "chunk_count": len(chunks),
                "parent_id": parent_id,
                "child_index": chunk.metadata["child_index"],
                "page_start": int(page_parts[1]),
                "page_end": int(page_parts[-1]),
            }
        )
    parent_vectors = []
    for members in section_vectors.values():
        # A parent centroid is the mean of its children's vectors.
        parent_vectors.append([sum(values) / len(members) for values in zip(*members)])
    parents.upsert(
        ids=list(parent_ids.values()),
        documents=[sections[index].text for index in parent_ids],
        embeddings=parent_vectors,
        metadatas=[
            {
                "document_id": source_id,
                "heading": sections[index].heading,
                "location": sections[index].location,
                "fingerprint": digest,
            }
            for index in parent_ids
        ],
    )
    store.upsert(
        ids=new_ids,
        documents=[chunk.page_content for chunk in chunks],
        embeddings=vectors,
        metadatas=metadata,
    )
    old_ids = [chunk_id for chunk_id in existing["ids"] if chunk_id not in new_ids]
    if old_ids:
        store.delete(ids=old_ids)
    stale_parents = [
        key for key in existing_parents["ids"] if key not in parent_ids.values()
    ]
    if stale_parents:
        parents.delete(ids=stale_parents)
    print(f"INDEX {row['title']}: {len(chunks)} chunks ({document_type})")
    return "indexed", len(chunks)


def remove_missing_papers(store, current_ids: set[str]) -> int:
    """Remove child and parent records absent from the validated source manifest."""

    removed_chunks = 0
    for index in (store, parent_collection()):
        saved = index.get(include=["metadatas"])
        stale_ids = [
            key
            for key, meta in zip(saved["ids"], saved["metadatas"])
            if meta["document_id"] not in current_ids
        ]
        if stale_ids:
            index.delete(ids=stale_ids)
        if index is store:
            removed_chunks = len(stale_ids)
    return removed_chunks


def run(manifest: Path, rebuild: bool = False) -> dict:
    """Sync the current PDF set; rebuild only when the user explicitly asks."""

    papers = load_manifest(manifest)
    initialize_database()
    counts = {"indexed": 0, "unchanged": 0, "skipped": 0, "chunks": 0}
    run_id, started_at = new_id(), datetime.now(UTC).isoformat()
    manifest_hash = file_sha256(manifest)
    try:
        if rebuild:
            reset_collection()
        store = collection()
        for row in papers:
            outcome, chunk_count = index_paper(store, row)
            counts[outcome] += 1
            counts["chunks"] += chunk_count
        counts["removed"] = remove_missing_papers(store, {row["id"] for row in papers})
        with open_database() as database:
            database.execute(
                "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
                ("last_ingested_at", datetime.now(UTC).isoformat()),
            )
        save_index_run(
            run_id, started_at, manifest_hash, rebuild, len(papers), "completed", counts
        )
    except Exception as exc:
        save_index_run(
            run_id,
            started_at,
            manifest_hash,
            rebuild,
            len(papers),
            "failed",
            counts,
            f"{type(exc).__name__}: {exc}",
        )
        raise
    print(json.dumps(counts))
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--rebuild", action="store_true")
    arguments = parser.parse_args()
    run(arguments.manifest.resolve(), arguments.rebuild)
