"""Build and refresh the local research-paper index before users ask questions.

Read the Drive manifest, extract PDF sections, embed child chunks and save them in
Chroma. SQLite records a summary of the run. This command does not watch Drive
automatically; run drive_sync.py first when files change.
"""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from urllib.parse import urlparse

from .db import create_record_id, initialize_database, open_database
from .document_processing import (
    EXTRACTION_PROFILE,
    extract_pdf_sections,
    split_sections_into_chunks,
)
from .llm import embed_document_chunks
from .neural_pdf_extraction import NeuralPageReader
from .paper_store import (
    build_index_signature,
    open_chunk_store,
    open_parent_section_store,
    reset_paper_stores,
)

EMBED_BATCH_SIZE = 16


def save_index_run_summary(
    run_id: str,
    started_at: str,
    manifest_hash: str,
    rebuild: bool,
    source_count: int,
    status: str,
    counts: dict,
    error: str = "",
) -> None:
    """Record what happened during one indexing run in SQLite.

    Store input/settings identifiers, timestamps, counts and any error so evaluation
    results can be traced to an index run. This is a summary, not a backup of the
    vectors or papers.
    """

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
                build_index_signature(),
                int(rebuild),
                source_count,
                status,
                json.dumps(counts),
                error,
            ),
        )


def load_source_manifest(path: Path) -> list[dict]:
    """Read the downloaded-paper list and validate it before changing the index.

    Resolve local paths relative to the manifest, require unique Drive IDs and
    source links, and keep PDF entries. Reject an empty or invalid source list
    rather than accidentally clearing indexed papers.
    """

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
        if Path(row["path"]).suffix.lower() != ".pdf":
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


def index_research_paper(store, row: dict, reader=None) -> tuple[str, int]:
    """Extract, chunk, embed and save one paper, or skip it if unchanged.

    Compare its file/settings fingerprint with existing records first. Finish
    extraction and embedding before replacing records so those failures leave the
    previous paper searchable. Save child vectors and section text together with
    source and page metadata.
    """

    started = perf_counter()
    source_id = row["id"]
    existing = store.get(where={"document_id": source_id}, include=["metadatas"])
    # Calculate a fingerprint from PDF bytes, source details and extraction settings.
    # Matching fingerprints let a resumed run skip completed work for this paper.
    identity = f"{row['title']}|{row['source_url']}|{build_index_signature()}|{EXTRACTION_PROFILE}".encode()
    digest = hashlib.sha256(row["_path"].read_bytes() + identity).hexdigest()
    parents = open_parent_section_store()
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
        print(f"UNCHANGED {row['title']}", flush=True)
        return "unchanged", 0

    print(f"EXTRACT {row['title']}: starting", flush=True)
    extracting = perf_counter()
    document_type, sections = extract_pdf_sections(row["_path"], reader=reader)
    chunks = split_sections_into_chunks(sections, row["title"])
    if not chunks:
        # Keep the old index intact if extraction fails (for example a scanned PDF).
        print(f"SKIP {row['title']}: no extractable text; OCR is not configured")
        return "skipped", 0
    print(
        f"EXTRACT {row['title']}: {perf_counter() - extracting:.2f}s, {len(sections)} sections, {len(chunks)} chunks",
        flush=True,
    )
    embedding = perf_counter()
    # Finish embedding before writing so a model failure leaves the old paper intact.
    vectors = []
    for offset in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[offset : offset + EMBED_BATCH_SIZE]
        print(
            f"EMBED {row['title']}: batch {offset // EMBED_BATCH_SIZE + 1}/{(len(chunks) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE}",
            flush=True,
        )
        vectors.extend(embed_document_chunks([chunk.page_content for chunk in batch]))
    print(f"EMBED {row['title']}: {perf_counter() - embedding:.2f}s", flush=True)
    writing = perf_counter()
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
                "section_path": chunk.metadata["section_path"],
                "heading_level": section.heading_level,
                "evidence_type": chunk.metadata["evidence_type"],
                "reference_label": section.reference_label,
                "extraction_profile": EXTRACTION_PROFILE,
                "extraction_method": section.extraction_method,
                "structure_confidence": section.structure_confidence,
                "review_reason": section.review_reason,
                "document_type": document_type,
                "fingerprint": digest,
                "indexed_at": indexed_at,
                "chunk_count": len(chunks),
                "parent_id": parent_id,
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
                "section_path": " > ".join(sections[index].heading_path),
                "heading_level": sections[index].heading_level,
                "reference_label": sections[index].reference_label,
                "evidence_type": "caption"
                if sections[index].strategy == "pdf-caption"
                else "prose",
                "extraction_profile": EXTRACTION_PROFILE,
                "extraction_method": sections[index].extraction_method,
                "structure_confidence": sections[index].structure_confidence,
                "review_reason": sections[index].review_reason,
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
    print(f"STORE {row['title']}: {perf_counter() - writing:.2f}s", flush=True)
    print(
        f"INDEX {row['title']}: {len(chunks)} chunks ({document_type}), total {perf_counter() - started:.2f}s",
        flush=True,
    )
    return "indexed", len(chunks)


def remove_unlisted_papers(store, current_ids: set[str]) -> int:
    """Remove indexed records for papers no longer present in the source manifest.

    Delete both child chunks and stored sections so removed papers cannot appear in
    search or context expansion.
    """

    removed_chunks = 0
    for index in (store, open_parent_section_store()):
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


def ingest_research_papers(manifest: Path, rebuild: bool = False) -> dict:
    """Coordinate a complete indexing run and return its result counts.

    Validate sources, optionally clear collections for an explicit rebuild, and
    reuse one neural worker across papers. Update the last-ingested time on success
    and record completed or failed runs. Always close the worker when the command
    exits.
    """

    papers = load_source_manifest(manifest)
    initialize_database()
    counts = {"indexed": 0, "unchanged": 0, "skipped": 0, "chunks": 0}
    run_id, started_at = create_record_id(), datetime.now(UTC).isoformat()
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    reader = NeuralPageReader()
    try:
        if rebuild:
            reset_paper_stores()
        store = open_chunk_store()
        for row in papers:
            outcome, chunk_count = index_research_paper(store, row, reader=reader)
            counts[outcome] += 1
            counts["chunks"] += chunk_count
        counts["removed"] = remove_unlisted_papers(store, {row["id"] for row in papers})
        with open_database() as database:
            database.execute(
                "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
                ("last_ingested_at", datetime.now(UTC).isoformat()),
            )
        save_index_run_summary(
            run_id, started_at, manifest_hash, rebuild, len(papers), "completed", counts
        )
    except Exception as exc:
        save_index_run_summary(
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
    finally:
        reader.close()
    print(json.dumps(counts))
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--rebuild", action="store_true")
    arguments = parser.parse_args()
    ingest_research_papers(arguments.manifest.resolve(), arguments.rebuild)
