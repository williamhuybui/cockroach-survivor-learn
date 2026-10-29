"""Open the two local Chroma collections used by indexing and retrieval.

The child collection supports vector search; the section collection supplies
surrounding text by ID. SQLite application history is managed separately in db.py.
No Chroma server is required for these persistent local collections.
"""

import chromadb

from .config import (
    CHROMA_PATH,
    CHUNK_OVERLAP_CHARACTERS,
    EMBED_DIM,
    EMBED_MODEL,
    MAX_CHUNK_CHARACTERS,
)
from .study_materials import load_paper_catalog

COLLECTION_NAME = "research_papers"
INDEX_VERSION = "paper-parents-v7-text-hierarchy-1.28.2"


def open_parent_section_store():
    """Open the collection containing complete section text for context expansion.

    A child's parent_id identifies its containing section here. Ingestion stores an
    average of child vectors for each section, but retrieval fetches this collection
    by ID rather than searching those vectors.
    """
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_or_create_collection(
        name="research_sections", embedding_function=None
    )


def build_index_signature() -> str:
    """Describe the vector model, storage version and chunk settings used by this index.

    Compare this signature when opening Chroma so incompatible stored vectors are
    not silently queried with different settings.
    """

    return f"{EMBED_MODEL}:{EMBED_DIM}:{INDEX_VERSION}:{MAX_CHUNK_CHARACTERS}:{CHUNK_OVERLAP_CHARACTERS}"


def open_chunk_store():
    """Open the persistent cosine-search collection and check its settings signature.

    EmbeddingGemma supplies vectors during ingestion and querying, so Chroma's
    automatic embedding function is disabled. Reject incompatible settings and
    require a deliberate rebuild.
    """

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    store = client().get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
        metadata={"hnsw:space": "cosine", "signature": build_index_signature()},
    )
    if store.metadata.get("signature") != build_index_signature():
        raise ValueError(
            "Embedding or chunking settings changed. Run ingestion with --rebuild."
        )
    return store


def reset_paper_stores() -> None:
    """Delete both paper collections before an explicitly requested full rebuild.

    This clears child vectors and section records; SQLite chat and quiz history are
    separate.
    """

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    for name in (COLLECTION_NAME, "research_sections"):
        try:
            client.delete_collection(name)
        except chromadb.errors.NotFoundError:
            pass


def list_indexed_papers() -> list[dict]:
    """Return one title/source card per paper currently represented in Chroma.

    Deduplicate child metadata by document ID and sort by title for /papers and the
    browser's sources list.
    """

    records = open_chunk_store().get(include=["metadatas"])
    catalog = {row["pdf_filename"].casefold(): row for row in load_paper_catalog()}
    sources = {}
    for metadata in records["metadatas"]:
        sources[metadata["document_id"]] = {
            **catalog.get(metadata["title"].casefold(), {}),
            "id": metadata["document_id"],
            "title": metadata["title"],
            "source_type": "google_drive",
            "url": metadata["source_url"],
            "indexed_at": metadata["indexed_at"],
            "active": True,
        }
    return sorted(sources.values(), key=lambda source: source["title"].casefold())
