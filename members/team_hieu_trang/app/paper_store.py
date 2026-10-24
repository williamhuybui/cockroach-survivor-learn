"""Local Chroma collections hold searchable children and their full parent sections."""

import chromadb

from .config import (
    CHROMA_PATH,
    CHUNK_OVERLAP_CHARACTERS,
    EMBED_DIM,
    EMBED_MODEL,
    MAX_CHUNK_CHARACTERS,
)

COLLECTION_NAME = "research_papers"
INDEX_VERSION = "paper-parents-v4"


def parent_collection():
    """Store full sections in the same Chroma directory for child-to-parent lookup.

    Parent vectors are child centroids, not another model call. Retrieval uses
    child vectors and parent IDs; the parent collection is fetched by ID only.
    """
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_or_create_collection(
        name="research_sections", embedding_function=None
    )


def embedding_signature() -> str:
    """Identify the embedding and chunking rules used to build this index."""

    return f"{EMBED_MODEL}:{EMBED_DIM}:{INDEX_VERSION}:{MAX_CHUNK_CHARACTERS}:{CHUNK_OVERLAP_CHARACTERS}"


def collection():
    """Open the local cosine index without Chroma downloading another model.

    We provide EmbeddingGemma vectors ourselves during both indexing and search.
    An existing collection with another embedding model needs a deliberate rebuild.
    """

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    store = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
        metadata={"hnsw:space": "cosine", "signature": embedding_signature()},
    )
    if store.metadata.get("signature") != embedding_signature():
        raise ValueError(
            "Embedding or chunking settings changed. Run ingestion with --rebuild."
        )
    return store


def reset_collection() -> None:
    """Remove this app's old Chroma collection before an explicit full rebuild."""

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    try:
        client.delete_collection(COLLECTION_NAME)
    except chromadb.errors.NotFoundError:
        pass
    try:
        client.delete_collection("research_sections")
    except chromadb.errors.NotFoundError:
        pass


def source_rows() -> list[dict]:
    """Return one document card per indexed PDF for the sources screen."""

    records = collection().get(include=["metadatas"])
    sources = {}
    for metadata in records["metadatas"]:
        sources[metadata["document_id"]] = {
            "id": metadata["document_id"],
            "title": metadata["title"],
            "source_type": "google_drive",
            "url": metadata["source_url"],
            "indexed_at": metadata["indexed_at"],
            "active": True,
        }
    return sorted(sources.values(), key=lambda source: source["title"].casefold())
