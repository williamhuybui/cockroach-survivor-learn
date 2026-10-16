from datetime import date
from langsmith import traceable
from sqlalchemy import select, func, literal_column

from .config import EMBED_DIM, EMBED_MODEL
from .db import ActionItem, Chunk, Document, Session, Setting
from .llm import embed_texts


def evidence_row(chunk, doc):
    """Attach source identity and location to a retrieved passage."""
    return {"id": chunk.id, "content": chunk.content, "title": doc.title, "source_url": doc.source_url,
            "location": chunk.location, "session_date": str(doc.session_date) if doc.session_date else None,
            "indexed_at": doc.indexed_at.isoformat()}


@traceable(name="pgvector_search", run_type="retriever")
def search(query: str, source_type: str | None = None, topic: str | None = None, author: str | None = None):
    """Retrieve active vector and keyword candidates, then combine their rankings."""
    vector = embed_texts([query])[0]
    with Session() as db:
        setting = db.get(Setting, "embedding_signature")
        if setting and not setting.value.startswith(f"{EMBED_MODEL}:{EMBED_DIM}:"):
            raise ValueError("Query embedding model differs from the indexed embedding model")
        stmt = select(Chunk, Document).join(Document).where(Document.active.is_(True))
        if source_type:
            stmt = stmt.where(Document.source_type == source_type)
        if topic:
            stmt = stmt.where(Document.topic.ilike(topic))
        if author:
            stmt = stmt.where(Document.author.ilike(author))
        vector_rows = list(db.execute(stmt.order_by(Chunk.embedding.cosine_distance(vector)).limit(20)))
        # 'simple' preserves technical terms without English stemming. This is not BM25.
        text_vector = func.to_tsvector(literal_column("'simple'"), Chunk.content)
        text_query = func.websearch_to_tsquery(literal_column("'simple'"), query)
        keyword_rows = list(db.execute(
            stmt.where(text_vector.op("@@")(text_query))
            .order_by(func.ts_rank_cd(text_vector, text_query).desc()).limit(20)
        ))
        return fuse_results(vector_rows, keyword_rows)


def fuse_results(vector_rows, keyword_rows, limit=4):
    """Merge ranks with RRF so incomparable keyword/vector scores are never added directly."""
    scores, evidence = {}, {}
    for rows in (vector_rows, keyword_rows):
        for rank, (chunk, document) in enumerate(rows, 1):
            scores[chunk.id] = scores.get(chunk.id, 0) + 1 / (60 + rank)
            evidence[chunk.id] = evidence_row(chunk, document)
    ordered = sorted(scores, key=lambda key: (-scores[key], key))
    return [evidence[key] for key in ordered[:limit]]


@traceable(name="postgres_week_materials")
def week_materials(start: str, end: str):
    """Fetch dated passages in a bounded interval without silently dropping sessions."""
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    if first > last or (last - first).days > 31:
        raise ValueError("Date range must be ordered and at most 31 days")
    with Session() as db:
        stmt = (select(Chunk, Document).join(Document).where(Document.active.is_(True),
                Document.session_date >= first, Document.session_date <= last)
                .order_by(Document.session_date, Document.id, Chunk.id).limit(101))
        rows = [evidence_row(c, d) for c, d in db.execute(stmt)]
    if len(rows) > 100 or sum(len(r["content"]) for r in rows) > 16000:
        raise ValueError("Too much material for one complete summary. Request a smaller date range.")
    return rows


@traceable(name="postgres_action_items")
def action_items(owner: str | None = None, status: str | None = None):
    """Read explicitly recorded action owners and statuses without inferring them."""
    if status and status not in {"open", "in_progress", "done"}:
        raise ValueError("Status must be open, in_progress, or done")
    with Session() as db:
        stmt = select(ActionItem, Document).join(Document).where(Document.active.is_(True))
        if owner:
            stmt = stmt.where(ActionItem.owner.ilike(owner))
        if status:
            stmt = stmt.where(ActionItem.status == status)
        rows = list(db.execute(stmt.order_by(ActionItem.id).limit(101)))
        if len(rows) > 100:
            raise ValueError("More than 100 matching action items; narrow the owner or status filter")
        return [{"id": item.id, "content": f"{item.description}; owner={item.owner}; status={item.status}; due={item.due_date}",
                 "title": doc.title, "source_url": doc.source_url, "location": item.location,
                 "indexed_at": doc.indexed_at.isoformat()} for item, doc in rows]
