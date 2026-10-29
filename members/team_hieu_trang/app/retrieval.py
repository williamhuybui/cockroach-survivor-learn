"""Find research-paper evidence for a question before Gemma writes an answer.

Combine Chroma vector search with Python BM25 keyword search, remove duplicate chunk
IDs, and rerank question/chunk pairs with MiniLM. Expand accepted children into
bounded section text and keep citation metadata for grounding.
"""

import math
import re
from collections import Counter
from functools import lru_cache
from time import perf_counter

from threading import Lock

from . import paper_store
from .config import MIN_RERANK_SCORE, RERANK_MODEL
from .llm import embed_search_question
from .paper_store import open_chunk_store, open_parent_section_store
from .study_materials import load_study_passages

CANDIDATE_COUNT = 24
EVIDENCE_COUNT = 5
# All tasks share this character allowance for passage text sent to the answer model.
CONTEXT_CHARACTERS = 14000


def tokenize_search_text(text: str) -> list[str]:
    """Extract lowercase words and numbers for BM25 keyword matching.

    Use the same token rules for questions and indexed chunks so technical names and
    numeric terms can match literally.
    """
    return re.findall(r"[a-z0-9]+", text.casefold())


def find_bm25_candidates(
    query: str, records: list[dict], limit: int = CANDIDATE_COUNT
) -> list[dict]:
    """Rank chunks by keyword relevance and return the highest-scoring candidates.

    BM25 rewards matching query terms while accounting for term frequency and chunk
    length. It runs locally in Python on the supplied records; it does not require a
    separate search server.
    """
    if not records:
        return []
    bags = [Counter(tokenize_search_text(row["content"])) for row in records]
    lengths = [sum(bag.values()) for bag in bags]
    average = sum(lengths) / len(lengths) or 1
    terms = set(tokenize_search_text(query))
    frequencies = {term: sum(term in bag for bag in bags) for term in terms}
    scored = []
    for row, bag, length in zip(records, bags, lengths):
        score = 0.0
        for term in terms:
            frequency = bag[term]
            if frequency:
                inverse = math.log1p(
                    (len(records) - frequencies[term] + 0.5) / (frequencies[term] + 0.5)
                )
                score += (
                    inverse
                    * frequency
                    * 2.5
                    / (frequency + 1.5 * (0.25 + 0.75 * length / average))
                )
        if score:
            scored.append({**row, "bm25_score": score})
    return sorted(scored, key=lambda row: row["bm25_score"], reverse=True)[:limit]


def index_version() -> int:
    """Chroma rewrites its SQLite file on ingestion, so its mtime invalidates caches."""
    database = paper_store.CHROMA_PATH / "chroma.sqlite3"
    return database.stat().st_mtime_ns if database.exists() else 0


@lru_cache(maxsize=1)
def all_records(version: int) -> tuple[dict, ...]:
    """Read every chunk once per index version instead of once per search task."""
    saved = collection().get(include=["documents", "metadatas"])
    return tuple(
        {"id": key, "content": text, **meta}
        for key, text, meta in zip(saved["ids"], saved["documents"], saved["metadatas"])
    )


def load_scoped_chunks(paper: str | None = None) -> list[dict]:
    """Read child text and metadata from Chroma for keyword search.

    Unscoped searches also read the catalog and guide as keyword candidates.
    When a task names an exact indexed paper, keep only that paper's chunks so
    retrieval respects the planned scope.
    """
    saved = open_chunk_store().get(include=["documents", "metadatas"])
    records = []
    for key, text, meta in zip(saved["ids"], saved["documents"], saved["metadatas"]):
        if paper is not None and meta.get("title") != paper:
            continue
        records.append({"id": key, "content": text, **meta})
    # Learning references join unscoped searches; exact-paper searches stay PDF-only.
    if paper is None:
        records.extend(load_study_passages())
    return records


def find_hybrid_candidates(query: str, paper: str | None = None) -> list[dict]:
    """Combine meaning-based and keyword-based retrieval for one search task.

    Use the same paper scope for both searches, then merge their hits by chunk ID.
    This step collects candidates; the cross-encoder makes the final relevance
    ordering.
    """
    records = load_scoped_chunks(paper)
    if not records:
        return []
    eligible = {row["id"]: row for row in records}
    dense = find_vector_candidates(query, count=CANDIDATE_COUNT * 5, paper=paper)
    candidates = [eligible[row["id"]] for row in dense if row["id"] in eligible]
    candidates.extend(find_bm25_candidates(query, records))
    return list({row["id"]: row for row in candidates}.values())


def expand_parent_section_windows(ranked: list[dict], limit: int) -> list[dict]:
    """Add surrounding section text to accepted child hits without exceeding the context budget.

    Fetch each child's containing section, locate the child text and take a bounded
    window around it. Keep one window per section in relevance order so repeated
    child hits do not fill the answer context with duplicates.
    """
    result, seen, remaining = [], set(), CONTEXT_CHARACTERS
    parents = open_parent_section_store()
    for row in ranked:
        parent_id = row.get("parent_id", row["id"])
        if parent_id in seen:
            continue
        budget = min(WINDOW_CHARACTERS, remaining)
        if budget < 400:
            break
        content = row["content"]
        parent = (
            parents.get(ids=[parent_id], include=["documents"])
            if row.get("parent_id")
            else {"documents": []}
        )
        if parent["documents"]:
            text = parent["documents"][0]
            # Remove the title/heading prefix to find the child inside stored section text.
            body = content.split("\n", 2)[-1]
            position = text.find(body)
            if position >= 0:
                start = max(0, position - 200)
                content = text[start : start + budget]
        result.append(
            {**row, "content": content[:budget], "child_content": row["content"]}
        )
        remaining -= len(result[-1]["content"])
        seen.add(parent_id)
        if len(result) >= limit:
            break
    return result


@lru_cache(maxsize=1)
def get_reranker_model():
    """Load and reuse the MiniLM cross-encoder that scores question/passage pairs.

    Unlike embedding search, it reads each question and candidate together. Cache
    the model instance to avoid reloading its weights for every question.
    """

    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANK_MODEL, max_length=512)


def find_vector_candidates(
    query: str, count: int = CANDIDATE_COUNT, paper: str | None = None
) -> list[dict]:
    """Embed the question and ask Chroma for nearby child-chunk vectors.

    Apply an exact-paper restriction before selecting results when the planner
    provides one. Return candidate text and citation data for the shared hybrid-
    search pipeline.
    """

    store = open_chunk_store()
    if store.count() == 0:
        return []
    vector = embed_search_question(query)
    result = store.query(
        query_embeddings=[vector],
        n_results=min(count, store.count()),
        include=["documents", "metadatas", "distances"],
        **({"where": {"title": paper}} if paper else {}),
    )
    candidates = []
    for chunk_id, content, metadata, distance in zip(
        result["ids"][0],
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        candidates.append(
            {
                "id": chunk_id,
                "content": content,
                "title": metadata["title"],
                "source_url": metadata["source_url"],
                "location": metadata["location"],
                "document_type": metadata["document_type"],
                "vector_distance": distance,
            }
        )
    return candidates


def rerank_passage_candidates(query: str, candidates: list[dict]) -> list[dict]:
    """Score each question/chunk pair and keep candidates above the relevance threshold.

    Sort accepted hits by their MiniLM scores before section expansion. These scores
    are model relevance scores, not probabilities that an answer is correct.
    """

    if not candidates:
        return []
    scores = get_reranker_model().predict(
        [(query, row["content"]) for row in candidates]
    )
    ranked = [
        {**row, "rerank_score": float(score)}
        for row, score in zip(candidates, scores)
        if float(score) >= min_score
    ]
    return sorted(ranked, key=lambda row: row["rerank_score"], reverse=True)


def retrieve_relevant_passages(
    query: str,
    limit: int = EVIDENCE_COUNT,
    paper: str | None = None,
    diagnostics: dict | None = None,
    fallback: int = 0,
) -> list[dict]:
    """Run hybrid retrieval, neural reranking and section expansion for one question.

    Return evidence with source links and PDF locations. Optional diagnostics record
    stage timings and intermediate passages so offline evaluation can distinguish
    retrieval failures from later context selection.
    """

    started = perf_counter()
    candidates = find_hybrid_candidates(query, paper)
    retrieved_at = perf_counter()
    # Score the small child chunks first, then add surrounding section context.
    ranked = rerank_passage_candidates(query, candidates)
    ranked_at = perf_counter()
    evidence = expand_parent_section_windows(ranked, limit)
    if diagnostics is not None:
        if diagnostics.get("capture_evidence"):
            # Preserve retrieval order and stable IDs only in offline audit reports.
            diagnostics.update(
                candidate_passages=candidates,
                reranked_passages=ranked,
                expanded_passages=evidence,
            )
        diagnostics.update(
            {
                "candidate_sources": sorted({row["title"] for row in candidates}),
                "reranked_sources": sorted({row["title"] for row in ranked}),
                "candidate_count": len(candidates),
                "accepted_count": len(ranked),
                "evidence_count": len(evidence),
                "retrieval_seconds": round(retrieved_at - started, 3),
                "reranking_seconds": round(ranked_at - retrieved_at, 3),
                "expansion_seconds": round(perf_counter() - ranked_at, 3),
            }
        )
    return evidence


def select_daily_paper_passages(day_number: int) -> list[dict]:
    """Select one indexed paper and a few child chunks for the daily quiz.

    Use the calendar-day number to choose consistently from sorted paper IDs, then
    return up to five chunks as quiz evidence.
    """

    store = open_chunk_store()
    records = store.get(include=["metadatas"])
    document_ids = sorted({meta["document_id"] for meta in records["metadatas"]})
    if not document_ids:
        return []
    chosen = document_ids[day_number % len(document_ids)]
    records = store.get(
        where={"document_id": chosen}, include=["documents", "metadatas"], limit=5
    )
    evidence = []
    for chunk_id, content, metadata in zip(
        records["ids"], records["documents"], records["metadatas"]
    ):
        evidence.append(
            {
                "id": chunk_id,
                "content": content,
                "title": metadata["title"],
                "source_url": metadata["source_url"],
                "location": metadata["location"],
            }
        )
    return evidence


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Inspect retrieved evidence without generating an answer"
    )
    parser.add_argument("query")
    args = parser.parse_args()
    print(
        json.dumps(retrieve_relevant_passages(args.query), indent=2, ensure_ascii=False)
    )
