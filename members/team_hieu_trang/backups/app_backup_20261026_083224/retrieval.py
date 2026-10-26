"""Embed a question, search Chroma, then rerank a small candidate set."""

import math
import re
from collections import Counter
from functools import lru_cache
from time import perf_counter

from .config import MIN_RERANK_SCORE, RERANK_MODEL
from .llm import embed_query
from .paper_store import collection, parent_collection

CANDIDATE_COUNT = 24
EVIDENCE_COUNT = 5
CONTEXT_CHARACTERS = 14000


def tokens(text: str) -> list[str]:
    """Keep words and numbers for literal technical-term matching."""
    return re.findall(r"[a-z0-9]+", text.casefold())


def keyword_candidates(
    query: str, records: list[dict], limit: int = CANDIDATE_COUNT
) -> list[dict]:
    """Rank text with BM25 (k1=1.5, b=0.75), independently of vector scores."""
    if not records:
        return []
    bags = [Counter(tokens(row["content"])) for row in records]
    lengths = [sum(bag.values()) for bag in bags]
    average = sum(lengths) / len(lengths) or 1
    terms = set(tokens(query))
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


def matching_records(paper: str | None = None) -> list[dict]:
    """Read chunks, optionally restricted to an exact paper chosen by the workflow."""
    saved = collection().get(include=["documents", "metadatas"])
    records = []
    for key, text, meta in zip(saved["ids"], saved["documents"], saved["metadatas"]):
        if paper is not None and meta.get("title") != paper:
            continue
        records.append({"id": key, "content": text, **meta})
    return records


def hybrid_candidates(query: str, paper: str | None = None) -> list[dict]:
    """Combine dense and BM25 hits for the scope already chosen by the planner."""
    records = matching_records(paper)
    if not records:
        return []
    eligible = {row["id"]: row for row in records}
    dense = retrieve_candidates(query, count=CANDIDATE_COUNT * 5, paper=paper)
    candidates = [eligible[row["id"]] for row in dense if row["id"] in eligible]
    candidates.extend(keyword_candidates(query, records))
    return list({row["id"]: row for row in candidates}.values())


def expand_parents(ranked: list[dict], limit: int) -> list[dict]:
    """Fetch one bounded parent window per section, retaining relevance order."""
    result, seen, remaining = [], set(), CONTEXT_CHARACTERS
    parents = parent_collection()
    for row in ranked:
        parent_id = row.get("parent_id", row["id"])
        if parent_id in seen:
            continue
        budget = min(2400, remaining)
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
            # Locate the actual child body; expanded context must still contain its hit.
            body = content.split("\n", 2)[-1]
            position = text.find(body)
            if position >= 0:
                start = max(0, position - 350)
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
def reranker():
    """Load the local cross-encoder once; loading it for every question is slow."""

    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANK_MODEL, max_length=512)


def retrieve_candidates(
    query: str, count: int = CANDIDATE_COUNT, paper: str | None = None
) -> list[dict]:
    """Find a small set of nearby vectors before the more costly reranker."""

    store = collection()
    if store.count() == 0:
        return []
    vector = embed_query(query)
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


def rerank_candidates(query: str, candidates: list[dict]) -> list[dict]:
    """Score question/passage pairs and discard weak matches before generation."""

    if not candidates:
        return []
    scores = reranker().predict([(query, row["content"]) for row in candidates])
    ranked = [
        {**row, "rerank_score": float(score)}
        for row, score in zip(candidates, scores)
        if float(score) >= MIN_RERANK_SCORE
    ]
    return sorted(ranked, key=lambda row: row["rerank_score"], reverse=True)


def search(
    query: str,
    limit: int = EVIDENCE_COUNT,
    paper: str | None = None,
    diagnostics: dict | None = None,
) -> list[dict]:
    """Retrieve and rerank passages with their exact Drive link and PDF location."""

    started = perf_counter()
    candidates = hybrid_candidates(query, paper)
    retrieved_at = perf_counter()
    ranked = rerank_candidates(query, candidates)
    ranked_at = perf_counter()
    evidence = expand_parents(ranked, limit)
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


def paper_for_daily(day_number: int) -> list[dict]:
    """Choose one indexed paper deterministically for the daily quiz."""

    store = collection()
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
    print(json.dumps(search(args.query), indent=2, ensure_ascii=False))
