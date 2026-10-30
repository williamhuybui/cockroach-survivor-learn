"""Find research-paper evidence for a question before Gemma writes an answer.

Combine Chroma vector search with Python BM25 keyword search, remove duplicate chunk
IDs, and rerank question/chunk pairs with MiniLM. Expand accepted children into
bounded section text and keep citation metadata for grounding.
"""

import math
import re
from collections import Counter
from itertools import zip_longest
from threading import Lock
from time import perf_counter

from .config import MIN_RERANK_SCORE, RERANK_MODEL
from .llm import embed_search_question
from .paper_store import open_chunk_store, open_parent_section_store
from .study_materials import load_study_passages

CANDIDATE_COUNT = 24
EVIDENCE_COUNT = 5
# All tasks share this character allowance for passage text sent to the answer model.
CONTEXT_CHARACTERS = 14000

# Load weights once per process. This lock is never held while scoring passages.
_reranker_model = None
_reranker_initialization_lock = Lock()


def expand_defined_abbreviations(query: str, paper: str | None = None) -> str:
    """Replace query abbreviations with unambiguous definitions from indexed text.

    For example, a stored definition 'natural language processing (NLP)' lets
    MiniLM see 'natural language processing' when the question says 'NLP'. Prefer
    definitions in the requested paper. Unknown or ambiguous terms stay unchanged;
    neither document text nor the question used to generate the answer is changed.
    """
    # Find uppercase short names, not ordinary words or parts of model names.
    abbreviations = set(re.findall(r"(?<![\w-])[A-Z]{2,10}(?![\w-])", query))
    if not abbreviations:
        return query
    filters = [{"$contains": f"({name})"} for name in sorted(abbreviations)]
    saved = open_chunk_store().get(
        where_document=filters[0] if len(filters) == 1 else {"$or": filters},
        include=["documents", "metadatas"],
    )
    definitions = {name: set() for name in abbreviations}
    scoped_definitions = {name: set() for name in abbreviations}
    for text, metadata in zip(saved["documents"], saved["metadatas"]):
        # Find words just before '(ABC)'; newlines and hyphens separate words.
        for match in re.finditer(r"\(([A-Z]{2,10})\)", text):
            name = match[1]
            if name not in abbreviations:
                continue
            words = re.findall(
                r"[A-Za-z]+", text[max(0, match.start() - 160) : match.start()]
            )
            # Test the shortest suffix first so preceding prose is not included.
            for length in range(2, min(len(words), 10) + 1):
                phrase = words[-length:]
                initials = "".join(
                    word[0]
                    for word in phrase
                    if word.casefold()
                    not in {"a", "an", "the", "of", "for", "from", "and", "in", "to"}
                )
                if initials.upper() == name:
                    definition = " ".join(phrase).casefold()
                    definitions[name].add(definition)
                    if paper and metadata.get("title") == paper:
                        scoped_definitions[name].add(definition)
                    break
    for name in sorted(abbreviations):
        choices = scoped_definitions[name] or definitions[name]
        if len(choices) == 1:
            query = re.sub(
                rf"(?<![\w-]){re.escape(name)}(?![\w-])", next(iter(choices)), query
            )
    return query


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
    # This term weight depends on the whole collection, not the individual chunk.
    frequencies = {term: sum(term in bag for bag in bags) for term in terms}
    term_weights = {
        term: math.log1p(
            (len(records) - frequency + 0.5) / (frequency + 0.5)
        )
        for term, frequency in frequencies.items()
    }
    scored = []
    for row, bag, length in zip(records, bags, lengths):
        score = 0.0
        for term in terms:
            frequency = bag[term]
            if frequency:
                score += (
                    term_weights[term]
                    * frequency
                    * 2.5
                    / (frequency + 1.5 * (0.25 + 0.75 * length / average))
                )
        if score:
            scored.append({**row, "bm25_score": score})
    return sorted(scored, key=lambda row: row["bm25_score"], reverse=True)[:limit]


def paper_query_filter(paper: str | list[str] | None) -> dict:
    """Use the same exact-title or candidate-title filter in both Chroma searches."""
    if paper is None:
        return {}
    if isinstance(paper, list):
        if not paper:
            raise ValueError("Candidate paper scope cannot be empty")
        return {"where": {"title": {"$in": paper}}}
    return {"where": {"title": paper}}


def load_scoped_chunks(paper: str | list[str] | None = None) -> list[dict]:
    """Read child text and metadata from Chroma for keyword search.

    Unscoped searches also read the catalog and guide as keyword candidates.
    When a task names an exact indexed paper, keep only that paper's chunks so
    retrieval respects the planned scope.
    """
    saved = open_chunk_store().get(
        include=["documents", "metadatas"],
        **paper_query_filter(paper),
    )
    records = []
    allowed = paper if isinstance(paper, list) else [paper]
    for key, text, meta in zip(saved["ids"], saved["documents"], saved["metadatas"]):
        if paper is not None and meta.get("title") not in allowed:
            continue
        records.append({"id": key, "content": text, **meta})
    # Learning references join unscoped searches; exact-paper searches stay PDF-only.
    if paper is None:
        records.extend(load_study_passages())
    return records


def find_hybrid_candidates(
    query: str, paper: str | list[str] | None = None
) -> list[dict]:
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
            # Remove the title/heading prefix to find the child inside stored section text.
            body = content.split("\n", 2)[-1]
            position = text.find(body)
            if len(text) <= budget:
                # A short section fits as-is; starting near its second child
                # would unnecessarily drop earlier list items or definitions.
                content = text
            elif position >= 0:
                start = max(0, position - 350)
                start = min(start, len(text) - budget)
                content = text[start : start + budget]
        result.append(
            {**row, "content": content[:budget], "child_content": row["content"]}
        )
        remaining -= len(result[-1]["content"])
        seen.add(parent_id)
        if len(result) >= limit:
            break
    return result


def get_reranker_model():
    """Load and reuse the MiniLM cross-encoder that scores question/passage pairs.

    Only simultaneous first callers wait for model loading. Once loaded, requests
    reuse the model without an application lock around prediction. Do not train or
    reconfigure this shared model while the server is answering questions.
    """

    global _reranker_model
    if _reranker_model is None:
        with _reranker_initialization_lock:
            # Another request may have finished loading while this one waited.
            if _reranker_model is None:
                from sentence_transformers import CrossEncoder

                _reranker_model = CrossEncoder(RERANK_MODEL, max_length=512)
    return _reranker_model


def find_vector_candidates(
    query: str, count: int = CANDIDATE_COUNT, paper: str | list[str] | None = None
) -> list[dict]:
    """Embed the question and ask Chroma for nearby child-chunk vectors.

    Apply an exact-paper restriction before selecting results when the planner
    provides one. Return candidate text and citation data for the shared hybrid-
    search pipeline.
    """

    store = open_chunk_store()
    total_chunks = store.count()
    if total_chunks == 0:
        return []
    vector = embed_search_question(query)
    result = store.query(
        query_embeddings=[vector],
        n_results=min(count, total_chunks),
        include=["documents", "metadatas", "distances"],
        **paper_query_filter(paper),
    )
    return [
        {
            "id": chunk_id,
            "content": content,
            "title": metadata["title"],
            "source_url": metadata["source_url"],
            "location": metadata["location"],
            "document_type": metadata["document_type"],
            "vector_distance": distance,
        }
        for chunk_id, content, metadata, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        )
    ]


def rerank_passage_candidates(query: str, candidates: list[dict]) -> list[dict]:
    """Score each question/chunk pair and sort candidates by relevance.

    Scores indicate relevance, not the probability that an answer is correct.
    """

    if not candidates:
        return []
    scores = get_reranker_model().predict(
        [(query, row["content"]) for row in candidates]
    )
    scored = [
        {**row, "rerank_score": float(score)}
        for row, score in zip(candidates, scores)
    ]
    ranked = [row for row in scored if MIN_RERANK_SCORE is None or row["rerank_score"] >= MIN_RERANK_SCORE]
    return sorted(ranked, key=lambda row: row["rerank_score"], reverse=True)


def retrieve_relevant_passages(
    query: str,
    limit: int = EVIDENCE_COUNT,
    paper: str | list[str] | None = None,
    diagnostics: dict | None = None,
    resolve_candidates: bool = False,
) -> list[dict]:
    """Run hybrid retrieval, neural reranking and section expansion for one question.

    Return evidence with source links and PDF locations. A verified paper set is
    searched one paper at a time, then interleaved to give each an early chance.
    Optional diagnostics record stage timings and intermediate passages. Set
    resolve_candidates only for an uncertain identity, which needs joint discovery.
    """

    if isinstance(paper, list) and not resolve_candidates:
        if not paper:
            raise ValueError("Paper scope cannot be empty")
        groups, traces = [], []
        for title in paper:
            trace = {
                "capture_evidence": bool(
                    diagnostics and diagnostics.get("capture_evidence")
                )
            }
            groups.append(
                retrieve_relevant_passages(
                    query, limit=limit, paper=title, diagnostics=trace
                )
            )
            traces.append(trace)
        # Alternate papers while retaining MiniLM's order within each paper.
        evidence = [row for position in zip_longest(*groups) for row in position if row]
        if diagnostics is not None:
            for field in ("candidate_sources", "reranked_sources"):
                diagnostics[field] = sorted(
                    {value for trace in traces for value in trace.get(field, [])}
                )
            for field in (
                "candidate_count",
                "accepted_count",
                "retrieval_seconds",
                "reranking_seconds",
                "expansion_seconds",
            ):
                diagnostics[field] = sum(trace.get(field, 0) for trace in traces)
            if diagnostics.get("capture_evidence"):
                for field in (
                    "candidate_passages",
                    "reranked_passages",
                    "expanded_passages",
                ):
                    diagnostics[field] = [
                        row
                        for position in zip_longest(
                            *(trace.get(field, []) for trace in traces)
                        )
                        for row in position
                        if row
                    ]
            diagnostics.update(
                original_query=query,
                search_query=query,
                paper_searches=traces,
                resolved_paper=None,
                scope_ambiguous=False,
                evidence_count=len(evidence),
            )
        return evidence

    started = perf_counter()
    search_query = expand_defined_abbreviations(
        query, paper if isinstance(paper, str) else None
    )
    candidates = find_hybrid_candidates(search_query, paper)
    retrieved_at = perf_counter()
    # Score the small child chunks first, then add surrounding section context.
    ranked = rerank_passage_candidates(search_query, candidates)
    ranked_at = perf_counter()
    usable = ranked
    resolved_paper = None
    ambiguous = False
    if resolve_candidates:
        # An uncertain identity must resolve before combining its evidence.
        # A candidate name may identify one paper only if all ranked hits agree.
        titles = {row["title"] for row in ranked}
        if len(titles) == 1:
            resolved_paper = next(iter(titles))
        else:
            usable = []
            ambiguous = len(titles) > 1
    evidence = expand_parent_section_windows(usable, limit)
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
                "resolved_paper": resolved_paper,
                "scope_ambiguous": ambiguous,
                "original_query": query,
                "search_query": search_query,
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
