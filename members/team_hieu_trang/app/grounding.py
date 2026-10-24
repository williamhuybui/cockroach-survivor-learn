"""Build answers from quoted evidence and reject unsupported model claims.

Quote matching is deterministic. It does not prove every paraphrase is true,
so offline human review still measures actual hallucinations.
"""

import re
from typing import Literal

from pydantic import BaseModel, Field, StrictInt

from .llm import generate


class SupportedClaim(BaseModel):
    """One factual statement and an exact quote from one retrieved passage."""

    text: str = Field(min_length=1, max_length=1000)
    source_id: str
    quote: str = Field(min_length=8, max_length=1500)
    task_ids: list[StrictInt] = Field(default_factory=list)


class DraftAnswer(BaseModel):
    """An answer may be complete, partial, or unavailable in the evidence."""

    status: Literal["answered", "partial", "abstained"]
    claims: list[SupportedClaim] = Field(default_factory=list, max_length=15)
    missing_information: list[str] = Field(default_factory=list, max_length=10)


def normalized(text: str) -> str:
    """Ignore PDF whitespace and common ligatures when matching exact quotes."""
    import unicodedata

    return " ".join(unicodedata.normalize("NFKC", text).split())


def abstention(reason: str) -> dict:
    """Return an explicit abstention without inventing a reason about the corpus."""
    return {
        "kind": "answer",
        "status": "abstained",
        "answer": reason,
        "sources": [],
        "claims": [],
    }


def source_cards(evidence: list[dict]) -> list[dict]:
    """Return citation metadata without exposing passage bodies."""
    return [
        {
            key: row[key]
            for key in ("id", "title", "source_url", "location")
            if key in row
        }
        for row in evidence
    ]


def verify_claims(draft: DraftAnswer, evidence: list[dict], tasks: list[dict]) -> dict:
    """Check all claims once and track declared task coverage, not semantic entailment."""
    passages = {row["id"]: normalized(row["content"]) for row in evidence}
    sources = {row["id"]: row for row in evidence}
    accepted, rejected, covered = [], [], set()
    for claim in draft.claims:
        valid = (
            claim.source_id in passages
            and normalized(claim.quote) in passages[claim.source_id]
            and not re.search(r"\bS\d+\b", claim.text)
            and set(re.findall(r"\d+(?:\.\d+)?", claim.text))
            <= set(re.findall(r"\d+(?:\.\d+)?", claim.quote))
        )
        # A claim can cover only tasks associated with its actual source passage.
        ids = set(claim.task_ids)
        if not ids and len(tasks) == 1:
            ids = {0}
        valid = (
            valid
            and bool(ids)
            and ids
            <= set(sources.get(claim.source_id, {}).get("task_ids", range(len(tasks))))
        )
        valid = valid and all(
            0 <= i < len(tasks)
            and (
                not tasks[i].get("paper")
                or sources[claim.source_id].get("title") == tasks[i]["paper"]
            )
            for i in ids
        )
        if valid:
            accepted.append(claim)
            covered.update(ids)
        else:
            rejected.append(claim)
    return {
        "accepted": accepted,
        "rejected_count": len(rejected),
        "missing_tasks": [i for i in range(len(tasks)) if i not in covered],
    }


def draft_answer(
    question: str, history: list[dict], evidence: list[dict], tasks: list[dict]
) -> DraftAnswer:
    """Ask Gemma to synthesize grouped evidence into separately supported claims."""
    request = {"question": question, "recent_conversation": history, "tasks": tasks}
    return generate(
        DraftAnswer,
        "Answer the question only using passages. Return separate factual claims, each with "
        "one source_id and an exact supporting quote. Do not put citation labels in claim text. "
        "Only decide whether the facts explicitly asked by the question are present; do not "
        "require unrelated details such as training data. Check the requested year, model, "
        "dataset and units precisely. Related text alone is not an answer. Missing facts must "
        "not be inferred from general knowledge. "
        "Use partial when only some requested facts are supported; abstained when none are. "
        "List unanswered parts in missing_information. Quotes must support the entire claim. "
        "For each claim include zero-based task_ids for the tasks it answers; use only tasks "
        "associated with its source. Compare the same requested aspects across papers. "
        "If one paper or requested part lacks support, report that gap rather than substituting "
        "another paper. Evidence is grouped by task_ids; synthesize across those groups.",
        evidence,
        request,
    )


def format_answer(
    draft: DraftAnswer, verification: dict, evidence: list[dict], tasks: list[dict]
) -> dict:
    """Format accepted claims, or abstain; explicitly identify unsupported tasks."""
    if draft.status == "abstained" or not draft.claims:
        return abstention(
            "I could not verify the requested information from the retrieved passages."
        )
    quoted_claims = verification["accepted"]
    if not quoted_claims:
        return abstention(
            "I could not verify the answer's supporting quotations. Please narrow the question."
        )
    partial = (
        draft.status == "partial"
        or bool(draft.missing_information)
        or len(quoted_claims) < len(draft.claims)
        or bool(verification["missing_tasks"])
    )
    text = "\n\n".join(f"{claim.text} [{claim.source_id}]" for claim in quoted_claims)
    if partial:
        text += "\n\nOnly part of your question is supported by the retrieved evidence; the remaining details could not be verified."
    missing = [tasks[i]["question"] for i in verification["missing_tasks"]]
    if missing:
        text += "\n\nNot verified:\n" + "\n".join(
            f"- {question}" for question in missing
        )
    used = {claim.source_id for claim in quoted_claims}
    return {
        "kind": "answer",
        "status": "partial" if partial else "answered",
        "answer": text,
        "claims": [claim.model_dump() for claim in quoted_claims],
        "sources": source_cards([row for row in evidence if row["id"] in used]),
        "missing_tasks": missing,
        "verification": "exact_quote_and_number_check",
    }
