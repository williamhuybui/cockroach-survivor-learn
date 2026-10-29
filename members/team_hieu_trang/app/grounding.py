"""Generate evidence-based answer claims and check their references before display.

Gemma proposes claims with quotes, source labels and task IDs. Python checks that
those references and numbers occur in the supplied evidence. Passing these checks
does not prove a quote supports every interpretation; human evaluation still checks
factual support.
"""

import re
from typing import Literal

from pydantic import BaseModel, Field, StrictInt

from .llm import generate_from_evidence


# Gemma proposes one statement with a quotation, source label and task indexes.
# Verification checks these references before the statement can appear in an answer.
class SupportedClaim(BaseModel):
    """One factual statement and an exact quote from one retrieved passage."""

    text: str = Field(min_length=1, max_length=1000)
    source_id: str
    quote: str = Field(min_length=8, max_length=1500)
    task_ids: list[StrictInt] = Field(default_factory=list)


# This is Gemma's structured draft, not the final verified response.
# Its claims and reported gaps are checked before formatting the user-facing answer.
class DraftAnswer(BaseModel):
    """An answer may be complete, partial, or unavailable in the evidence."""

    status: Literal["answered", "partial", "abstained"]
    claims: list[SupportedClaim] = Field(default_factory=list, max_length=15)
    missing_information: list[str] = Field(default_factory=list, max_length=10)


def normalize_quote_text(text: str) -> str:
    """Normalize Unicode typography and whitespace before comparing quotations.

    This tolerates PDF spacing and joined-letter characters while still requiring
    the quoted words to occur in the evidence.
    """
    import unicodedata

    return " ".join(unicodedata.normalize("NFKC", text).split())


def build_abstention_response(reason: str) -> dict:
    """Build a standard response explaining why a supported answer is unavailable.

    Return no claims or sources so callers do not accidentally present an
    unsupported answer as verified.
    """
    return {
        "kind": "answer",
        "status": "abstained",
        "answer": reason,
        "sources": [],
        "claims": [],
    }


def build_source_citations(evidence: list[dict]) -> list[dict]:
    """Select the source label, paper title, link and location for display.

    Keep passage bodies in the evidence used for generation and evaluation rather
    than repeating them in the citation list.
    """
    return [
        {
            key: row[key]
            for key in ("id", "title", "source_url", "location")
            if key in row
        }
        for row in evidence
    ]


def verify_answer_claims(
    draft: DraftAnswer, evidence: list[dict], tasks: list[dict]
) -> dict:
    """Reject claims with invalid source labels, missing quotes, new numbers or wrong task scope.

    Return accepted claims and the tasks they declare coverage for. These are
    text/reference checks: they cannot prove that a paraphrase follows from its
    quotation.
    """
    passages = {row["id"]: normalize_quote_text(row["content"]) for row in evidence}
    sources = {row["id"]: row for row in evidence}
    accepted, covered = [], set()
    for claim in draft.claims:
        # Reject unknown sources and quotes that do not occur in the passage.
        source = sources.get(claim.source_id)
        if (
            source is None
            or normalize_quote_text(claim.quote) not in passages[claim.source_id]
        ):
            continue
        # Citation labels belong in source_id, not inside the claim's text.
        if re.search(r"\bS\d+\b", claim.text):
            continue
        # Every number in the claim must also appear in its supporting quote.
        claim_numbers = set(re.findall(r"\d+(?:\.\d+)?", claim.text))
        quote_numbers = set(re.findall(r"\d+(?:\.\d+)?", claim.quote))
        if not claim_numbers <= quote_numbers:
            continue
        # A claim can cover only tasks associated with its actual source passage.
        ids = set(claim.task_ids)
        if not ids and len(tasks) == 1:
            ids = {0}
        allowed_tasks = set(source.get("task_ids", range(len(tasks))))
        if not ids or not ids <= allowed_tasks:
            continue
        if any(
            i < 0
            or i >= len(tasks)
            or (tasks[i].get("paper") and source.get("title") != tasks[i]["paper"])
            for i in ids
        ):
            continue
        accepted.append(claim)
        covered.update(ids)
    return {
        "accepted": accepted,
        "rejected_count": len(draft.claims) - len(accepted),
        "missing_tasks": [i for i in range(len(tasks)) if i not in covered],
    }


def generate_grounded_draft(
    question: str, history: list[dict], evidence: list[dict], tasks: list[dict]
) -> DraftAnswer:
    """Ask Gemma to answer using only the selected passages and return structured claims.

    Send the question, short history and task list alongside evidence. Require a
    supporting quote and source/task association for each claim, and instruct the
    model to report gaps instead of guessing table values or missing facts.
    """
    request = {"question": question, "recent_conversation": history, "tasks": tasks}
    return generate_from_evidence(
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
        "another paper. Evidence is grouped by task_ids; synthesize across those groups. "
        "Tables, figures and equations are not interpreted. Captions support only their "
        "explicit words and a pointer to the labeled item on that page. Do not infer "
        "table-cell relationships or graph values from isolated text or numbers. If a "
        "requested value is only in an unread table, report the gap and point to its "
        "caption when available; never claim to have read its cells. Use citation "
        "metadata for page locations; do not invent page numbers in factual claims. "
        "A source marked structure_confidence=uncertain is page-based fallback text. "
        "Its words can support a quoted fact, but its ordering and section relationships "
        "are not verified; do not infer those relationships from that source. "
        "Evidence marked study_guide contains learning recommendations, not universal rules. "
        "Attribute its reading depths, priorities and prerequisites to the supplied guide. "
        "Specific focus notes may qualify the guide's default reading-depth table. "
        "Its time estimates describe human study, not runtime limits. Paper_catalog evidence "
        "supports catalog metadata such as phase/year, not experimental findings. Use PDF "
        "evidence for research claims, and do not claim to have examined excluded visuals.",
        evidence,
        request,
    )


def format_verified_answer(
    draft: DraftAnswer, verification: dict, evidence: list[dict], tasks: list[dict]
) -> dict:
    """Turn accepted claims into an answer with source labels and explicit gaps.

    Return a partial answer when some claims or tasks lack support, and abstain when
    none pass the checks. Include only citations used by accepted claims.
    """
    if draft.status == "abstained" or not draft.claims:
        return build_abstention_response(
            "I could not verify the requested information from the retrieved passages."
        )
    quoted_claims = verification["accepted"]
    if not quoted_claims:
        return build_abstention_response(
            "I could not verify the answer's supporting quotations. Please narrow the question."
        )
    partial = (
        draft.status == "partial"
        or bool(missing)
        or bool(verification.get("rejected_count"))
    )
    # Unanswered tasks stay in missing_tasks; coverage labels are too unreliable to print.
    text = "\n\n".join(f"{claim.text} [{claim.source_id}]" for claim in quoted_claims)
    used = {claim.source_id for claim in quoted_claims}
    return {
        "kind": "answer",
        "status": "partial" if partial else "answered",
        "answer": text,
        "claims": [claim.model_dump() for claim in quoted_claims],
        "sources": build_source_citations(
            [row for row in evidence if row["id"] in used]
        ),
        "missing_tasks": missing,
        # Lexical conflict hints stay available to reviewers, outside the answer text.
        "consistency_notes": (consistency_report or {}).get("potential_conflicts", []),
        "verification": "exact_quote_and_number_check",
    }
