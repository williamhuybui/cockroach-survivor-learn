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

def claim_is_relevant(claim_text: str, question: str) -> bool:
    """Basic relevance guard for explicitly requested metrics."""
    text = claim_text.casefold()
    query = question.casefold()

    metrics = (
        "bleu", "accuracy", "precision", "recall",
        "f1", "perplexity", "latency"
    )

    requested = [metric for metric in metrics if metric in query]

    if requested and not any(metric in text for metric in requested):
        return False

    return True

def quote_key(text: str) -> str:
    """Compare quotes case-insensitively and rejoin words split by PDF line-end hyphens."""
    return re.sub(r"(?<=[a-z])- ?(?=[a-z])", "", normalized(text).casefold())


STOPWORDS = {
    "that", "this", "with", "from", "were", "have", "which", "their", "they", "than",
    "into", "also", "such", "each", "used", "using", "paper", "model", "models",
    "the", "and", "for", "are", "was", "its", "our", "has", "been", "what", "does",
}
# A paraphrase dropping these words can reverse the quoted finding.
LIMITING_WORDS = re.compile(
    r"\b(minimal|negligible|marginal|insignificant|little effect|no effect|"
    r"not significant|no significant|does not depend|weakly)\b"
)
QUOTE_REASONS = {"quote_not_found", "quote_truncated", "unsupported_number", "quote_unrelated"}


def content_words(text: str) -> set[str]:
    """Distinctive words or numbers used to check that a quote is about its claim."""
    return {
        word
        for word in re.findall(r"[a-z0-9]+(?:[.-][a-z0-9]+)*", quote_key(text))
        if (len(word) >= 4 or any(ch.isdigit() for ch in word)) and word not in STOPWORDS
    }


def identifiers(text: str) -> set[str]:
    """Versioned names in a question, e.g. 2019, 8x7B, 405B, GPT-3."""
    return {
        quote_key(token)
        for token in re.findall(r"[A-Za-z0-9]+(?:[.-][A-Za-z0-9]+)*", text)
        if any(ch.isdigit() for ch in token) and any(ch.isalpha() for ch in token)
        or re.fullmatch(r"(19|20)\d\d", token)
    }


def numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text))


DIFFERENCE = r"(?:by|more than|over|up to|an? (?:gain|improvement) of)\s+(?:about\s+)?"


def differences_as_values(claim_text: str, quote: str) -> bool:
    """'outperforms by more than 2.0 BLEU' must not become 'a BLEU score of 2.0'."""
    for value in numbers(claim_text):
        quoted_as_difference = re.search(DIFFERENCE + re.escape(value) + r"\b", quote, re.IGNORECASE)
        stated_as_difference = re.search(DIFFERENCE + re.escape(value) + r"\b", claim_text, re.IGNORECASE)
        if quoted_as_difference and not stated_as_difference and value not in re.sub(
            DIFFERENCE + re.escape(value), "", quote, flags=re.IGNORECASE
        ):
            return True
    return False


def prefer_headline_values(claims: list, sources: dict) -> list:
    """Drop a body-text claim that an abstract claim from the same paper restates
    with a different number; papers update headline numbers before body text."""
    def shape(claim):
        return content_words(re.sub(r"\d+(?:\.\d+)?", " ", claim.text))

    def names(claim):
        # Models, metrics and datasets ("Transformer", "BLEU", "English-to-French").
        return {
            token.casefold()
            for token in re.findall(r"[\w-]*[A-Z][\w-]*", claim.text)
            if not re.fullmatch(r"[\d.]+", token)
        }

    def in_abstract(claim):
        return "abstract" in sources.get(claim.source_id, {}).get("location", "").casefold()

    dropped = set()
    for i, left in enumerate(claims):
        for j, right in enumerate(claims):
            if i == j or not in_abstract(left) or in_abstract(right):
                continue
            same_paper = sources.get(left.source_id, {}).get("title") == sources.get(
                right.source_id, {}
            ).get("title")
            a, b = shape(left), shape(right)
            similar = (
                a and b and names(left) == names(right) and len(a & b) / len(a | b) >= 0.6
            )
            if same_paper and similar and numbers(left.text) != numbers(right.text):
                dropped.add(j)
    return [claim for k, claim in enumerate(claims) if k not in dropped]


def supporting_sentence(claim_text: str, content: str, question: str = "") -> str | None:
    """Best sentence of the cited passage holding every claim number and most claim words."""
    claim_words = content_words(claim_text)
    needed = max(2, (len(claim_words) + 1) // 2)
    best, best_overlap = None, 0
    for sentence in re.split(r"(?<=[.!?])\s+", normalized(content)):
        if len(sentence) < 20 or not numbers(claim_text) <= numbers(sentence) | numbers(question):
            continue
        overlap = len(claim_words & content_words(sentence))
        if overlap >= needed and overlap > best_overlap:
            best, best_overlap = sentence[:1500], overlap
    return best


def quote_problem(quote: str, passage: str) -> str | None:
    """Exact span check, also rejecting quotes cut inside a word."""
    key = quote_key(quote)
    position = passage.find(key)
    if position < 0:
        return "quote_not_found"
    before = passage[position - 1] if position else " "
    after = passage[position + len(key)] if position + len(key) < len(passage) else " "
    if (before.isalnum() and key[0].isalnum()) or (after.isalnum() and key[-1].isalnum()):
        return "quote_truncated"
    return None


def verify_claims(
    draft: DraftAnswer,
    evidence: list[dict],
    tasks: list[dict],
    question: str = "",
) -> dict:
    """Check all claims once and track declared task coverage, not semantic entailment."""
    passages = {row["id"]: quote_key(row["content"]) for row in evidence}
    sources = {row["id"]: row for row in evidence}
    corpus = " ".join(passages.values())
    missing = {name for name in identifiers(question) if name not in corpus}
    # A task naming a year, version or size (2019, 8x7B) needs a passage that mentions it.
    subjects = [identifiers(task["question"]) for task in tasks]
    # Words unique to one unscoped task let a claim cover it even if mislabelled.
    task_words = [content_words(task["question"]) for task in tasks]
    shared = set.intersection(*task_words) if len(tasks) > 1 else set()
    accepted, rejected, covered, seen_quotes = [], [], set(), set()

    def problems(claim) -> list[str]:
        reasons = []
        if claim.source_id not in passages:
            reasons.append("unknown_source")
        else:
            problem = quote_problem(claim.quote, passages[claim.source_id])
            if problem:
                reasons.append(problem)
        if re.search(r"\bS\d+\b", claim.text):
            reasons.append("source_id_in_text")
        if not numbers(claim.text) <= numbers(claim.quote) | numbers(question):
            reasons.append("unsupported_number")
        elif differences_as_values(claim.text, claim.quote):
            reasons.append("difference_stated_as_value")
        claim_words = content_words(claim.text)
        if claim_words and not claim_words & content_words(claim.quote):
            reasons.append("quote_unrelated")
        if question and not claim_is_relevant(claim.text, question):
            reasons.append("irrelevant_metric")
        return reasons

    for claim in draft.claims:
        reasons = problems(claim)
        if reasons and set(reasons) <= QUOTE_REASONS and claim.source_id in sources:
            # The model often cites the right paper with the wrong sentence or passage;
            # search the cited passage first, then other passages of the same paper.
            title = sources[claim.source_id].get("title")
            for row in sorted(evidence, key=lambda row: row["id"] != claim.source_id):
                if row.get("title") != title:
                    continue
                sentence = supporting_sentence(claim.text, row["content"], question)
                if sentence:
                    claim = claim.model_copy(update={"quote": sentence, "source_id": row["id"]})
                    reasons = problems(claim)
                    break
        source = sources.get(claim.source_id, {})
        retrieved_for = set(source.get("task_ids", range(len(tasks))))
        ids = {i for i in claim.task_ids if 0 <= i < len(tasks)} or (
            retrieved_for & set(range(len(tasks)))
        )
        # Paper-scoped tasks need that paper's passage; unscoped tasks may share passages.
        if not ids or any(
            tasks[i].get("paper") and source.get("title") != tasks[i]["paper"] for i in ids
        ):
            reasons.append("task_mismatch")
        passage = passages.get(claim.source_id, "")
        if any(subjects[i] and not any(name in passage for name in subjects[i]) for i in ids):
            reasons.append("requested_setting_missing")
        if not reasons and (claim.source_id, quote_key(claim.quote)) in seen_quotes:
            reasons.append("duplicate_quote")
        if reasons:
            rejected.append({"claim": claim.text, "source_id": claim.source_id, "reasons": reasons})
            continue
        seen_quotes.add((claim.source_id, quote_key(claim.quote)))
        if LIMITING_WORDS.search(quote_key(claim.quote)) and not LIMITING_WORDS.search(
            claim.text.casefold()
        ):
            # Show the quoted wording rather than a paraphrase that may invert it.
            claim = claim.model_copy(update={"text": normalized(claim.quote)})
        accepted.append(claim)
        covered.update(ids)
        words = content_words(claim.text)
        covered.update(
            i
            for i, task in enumerate(tasks)
            if not task.get("paper") and len((task_words[i] - shared) & words) >= 2
        )
    accepted = prefer_headline_values(accepted, sources)
    return {
        "accepted": accepted,
        "rejected_count": len(rejected),
        "rejection_details": rejected,
        "missing_tasks": [i for i in range(len(tasks)) if i not in covered],
        "missing_identifiers": sorted(missing),
    }


METRIC_TERMS = (
    "bleu", "rouge", "accuracy", "precision", "recall", "f1",
    "perplexity", "exact match", "win rate", "pass@1",
)


def needs_numerical_consistency(question: str) -> bool:
    """Only questions naming an evaluation metric need the cross-source value check."""
    query = question.casefold()
    return any(re.search(rf"\b{re.escape(term)}", query) for term in METRIC_TERMS)


def metric_values(text: str, term: str) -> set[str]:
    """Decimal values written close to a metric name, e.g. 'BLEU score of 41.0'."""
    values = set()
    for match in re.finditer(rf"\b{re.escape(term)}", text, re.IGNORECASE):
        window = text[max(0, match.start() - 30) : match.end() + 60]
        values.update(re.findall(r"\d+\.\d+", window))
    return values


def analyze_evidence_consistency(question, evidence):
    """Flag sources that report different values for a requested metric, without a model call.

    This is a lexical hint for the generator, not proof that settings are comparable.
    """
    terms = [
        term for term in METRIC_TERMS if re.search(rf"\b{re.escape(term)}", question.casefold())
    ]
    potential = []
    for term in terms:
        reported = [
            (row, metric_values(normalized(row["content"]), term)) for row in evidence
        ]
        reported = [(row, values) for row, values in reported if values]
        for i, (left, left_values) in enumerate(reported):
            for right, right_values in reported[i + 1 :]:
                only_left, only_right = left_values - right_values, right_values - left_values
                # Many differing numbers usually means a table of different settings.
                if not only_left or not only_right or len(only_left | only_right) > 4:
                    continue
                potential.append(
                    {
                        "metric": term,
                        "source_1": left["id"],
                        "value_1": ", ".join(sorted(only_left)),
                        "location_1": left.get("location", ""),
                        "source_2": right["id"],
                        "value_2": ", ".join(sorted(only_right)),
                        "location_2": right.get("location", ""),
                        "status": "requires_context_review",
                    }
                )
    return {"potential_conflicts": potential}


def draft_answer(question, history, evidence, tasks):
    """Generate an evidence-grounded answer with one model call."""
    request = {
        "question": question,
        "recent_conversation": history,
        "tasks": tasks,
    }
    consistency_report = (
        analyze_evidence_consistency(question, evidence)
        if needs_numerical_consistency(question)
        else None
    )
    if consistency_report:
        request["evidence_consistency"] = consistency_report

    instruction = """
You are answering questions about research papers using only the supplied evidence.

GROUNDING
- Do not use outside knowledge or invent facts, numbers, source IDs, or quotes.
- Answer exactly what is asked; do not add background, training details,
  or results that were not requested.
- Match the requested model, paper, dataset, year, and setting exactly.
  If the evidence covers a different one (for example another year,
  dataset, or model), do not substitute it: report that part as missing.
- When the question asks for several items (models, datasets, settings),
  name every one exactly as written in the evidence.
- When passages report different values for the same result, give the
  value from the abstract or results table and mention the other value.

CLAIMS
Return a structured DraftAnswer with at most 4 claims. Each claim has:
- text: one direct factual statement answering the question, with no source IDs.
- source_id: an existing evidence source ID.
- quote: the shortest exact continuous substring of that source (under
  30 words) that supports the claim, copied character for character.
- task_ids: indices of the request.tasks the claim answers.
Every number in a claim must appear in its quote or in the question.
Cover every task in request.tasks.
Review request.evidence_consistency when it is present.

STATUS
- answered: every task is supported by the evidence.
- partial: some tasks are supported; list the rest in missing_information.
- abstained: the evidence does not answer the question; return no claims.

Return only structured DraftAnswer.
"""
    draft = generate(DraftAnswer, instruction, evidence, request)
    return draft, consistency_report

NOT_IN_DOCUMENTS = "I don't know - this is not in the documents."


def format_answer(
    draft,
    verification,
    evidence,
    tasks,
    consistency_report=None,
):
    """Format verified claims; unverified parts are named briefly, not as boilerplate."""
    quoted_claims = verification["accepted"]
    if draft.status == "abstained" or not draft.claims or not quoted_claims:
        return abstention(NOT_IN_DOCUMENTS)

    missing = [tasks[i]["question"] for i in verification["missing_tasks"]]
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
        "sources": source_cards([row for row in evidence if row["id"] in used]),
        "missing_tasks": missing,
        # Lexical conflict hints stay available to reviewers, outside the answer text.
        "consistency_notes": (consistency_report or {}).get("potential_conflicts", []),
        "verification": "exact_quote_and_number_check",
    }
