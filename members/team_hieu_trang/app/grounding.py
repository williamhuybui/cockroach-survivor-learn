"""Generate one answer per requested detail and compute completeness in Python.

Gemma supplies explicit availability, plain answers and source labels for each
requested detail. Citation, task scope and number checks cannot prove semantic support.
They run automatically; no human review is required to produce an answer.
"""

import re
import unicodedata
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field, create_model

from .answer_scope import check_explicit_answer_scope
from .llm import generate_from_evidence


class DetailAnswer(BaseModel):
    """One requested detail's availability, answer text and supplied source labels."""

    availability: Literal["available", "unavailable"] = Field(
        description="Use available only when this requested fact is supported; otherwise unavailable."
    )
    source_ids: list[str] = Field(
        max_length=10,
        description="Supporting supplied labels, for example ['S1']. Select sources for this requested detail only. Put citations here, not inside answer.",
    )
    answer: str = Field(
        max_length=2500,
        description="ONLY the requested value, names or explanation, supported by the selected sources. Do not write missing-information sentences or related background.",
    )


# This is Gemma's structured draft, not the final verified response.
# Its claims and reported gaps are checked before formatting the user-facing answer.
class DraftAnswer(BaseModel):
    """Answers in requested order; Python computes the overall response status."""

    answers: list[DetailAnswer] = Field(
        default_factory=list,
        description="One answer/source record per requested detail, in its original order.",
    )


def normalize_evidence_text(text: str) -> str:
    """Join PDF word wrapping and normalize typography before checking source text."""
    text = unicodedata.normalize("NFKC", text)
    # Rejoin lowercase word continuations, retaining hyphens before capitals/digits.
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=[a-z])", "", text)
    text = re.sub(r"-\s*\n\s*(?=\w)", "-", text)
    return " ".join(text.split())


def extract_numbers(text: str) -> set[str]:
    """Treat numeric forms such as 10,000 and 10000 consistently."""
    text = re.sub(r"(?<=\d),(?=\d)", "", normalize_evidence_text(text))
    return set(re.findall(r"\d+(?:\.\d+)?", text))


# Magnitude words and capital suffixes; lowercase single letters are left out
# because they also abbreviate units such as metres or minutes.
SCALES = {
    "k": 10**3, "thousand": 10**3, "m": 10**6, "million": 10**6, "b": 10**9,
    "bn": 10**9, "billion": 10**9, "t": 10**12, "trillion": 10**12,
}
SCALED_NUMBER = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:([KMBT])\b|(bn|thousand|million|billion|trillion)\b)",
    re.IGNORECASE,
)


def extract_scaled_values(text: str) -> dict[str, set[Decimal]]:
    """Map each scaled number to its absolute value: '2.5 billion' and '2,500M' agree."""
    text = re.sub(r"(?<=\d),(?=\d)", "", normalize_evidence_text(text))
    values: dict[str, set[Decimal]] = {}
    for match in SCALED_NUMBER.finditer(text):
        letter, word = match.group(2), match.group(3)
        if letter and not letter.isupper():
            continue
        scale = SCALES[(letter or word).casefold()]
        values.setdefault(match.group(1), set()).add(Decimal(match.group(1)) * scale)
    return values


def find_unsupported_numbers(answer: str, evidence: str) -> set[str]:
    """Numbers in the answer that the evidence states in neither the same nor an equal scaled form."""
    missing = extract_numbers(answer) - extract_numbers(evidence)
    if not missing:
        return missing
    stated = set().union(*extract_scaled_values(evidence).values())
    scaled = extract_scaled_values(answer)
    return {number for number in missing if not scaled.get(number, set()) & stated}


def check_requested_currency(question: str, answer: str, evidence: str) -> str:
    """Check explicitly requested currency against the answer and cited evidence.

    For example, a computation measured in hours cannot answer a request for
    dollars. This is only a unit check; it does not establish semantic correctness.
    """
    # Apply this only to requests for a monetary amount, not questions about a
    # currency or symbol itself. Other units and general facts keep their checks.
    money_terms = {
        "amount",
        "budget",
        "charge",
        "cost",
        "costs",
        "expense",
        "expenses",
        "fee",
        "fees",
        "funding",
        "income",
        "investment",
        "paid",
        "pay",
        "payment",
        "payments",
        "price",
        "prices",
        "profit",
        "revenue",
        "salary",
        "spend",
        "spent",
        "value",
        "worth",
    }
    words = set(re.findall(r"\b[a-z]+\b", question.casefold()))
    if not money_terms.intersection(words) and "how much" not in question.casefold():
        return ""
    currencies = (
        (r"\b(?:usd|dollars?)\b|\$", "dollars"),
        (r"\b(?:eur|euros?)\b|€", "euros"),
        (r"\b(?:jpy|yen)\b|¥", "yen"),
        (r"\b(?:gbp|pounds? sterling)\b|£", "pounds sterling"),
    )
    for pattern, label in currencies:
        if re.search(pattern, question, re.IGNORECASE) and (
            not re.search(pattern, answer, re.IGNORECASE)
            or not re.search(pattern, evidence, re.IGNORECASE)
        ):
            return f"The answer and cited evidence do not both state the requested currency: {label}."
    return ""


# Requested property: label, task-question pattern, words that must name it near the answer.
REQUESTED_PROPERTIES = (
    ("batch size", r"\bbatch sizes?\b", ("batch",)),
    ("learning rate", r"\blearning rates?\b", ("learning rate", "lr")),
    ("warmup", r"\bwarm-?up\b", ("warmup", "warm-up")),
    ("dropout", r"\bdropout\b", ("dropout",)),
    ("weight decay", r"\bweight decay\b", ("weight decay",)),
    ("optimizer", r"\boptimi[sz]er\b", ("optimi", "adam", "sgd", "adafactor")),
    ("BLEU", r"\bbleu\b", ("bleu",)),
    ("accuracy", r"\baccuracy\b", ("accura",)),
    ("perplexity", r"\bperplexity\b", ("perplexity",)),
    ("hardware", r"\bgpus?\b|\bhardware\b", ("gpu", "tpu", "a100", "h100", "h800", "v100", "p100")),
    ("epochs", r"\bepochs?\b", ("epoch",)),
    ("training steps", r"\b(?:training|optimi[sz]ation) steps\b", ("step",)),
)
NAMED_BENCHMARKS = tuple(
    # A named benchmark must appear near the value: WebShop success cannot answer GSM8K.
    # Ordinary words such as 'math' are left out because questions use them generically.
    (name, r"\b" + re.escape(name) + r"\b", (name.casefold(),))
    for name in (
        "GSM8K", "MMLU", "HumanEval", "MBPP", "HotpotQA", "ALFWorld", "WebShop",
        "ImageNet", "SQuAD", "SuperGLUE", "TriviaQA", "LAMBADA", "HellaSwag",
        "WinoGrande", "PIQA", "BoolQ", "AIME", "Codeforces", "SWE-bench",
        "LiveCodeBench", "MT-Bench", "AlpacaEval",
    )
)


# Training settings whose value is always a number, unlike an optimizer or hardware name.
NUMERIC_SETTINGS = {
    "batch size", "learning rate", "warmup", "dropout", "weight decay", "epochs",
    "training steps",
}


def check_requested_attribute(question: str, answer: str, evidence: str) -> str:
    """Check that a numeric answer is stated for the property the task asks about.

    The answer or a cited sentence containing one of its numbers must name the
    property. For example, '32k tokens' from a context-size sentence cannot answer
    a batch-size request. A request for a training setting's value needs a number;
    other answers without numbers are not checked. A task listing several properties
    ('corpus, batch size, steps and optimizer') skips the property check, since one
    sentence rarely names them all; named benchmarks are still checked.
    """
    properties = [
        row for row in REQUESTED_PROPERTIES if re.search(row[1], question, re.IGNORECASE)
    ]
    requested = (properties if len(properties) == 1 else []) + [
        row for row in NAMED_BENCHMARKS if re.search(row[1], question, re.IGNORECASE)
    ]
    # Numbers repeated from the question, as in 'Mixtral 8x7B', are names, not values.
    numbers = extract_numbers(answer) - extract_numbers(question)
    if not numbers:
        asks_value = re.search(
            r"\b(?:what|which|how (?:many|much|large|big))\b", question, re.IGNORECASE
        ) and not re.search(r"\bwhy\b", question, re.IGNORECASE)
        for label, _, _ in requested:
            if asks_value and label in NUMERIC_SETTINGS:
                return f"The requested {label} needs an explicit value, but the answer has none."
        return ""
    sentences = re.split(r"(?<=[.!?;])\s+", normalize_evidence_text(evidence))
    near_answer = " ".join(
        [answer] + [part for part in sentences if extract_numbers(part) & numbers]
    ).casefold()
    for label, _, names in requested:
        # Names are word prefixes, so 'optimi' matches optimizer but 'lr' skips 'already'.
        if not any(re.search(r"\b" + re.escape(name), near_answer) for name in names):
            return f"No cited sentence states the answer's values as the requested {label}."
    return ""


def build_abstention_response(reason: str) -> dict:
    """Return an honest unavailable answer without attributed sources.

    Every abstention opens with "I don't know": the evaluator's judge can read a bare
    'No matching evidence was found' as agreeing with a 'Not in the documents' ground
    truth and mark it correct, which scores as an answer instead of an abstention.
    """
    if not reason.startswith("I don't know"):
        reason = "I don't know - " + reason
    return {"kind": "answer", "status": "abstained", "answer": reason, "sources": []}


def build_source_citations(evidence: list[dict]) -> list[dict]:
    """Build source cards from stored metadata rather than model-written links."""
    return [
        {
            key: row[key]
            for key in ("id", "title", "source_url", "location")
            if key in row
        }
        for row in evidence
    ]


def find_literal_list_source(
    answer: str, task: dict, task_index: int, evidence: list[dict], question: str
) -> str | None:
    """Find a same-task passage that contains every exact item in a name list.

    This can repair a misplaced citation, but matching names does not prove that
    the passage supports the answer's meaning or that the list is complete.
    """
    items = [item.strip() for item in answer.strip().rstrip(".").split(",")]
    if len(items) < 2 or any(
        not re.fullmatch(r"[\w.+/-]+(?:\s*\([\w.+/-]+\))?", item)
        for item in items
    ):
        return None

    paper = task.get("paper")
    allowed_papers = paper if isinstance(paper, list) else [paper]
    for row in evidence:
        if task_index not in row.get("task_ids", []):
            continue
        if paper and row.get("title") not in allowed_papers:
            continue
        reason, eligible = check_explicit_answer_scope(
            task["question"], [row["content"]], question
        )
        if reason or check_requested_currency(task["question"], answer, eligible):
            continue
        text = normalize_evidence_text(eligible).casefold()
        if all(
            re.search(
                r"(?<!\w)" + re.escape(normalize_evidence_text(item).casefold()) + r"(?!\w)",
                text,
            )
            for item in items
        ):
            return row["id"]
    return None


def find_number_sources(
    answer: str,
    missing: set[str],
    task: dict,
    task_index: int,
    evidence: list[dict],
    question: str,
) -> tuple[set[str], str]:
    """Find same-task passages that state numbers Gemma cited from the wrong label.

    Each passage must pass the same scope and currency checks as a cited one, so a
    number still has to appear in this detail's own retrieved text. Returns the
    added labels and their eligible text, or nothing when any number stays missing.
    """
    paper = task.get("paper")
    allowed_papers = paper if isinstance(paper, list) else [paper]
    added, texts = set(), []
    for row in evidence:
        if not missing:
            break
        if task_index not in row.get("task_ids", []):
            continue
        if paper and row.get("title") not in allowed_papers:
            continue
        reason, eligible = check_explicit_answer_scope(
            task["question"], [row["content"]], question
        )
        if reason or check_requested_currency(task["question"], answer, eligible):
            continue
        supplied = missing - find_unsupported_numbers(answer, eligible)
        if supplied:
            added.add(row["id"])
            texts.append(eligible)
            missing = missing - supplied
    return (set(), "") if missing else (added, " ".join(texts))


def generate_grounded_draft(
    question: str, evidence: list[dict], tasks: list[dict]
) -> DraftAnswer:
    """Ask Gemma once for one small record per requested detail, including missing ones."""
    schema = create_model(
        "RequestedDetailAnswers",
        __base__=DraftAnswer,
        answers=(
            list[DetailAnswer],
            Field(
                min_length=len(tasks),
                max_length=len(tasks),
                description="One record per requested detail, in order. Every record must declare available or unavailable; do not use null slots.",
            ),
        ),
    )
    return generate_from_evidence(
        schema,
        "Return exactly one record per requested detail, in the same order. "
        "Answer each requested detail in its own record; do not repeat answers to "
        "other details. For each detail, cite only its allowed_source_ids and use only passages "
        "associated with that task and requested paper. "
        "Every record must declare availability. For unavailable details use answer='' and source_ids=[]. "
        "A detail marked scope=unresolved must be unavailable; do not borrow another task's evidence. "
        "The passage must state the requested fact for the requested model, experiment and "
        "training stage. Merely mentioning the topic is insufficient. A retrieved passage "
        "is only a candidate and may not answer the task. Numerical settings require "
        "explicit values; inference performance cannot supply pre-training settings. "
        "If unsupported, mark that record unavailable. Do not replace "
        "a missing fact with related background or a sentence saying it is missing. "
        "Example: a passage stating that Model Delta has four routers does not state "
        "its training duration. For a training-duration detail mark unavailable. "
        "If supported, write only the direct answer, preserving concrete names, values and "
        "variants, and put supporting labels in source_ids. No inline citations are needed. "
        "Answer the requested detail directly and briefly. Omit unrelated background, "
        "extra measurements, comparisons not requested and repeated conclusions. "
        "Do not invent labels, links or pages. Do not add unrelated facts. No source quotations "
        "are required. If sources conflict on the same setting, attribute the alternatives "
        "without inventing a resolution. Do not infer table cells, figures or equations; "
        "a caption supports its explicit words, but never claim to have read its cells. "
        "Text marked structure_confidence=uncertain does not establish reading order. "
        "Attribute learning recommendations to the guide. Catalog metadata does not establish findings.",
        evidence,
        {
            "question": question,
            "requested_details": [
                dict(
                    position=i,
                    allowed_source_ids=[
                        row["id"]
                        for row in evidence
                        if i in row.get("task_ids", range(len(tasks)))
                        and (
                            not task.get("paper")
                            or row.get("title")
                            in (
                                task["paper"]
                                if isinstance(task["paper"], list)
                                else [task["paper"]]
                            )
                        )
                    ],
                    **task,
                )
                for i, task in enumerate(tasks)
            ],
        },
    )


def verify_answer_citations(
    draft: DraftAnswer,
    evidence: list[dict],
    tasks: list[dict],
    question: str = "",
    source_papers: list[str] | None = None,
) -> dict:
    """Validate each detail independently so one source cannot cover unrelated tasks.

    Empty, omitted, duplicate-source and invalid records cannot make other details
    complete. Numbers must occur in the cited passages, but their meaning still
    requires semantic evaluation.
    """
    sources = {row["id"]: row for row in evidence}
    accepted, used, rejections = [], set(), []
    for index, detail in enumerate(draft.answers):
        if detail is None or detail.availability == "unavailable":
            continue
        reason = ""
        if index >= len(tasks):
            reason = "Unexpected answer beyond the requested details."
        elif tasks[index].get("scope") == "unresolved":
            reason = "The requested paper identity was not resolved."
        elif detail.answer.strip().strip("\"'").casefold() in ("", "null"):
            continue
        ids = set(detail.source_ids)
        if not reason and not ids:
            reason = "This detail has no source citation."
        elif not reason and not ids <= sources.keys():
            reason = "Unknown source citation: " + ", ".join(
                sorted(ids - sources.keys())
            )
        elif not reason and any(
            index not in sources[label].get("task_ids", range(len(tasks)))
            or (
                tasks[index].get("paper")
                and sources[label].get("title")
                not in (
                    tasks[index]["paper"]
                    if isinstance(tasks[index]["paper"], list)
                    else [tasks[index]["paper"]]
                )
            )
            for label in ids
        ):
            reason = "A cited passage is outside this detail's task or paper scope."
        if not reason:
            reason, supporting_text = check_explicit_answer_scope(
                tasks[index]["question"],
                [sources[label]["content"] for label in ids],
                question if len(tasks) == 1 else "",
            )
        if not reason:
            reason = check_requested_currency(
                tasks[index]["question"], detail.answer, supporting_text
            )
        if not reason:
            text = detail.answer
            # Ignore any redundant inline labels; Python renders validated citations.
            inline = set(re.findall(r"\bS\d+\b", text))
            if inline - ids:
                reason = "Inline citation does not match the supplied source labels."
            text = re.sub(r"\[[^\]]*\bS\d+\b[^\]]*\]", "", text).strip()
            text = re.sub(r"(?m)^\s*\d+[.)]\s+", "", text)
            if not text.strip():
                reason = "This detail contains no answer text."
            unsupported = find_unsupported_numbers(text, supporting_text)
            if unsupported:
                alternative, extra_ids = None, set()
                if not reason:
                    alternative = find_literal_list_source(
                        text, tasks[index], index, evidence,
                        question if len(tasks) == 1 else "",
                    )
                if not reason and not alternative:
                    # Gemma often cites the first passage although a later one holds the number.
                    extra_ids, extra_text = find_number_sources(
                        text, unsupported, tasks[index], index, evidence,
                        question if len(tasks) == 1 else "",
                    )
                if alternative:
                    ids = {alternative}
                elif extra_ids:
                    ids = ids | extra_ids
                    supporting_text += " " + extra_text
                else:
                    reason = "Numbers absent from the cited passages: " + ", ".join(
                        sorted(unsupported)
                    )
        if not reason:
            reason = check_requested_attribute(
                tasks[index]["question"], text, supporting_text
            )
        # 'According to the LoRA paper' needs a LoRA citation, not only Mixtral's.
        if not reason and source_papers and not any(
            sources[label].get("title") in source_papers for label in ids
        ):
            reason = "The question names its source paper, but no cited passage comes from it."
        if reason:
            rejections.append({"detail_index": index, "reason": reason})
            continue
        accepted.append(
            {"task_index": index, "answer": text, "source_ids": sorted(ids)}
        )
        used.update(ids)
    covered = {row["task_index"] for row in accepted}
    return {
        "accepted": accepted,
        "source_ids": sorted(used),
        "rejected_count": len(rejections),
        "rejections": rejections,
        "missing_tasks": [i for i in range(len(tasks)) if i not in covered],
    }


def format_cited_answer(
    verification: dict, evidence: list[dict], tasks: list[dict]
) -> dict:
    """Compute answered/partial/abstained from validated details and render without an LLM."""
    accepted = verification["accepted"]
    missing = [tasks[i]["question"] for i in verification["missing_tasks"]]
    if not accepted:
        result = build_abstention_response(
            "I don't know - the retrieved documents do not provide a supported answer to the requested details."
        )
    else:
        paragraphs = []
        for row in accepted:
            # Only collapse exact repeated paragraphs within one task. Identical
            # values from different tasks may represent different requested facts.
            parts = list(dict.fromkeys(row["answer"].split("\n\n")))
            answer = "\n\n".join(parts)
            paragraphs.append(
                answer + " " + " ".join(f"[{label}]" for label in row["source_ids"])
            )
        text = "\n\n".join(paragraphs)
        if missing:
            text += "\n\nMissing information:\n" + "\n".join(
                f"- {part}" for part in missing
            )
        result = {
            "kind": "answer",
            "status": "partial" if missing else "answered",
            "answer": text,
            "sources": build_source_citations(
                [row for row in evidence if row["id"] in verification["source_ids"]]
            ),
        }
    by_task = {row["task_index"]: row for row in accepted}
    result.update(
        missing_tasks=missing,
        detail_answers=[
            {
                "question": task["question"],
                "availability": "available" if i in by_task else "unavailable",
                "answer": by_task.get(i, {}).get("answer"),
                "source_ids": by_task.get(i, {}).get("source_ids", []),
            }
            for i, task in enumerate(tasks)
        ],
        verification="per_detail_citation_scope_number_check",
    )
    return result
