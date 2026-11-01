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
from .stage_scope import find_stage_mismatch, find_stage_repair, numbers_in, requested_stage


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


# The answer itself says the requested fact is missing ("The passage does not specify the GPUs",
# "The GPU type was unspecified"). The prompt asks Gemma to mark such a detail unavailable instead,
# so an answer containing this did not find the fact and usually fills the gap with something else.
MISSING_FACT_STATEMENT = re.compile(
    r"\bunspecified\b"
    r"|\b(?:is|are|was|were)\s+not\s+(?:explicitly\s+)?(?:specified|mentioned|stated|reported)\b"
    r"|\bdoes\s+not\s+(?:specify|state|mention|report)\b"
    r"|\bno\s+information\b",
    re.IGNORECASE,
)


def normalize_evidence_text(text: str) -> str:
    """Join PDF word wrapping and normalize typography before checking source text."""
    text = unicodedata.normalize("NFKC", text)
    # Rejoin lowercase word continuations, retaining hyphens before capitals/digits.
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=[a-z])", "", text)
    text = re.sub(r"-\s*\n\s*(?=\w)", "-", text)
    return " ".join(text.split())


# A thousands separator between digits, so 10,000 reads as 10000.
THOUSANDS_COMMA = re.compile(r"(?<=\d),(?=\d)")
# Where one evidence sentence ends and the next begins.
SENTENCE_END = re.compile(r"(?<=[.!?;])\s+")


def extract_numbers(text: str) -> set[str]:
    """Treat numeric forms such as 10,000 and 10000 consistently."""
    text = THOUSANDS_COMMA.sub("", normalize_evidence_text(text))
    return set(re.findall(r"\d+(?:\.\d+)?", text))


# Papers often spell small numbers out: "eight P100 GPUs", "two experts".
NUMBER_WORDS = {
    word: str(value)
    for value, word in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
    )
} | {
    word: str(value)
    for value, word in zip(range(30, 100, 10), "thirty forty fifty sixty seventy eighty ninety".split())
} | {"single": "1", "once": "1", "twice": "2"}


def extract_spelled_values(text: str) -> set[str]:
    """Values written as words, such as 'eight' or 'a single epoch', as digit strings."""
    words = re.findall(r"[a-z]+", normalize_evidence_text(text).casefold())
    return {NUMBER_WORDS[word] for word in words if word in NUMBER_WORDS}


def extract_evidence_numbers(text: str) -> set[str]:
    """Numbers a source passage states, as digits or as words such as 'eight'.

    Only source text is read this way, so an answer's ordinary words such as
    'one of the models' never become values that need support.
    """
    return extract_numbers(text) | extract_spelled_values(text)


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
    text = THOUSANDS_COMMA.sub("", normalize_evidence_text(text))
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
    missing = extract_numbers(answer) - extract_evidence_numbers(evidence)
    if not missing:
        return missing
    stated = set().union(*extract_scaled_values(evidence).values())
    scaled = extract_scaled_values(answer)
    # A written-out "180,000" equals the passage's "180K".
    return {number for number in missing if not (scaled.get(number, set()) | {Decimal(number)}) & stated}


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


# A request for "GPU hours" names a unit; FLOPs or days on a cluster do not answer it.
COMPUTE_UNIT = re.compile(r"\b(gpu|tpu)[\s-]+(hour|day)s?\b", re.IGNORECASE)


def check_requested_compute_unit(question: str, answer: str, evidence: str) -> str:
    """Check that a request for GPU or TPU hours (or days) is answered in that unit.

    Papers also report training compute as petaflop/s-days or as days on a cluster,
    which cannot answer a request for GPU hours. Like the currency check, this is a
    unit check only; it does not establish semantic correctness.
    """
    asked = COMPUTE_UNIT.search(question)
    if not asked:
        return ""
    unit = re.compile(rf"\b{asked[1]}[\s-]+{asked[2]}s?\b", re.IGNORECASE)
    if unit.search(answer) and unit.search(evidence):
        return ""
    return f"The answer and cited evidence do not both state the requested unit: {asked[1].upper()} {asked[2].lower()}s."


def check_requested_unit(question: str, answer: str, evidence: str) -> str:
    """Return why the answer lacks an explicitly requested currency or compute unit, if it does."""
    return check_requested_currency(question, answer, evidence) or check_requested_compute_unit(
        question, answer, evidence
    )


# Words that name hardware. 'NVIDIA Quadro RTX8000' names a GPU without the word 'GPU'.
HARDWARE_NAMES = (
    "gpu", "tpu", "cpu", "a100", "h100", "h800", "h20", "v100", "p100", "k80", "rtx", "quadro",
    "titan", "tesla", "nvidia", "accelerator", "chip",
)

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
    ("hardware", r"\bgpus?\b|\bhardware\b", HARDWARE_NAMES),
    ("epochs", r"\bepochs?\b", ("epoch",)),
    ("training steps", r"\b(?:training|optimi[sz]ation) steps\b", ("step",)),
    # Architecture and decoding settings: a 20,480-token generation limit is not a vocabulary size.
    ("vocabulary", r"\bvocab(?:ulary)?\b", ("vocab",)),
    ("context length", r"\bcontext (?:length|window|size)s?\b", ("context",)),
    ("layers", r"\blayers\b", ("layer",)),
    ("attention heads", r"\bheads\b", ("head",)),
    ("temperature", r"\btemperature\b", ("temperature",)),
    # Measurement settings: '13B active parameters' is not a latency, '2655 queries' not a rater count.
    ("latency", r"\blatenc(?:y|ies)\b", ("latenc",)),
    ("random seed", r"\bseeds?\b", ("seed",)),
    (
        "raters",
        r"\b(?:raters?|annotators?|label(?:l)?ers?|crowd-?workers?|contractors?)\b",
        ("rater", "annotat", "label", "contractor", "crowd", "worker", "participant", "evaluator"),
    ),
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
# Latency is not one: 'no additional inference latency' is a complete answer.
NUMERIC_SETTINGS = {
    "batch size", "learning rate", "warmup", "dropout", "weight decay", "epochs",
    "training steps", "random seed",
}


def check_requested_attribute(question: str, answer: str, evidence: str) -> str:
    """Check that a numeric answer is stated for the property the task asks about.

    The answer or a cited sentence containing one of its numbers must name the
    property. For example, '32k tokens' from a context-size sentence cannot answer
    a batch-size request. A request for a training setting's value needs a number;
    other answers without numbers are not checked. A task listing several properties
    ('corpus, batch size, steps and optimizer') needs only one of them named, since
    one sentence rarely names them all; named benchmarks are still checked.
    """
    properties = [
        row for row in REQUESTED_PROPERTIES if re.search(row[1], question, re.IGNORECASE)
    ]
    requested = (properties if len(properties) == 1 else []) + [
        row for row in NAMED_BENCHMARKS if re.search(row[1], question, re.IGNORECASE)
    ]
    # Numbers repeated from the question, as in 'Orion 8x7B', are names, not values.
    numbers = extract_numbers(answer) - extract_numbers(question)
    if not numbers:
        # A value written as a word ("a single epoch") stands in for digits when the text states it.
        numbers = extract_spelled_values(answer) - extract_spelled_values(question)
        if numbers and not numbers <= extract_evidence_numbers(evidence):
            return "No cited sentence states the answer's spelled-out value."
    if not numbers:
        asks_value = re.search(
            r"\b(?:what|which|how (?:many|much|large|big))\b", question, re.IGNORECASE
        ) and not re.search(r"\bwhy\b", question, re.IGNORECASE)
        for label, _, _ in requested:
            if asks_value and label in NUMERIC_SETTINGS:
                return f"The requested {label} needs an explicit value, but the answer has none."
        return ""
    sentences = SENTENCE_END.split(normalize_evidence_text(evidence))
    near_answer = " ".join(
        [answer] + [part for part in sentences if extract_evidence_numbers(part) & numbers]
    ).casefold()
    for label, _, names in requested:
        # Names are word prefixes, so 'optimi' matches optimizer but 'lr' skips 'already'.
        if not any(re.search(r"\b" + re.escape(name), near_answer) for name in names):
            return f"No cited sentence states the answer's values as the requested {label}."
    # Values that no requested property names, such as a context size given for
    # 'batch size and learning rate', answer a different question.
    if len(properties) > 1 and not any(
        re.search(r"\b" + re.escape(name), near_answer)
        for _, _, names in properties
        for name in names
    ):
        labels = ", ".join(label for label, _, _ in properties)
        return f"No cited sentence states the answer's values as any requested property: {labels}."
    return ""


# "Which/what" directly before the asked kind, with at most one word between them ("What training
# hardware"). "What beta2 value does the Adam optimizer use?" asks for a value, not for an optimizer.
ASKS_FOR = r"\b(?:which|what)\s+(?:kind\s+of\s+|type\s+of\s+)?(?:[\w-]+\s+)?"
# Asked kind: label, question pattern, and names (regex word starts) that identify one instance.
# The bare word "normalization" is not a name: a "text normalization (TN) module" is not a layer norm.
NAMED_KINDS = (
    ("hardware", ASKS_FOR + r"(?:hardware|gpus?|tpus?|accelerators?|chips?)\b", HARDWARE_NAMES),
    (
        "optimizer",
        ASKS_FOR + r"(?:optimi[sz]er|optimi[sz]ation\s+(?:algorithm|method|procedure))\b",
        ("adam", "sgd", r"gradient\s+descent", "adafactor", "adagrad", "adadelta", "rmsprop", "lamb",
         "lion", "momentum", "l-?bfgs", "shampoo", "muon", "sophia"),
    ),
    (
        "tokenizer",
        ASKS_FOR + r"tokeni[sz](?:er|ation)\b",
        ("tokeni", "bpe", "bbpe", r"byte-?\s?pair", r"sentence-?\s?piece", r"word-?\s?piece", "tiktoken", "unigram"),
    ),
    (
        "positional encoding",
        # Also "How does Orion encode token positions?"
        ASKS_FOR + r"(?:positional|position)\s+(?:encodings?|embeddings?)\b|\bhow\b[^?]*\bencod\w*\b[^?]*\bpositions?\b",
        ("position", "rotary", "rope", "sinusoid", "sine", "cosine", "alibi"),
    ),
    (
        "normalization",
        ASKS_FOR + r"normali[sz]ation\b",
        (r"rms\s?norm", "root mean square", r"layer\s?norm", r"batch\s?norm", r"pre-?norm", r"post-?norm", r"qk-?norm"),
    ),
    (
        "activation",
        ASKS_FOR + r"(?:activation(?:\s+functions?)?|non-?linearit(?:y|ies))\b",
        ("relu", "gelu", "swiglu", "silu", "swish", "geglu", "tanh", "sigmoid", "softplus"),
    ),
)
QUESTION_WORDS = re.compile(r"\b(?:what|which|how|when|where|why|who)\b", re.IGNORECASE)


def check_named_kind(question: str, answer: str) -> str:
    """Check that an answer to 'Which optimizer does Orion use?' names an optimizer.

    The same applies to hardware, tokenizers, positional encodings, normalization and
    activation functions. "Orion uses multi-head attention" names no optimizer, so it
    answers another question. Only a one-part question that asks for this one kind is
    checked, so a partial answer to a longer task is kept. Naming an optimizer does not
    prove that the cited text supports it.
    """
    asked = [row for row in NAMED_KINDS if re.search(row[1], question, re.IGNORECASE)]
    if len(asked) != 1 or len(QUESTION_WORDS.findall(question)) != 1:
        return ""
    label, _, names = asked[0]
    # "Which optimizer and learning rate ..." also accepts an answer giving only the learning rate.
    others = {row[0] for row in REQUESTED_PROPERTIES if re.search(row[1], question, re.IGNORECASE)}
    if others - {label}:
        return ""
    text = normalize_evidence_text(answer).casefold()
    if any(re.search(r"\b" + name, text) for name in names):
        return ""
    return f"The answer names no {label}."


PER_UNIT =re.compile(r"\b(?:per|each|every)\b", re.IGNORECASE)
# Words after a number that name no unit. Prepositions and comparisons such as "over" in
# "from 70M to over 10B" follow many numbers, so sharing one does not make a shared unit.
UNIT_STOPWORDS = {
    "on", "in", "of", "for", "to", "with", "per", "each", "every", "and", "or", "the", "a", "an", "i", "e",
    "over", "under", "up", "from", "between", "than", "about", "around", "at", "by", "as", "more", "less",
}


def words_after_numbers(sentence: str) -> list[tuple[Decimal, set[str]]]:
    """Pair each value with the next few words, its unit: '180K H800 GPU hours'.

    Values are compared by size ('180K' equals '180,000'); digits inside a name
    such as 'H800' are not values.
    """
    sentence = THOUSANDS_COMMA.sub("", sentence)
    pairs = []
    for match in re.finditer(r"(?<![\w.])(\d+(?:\.\d+)?)([KMBT])?\b((?:\W+[A-Za-z][\w-]*){0,3})", sentence):
        value = Decimal(match[1]) * (SCALES[match[2].casefold()] if match[2] else 1)
        unit = {word for word in re.findall(r"[a-z][\w-]*", match[3].casefold()) if word not in UNIT_STOPWORDS}
        pairs.append((value, unit))
    return pairs


def check_rate_given_as_total(question: str, answer: str, evidence: str) -> str:
    """Reject a per-unit value given as a total when the passage also states the total.

    'On each trillion tokens training requires only 180K GPU hours' is a rate; if the
    same evidence says 'pre-training costs 2664K GPU hours', an answer giving 180K
    GPU hours without 'per' took the rate. Questions asking for a rate are not checked.
    """
    if PER_UNIT.search(question) or PER_UNIT.search(answer):
        return ""
    rates, totals = {}, []
    for sentence in SENTENCE_END.split(normalize_evidence_text(evidence)):
        pairs = words_after_numbers(sentence)
        if PER_UNIT.search(sentence):
            for number, unit in pairs:
                rates.setdefault(number, set()).update(unit)
        else:
            totals.extend(pairs)
    total_values = {value for value, _ in totals}
    asked = {value for value, _ in words_after_numbers(question)}
    for value, _ in words_after_numbers(answer):
        if value in rates and value not in total_values | asked and any(
            other != value and unit & rates[value] for other, unit in totals
        ):
            return f"The cited text states {value:f} per unit, but the answer gives it as a total."
    return ""


def task_evidence(evidence: list[dict], tasks: list[dict], index: int) -> list[dict]:
    """Passages retrieved for one task, from the papers that task may use."""
    paper = tasks[index].get("paper")
    allowed = paper if isinstance(paper, list) else [paper]
    return [
        row
        for row in evidence
        if index in row.get("task_ids", range(len(tasks)))
        and (not paper or row.get("title") in allowed)
    ]


def stage_text(text: str) -> str:
    """Normalize text for stage binding: one line, no thousands separators."""
    return THOUSANDS_COMMA.sub("", normalize_evidence_text(text))


def stage_passages(rows: list[dict]) -> list[tuple[str, str]]:
    """Pair each passage's normalized text with its section heading."""
    return [(stage_text(row["content"]), row.get("location") or "") for row in rows]


def stage_numbers(question: str, answer: str) -> set[str]:
    """Values the answer states beyond the question's own, normalized for stage binding."""
    return numbers_in(stage_text(answer)) - numbers_in(stage_text(question))


def check_requested_stage(question: str, answer: str, cited: list[dict], available: list[dict]) -> str:
    """Check that the answer's numbers are stated for the training stage the task names.

    'Batch size for pre-training' is not answered by the fine-tuning batch size, and
    'GPU hours for context extension' is not answered by the cost of the whole run.
    A number that nothing ties to another stage is accepted; this compares wording only
    and does not establish that the value is right.
    """
    asked = requested_stage(question)
    if not asked:
        return ""
    numbers = stage_numbers(question, answer)
    if not numbers:
        return ""
    return find_stage_mismatch(asked, numbers, stage_passages(cited), stage_passages(available))


def repair_requested_stage(
    question: str, answer: str, cited: list[dict], available: list[dict]
) -> tuple[str, str] | None:
    """Find the source label and sentence that state the stage's own figure for a whole-run answer.

    For an answer that gave the headline total where a stage was asked, the passages may
    state the stage's figure outright. The sentence is quoted as written; the caller must
    verify it like any other answer.
    """
    found = find_stage_repair(
        question, stage_numbers(question, answer), stage_passages(cited), stage_passages(available)
    )
    return (available[found[0]]["id"], found[1]) if found else None


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
        if reason or check_requested_unit(task["question"], answer, eligible):
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
        if reason or check_requested_unit(task["question"], answer, eligible):
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
                    allowed_source_ids=[row["id"] for row in task_evidence(evidence, tasks, i)],
                    **task,
                )
                for i, task in enumerate(tasks)
            ],
        },
    )


# "According to the Orion paper, ..." frames every part of the question that follows it.
FRAMING_SOURCE = re.compile(
    r"^\W*(?:according\s+to|in|from|per|based\s+on)\s+the\s+[^,;?]{2,80}?\s+paper\b", re.IGNORECASE
)


def named_source_governs(tasks: list[dict], index: int, source_papers: list[str], question: str = "") -> bool:
    """Whether a source paper the question names ('the Orion paper') applies to this task.

    A one-task question is governed by it, as is every task of a question that opens with
    'According to the X paper', a task without a paper scope, or one whose papers include a
    named source. Otherwise a task scoped only to other papers answers another clause.
    """
    if len(tasks) == 1 or index >= len(tasks) or FRAMING_SOURCE.search(question):
        return True
    paper = tasks[index].get("paper")
    scope = set(paper if isinstance(paper, list) else [paper] if paper else [])
    return not scope or bool(scope & set(source_papers))


def verify_answer_citations(
    draft: DraftAnswer,
    evidence: list[dict],
    tasks: list[dict],
    question: str = "",
    source_papers: list[str] | None = None,
    repair: bool = True,
) -> dict:
    """Validate each detail independently so one source cannot cover unrelated tasks.

    Empty, omitted, duplicate-source and invalid records cannot make other details
    complete. Numbers must occur in the cited passages, but their meaning still
    requires semantic evaluation. A whole-run total given for one training stage is
    replaced by the sentence that states the stage's own figure, once that sentence
    passes the same checks (`repair`).
    """
    sources = {row["id"]: row for row in evidence}
    accepted, used, rejections, repairs = [], set(), [], []
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
        if not reason:
            outside = ids - {row["id"] for row in task_evidence(evidence, tasks, index)}
            # Gemma sometimes cites a label from another task, or only such labels. Drop
            # them only when the answer states values beyond the question's own numbers:
            # the number check below must then find those values in this detail's own
            # passages, and it cites them. Nothing would verify a value-free answer.
            values = extract_numbers(detail.answer) - extract_numbers(tasks[index]["question"])
            if outside and not values:
                reason = "A cited passage is outside this detail's task or paper scope."
            ids -= outside
        if not reason:
            reason, supporting_text = check_explicit_answer_scope(
                tasks[index]["question"],
                [sources[label]["content"] for label in ids],
                question if len(tasks) == 1 else "",
            )
        if not reason:
            reason = check_requested_unit(
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
            elif MISSING_FACT_STATEMENT.search(text):
                reason = "The answer says the requested fact is not stated."
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
        if not reason:
            reason = check_named_kind(tasks[index]["question"], text)
        if not reason:
            reason = check_rate_given_as_total(tasks[index]["question"], text, supporting_text)
        if not reason:
            stage_args = (
                tasks[index]["question"],
                text,
                [sources[label] for label in sorted(ids)],
                task_evidence(evidence, tasks, index),
            )
            reason = check_requested_stage(*stage_args)
            fix = repair_requested_stage(*stage_args) if reason and repair else None
            if fix:
                # The passages state the stage's own figure; that sentence must pass every check above.
                label, sentence = fix
                absent = DetailAnswer(availability="unavailable", source_ids=[], answer="")
                again = verify_answer_citations(
                    DraftAnswer(
                        answers=[absent] * index
                        + [DetailAnswer(availability="available", source_ids=[label], answer=sentence)]
                    ),
                    evidence,
                    tasks,
                    question,
                    source_papers,
                    repair=False,
                )
                if again["accepted"]:
                    accepted.extend(again["accepted"])
                    used.update(again["source_ids"])
                    repairs.append(
                        {"detail_index": index, "reason": reason, "answer": again["accepted"][0]["answer"]}
                    )
                    continue
        # 'According to the Orion paper' needs an Orion citation, not only Sirius's. With several
        # tasks the phrase governs only the tasks about that paper: in "Which algorithm does
        # Vega use, and which does the Orion paper introduce?" the Vega task cannot cite Orion.
        if not reason and source_papers and named_source_governs(tasks, index, source_papers, question) and not any(
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
        "repairs": repairs,
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
