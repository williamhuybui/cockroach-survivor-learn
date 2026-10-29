"""Check explicit benchmark editions and named model variants in cited text.

This is a narrow scope check, not a semantic judge. It handles named benchmark
years, model sizes such as Orion 8x7B, and explicit versions such as Orion v2.
General dates, requested list lengths and arbitrary numbers are not constraints.
"""

import re

# Read a benchmark name and year, such as "Trial 2023" or "WMT-2019".
BENCHMARK_YEAR_PATTERN = re.compile(
    r"\b(?P<name>[A-Za-z][A-Za-z0-9_-]*)[\s-]+(?P<edition>(?:19|20)\d{2})\b"
)
# Read a named model size, such as "Orion-13B" or "Orion 8x7B".
MODEL_SIZE_PATTERN = re.compile(
    r"\b(?P<name>[A-Za-z][A-Za-z_-]*?)[ -]*(?P<edition>\d+(?:x\d+)?(?:\.\d+)?b)\b",
    re.IGNORECASE,
)
# Read an explicit version label, such as "Orion v2" or "Orion version 2.1".
MODEL_VERSION_PATTERN = re.compile(
    r"\b(?P<name>[A-Za-z][A-Za-z0-9_-]*?)[ -]+(?:v|version\s+)(?P<edition>\d+(?:\.\d+)*)\b",
    re.IGNORECASE,
)
# These words introduce dates or quantities, rather than naming benchmarks/models.
SCOPE_STOPWORDS = {
    "in",
    "on",
    "of",
    "from",
    "for",
    "during",
    "by",
    "since",
    "until",
    "before",
    "after",
    "between",
    "through",
    "the",
    "and",
    "or",
    "uses",
    "has",
    "with",
    "parameters",
    "active",
    "total",
    "about",
    "approximately",
    "our",
    "their",
    "a",
    "an",
    "to",
    "model",
    "models",
}


def extract_named_constraints(text: str) -> set[tuple[str, str, str]]:
    """Read explicit name/edition pairs without treating every number as a condition."""
    text = " ".join(text.replace("×", "x").split())
    constraints = set()
    for kind, pattern in (
        ("benchmark", BENCHMARK_YEAR_PATTERN),
        ("model size", MODEL_SIZE_PATTERN),
        ("model version", MODEL_VERSION_PATTERN),
    ):
        for match in pattern.finditer(text):
            name = match["name"].strip("-_").casefold()
            if name not in SCOPE_STOPWORDS:
                constraints.add((kind, name, match["edition"].casefold()))
    return constraints


def has_unqualified_variant(text: str, kind: str, edition: str) -> bool:
    """Recognize 'our 65B model' without borrowing another named model's size.

    A source can identify its own model through the surrounding paper context.
    Ignore occurrences that explicitly belong to another named model, such as
    'Other-65B', so they cannot excuse a conflicting 'Orion-7B' statement.
    """
    text = " ".join(text.replace("×", "x").split())
    pattern = MODEL_SIZE_PATTERN if kind == "model size" else MODEL_VERSION_PATTERN
    named_spans = [
        match.span("edition")
        for match in pattern.finditer(text)
        if match["name"].strip("-_").casefold() not in SCOPE_STOPWORDS
    ]
    # Size labels can stand alone; numeric version labels need v/version to
    # distinguish them from ordinary counts elsewhere in the paragraph.
    value_pattern = (
        rf"(?<![\w.]){re.escape(edition)}(?![\w.])"
        if kind == "model size"
        else rf"\b(?:v|version\s+){re.escape(edition)}(?![\w.])"
    )
    return any(
        not any(start <= match.end() - 1 < stop for start, stop in named_spans)
        for match in re.finditer(value_pattern, text, re.IGNORECASE)
    )


def check_explicit_answer_scope(
    task_question: str, passages: list[str], original_question: str = ""
) -> tuple[str, str]:
    """Return a mismatch reason and the text eligible to support this answer part.

    Benchmark editions must appear in the cited body paragraphs. Model variants
    reject explicit conflicts, but equivalent wording such as 'our 65B model'
    does not require the full name again. Paper/task association is checked by the
    caller. This function does not prove semantic support for every statement.
    Original constraints override a planner's changed edition when unambiguous.
    Callers supply the original question only when it belongs to this task.
    """
    required = extract_named_constraints(task_question)
    originals = extract_named_constraints(original_question)
    for kind, name, edition in originals:
        editions = {value for k, n, value in originals if (k, n) == (kind, name)}
        if len(editions) == 1:
            required = {item for item in required if item[:2] != (kind, name)}
    required.update(originals)
    if not required:
        return "", " ".join(passages)
    paragraphs = [part for text in passages for part in re.split(r"\n\s*\n", text)]
    paragraph_constraints = [extract_named_constraints(part) for part in paragraphs]
    named = set().union(*paragraph_constraints)
    eligible = []
    for constraint in sorted(required):
        kind, name, edition = constraint
        matches = [
            part for part, constraints in zip(paragraphs, paragraph_constraints)
            if constraint in constraints
        ]
        if kind != "benchmark":
            conflicts = {item for item in named if item[:2] == (kind, name)}
            if (
                matches
                or not conflicts
                or any(
                    has_unqualified_variant(part, kind, edition) for part in paragraphs
                )
            ):
                continue
            return (
                f"Cited evidence contradicts the requested {kind}: {name} {edition}.",
                "",
            )
        if not matches:
            return (
                f"Cited evidence does not establish the requested {kind}: {name} {edition}.",
                "",
            )
        eligible.extend(matches)
    # Keep the paragraph restriction for benchmark years. For model wording,
    # retain ordinary citation/number checks instead of demanding a literal name.
    return "", "\n\n".join(dict.fromkeys(eligible or paragraphs))
