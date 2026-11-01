"""Tell which training stage a number in a passage belongs to.

A request for a pre-training value is answered wrongly by the fine-tuning value or by the
cost of the whole run, and a small model often picks the headline number. These helpers
bind each number to a stage from its own clause, from a scope-setting clause of the same
sentence ("For fine-tuning, ...") or from the section heading. They compare words only and
never decide that an answer is right: a number that nothing ties to another stage is accepted.
"""

import re
from decimal import Decimal
from itertools import product
from typing import NamedTuple

# Stages a paper can report separate values for. Anything else (annealing, ...) is not tracked.
STAGES = {
    "pre-training": re.compile(r"\bpre[- ]{0,2}train(?:ing|ed|s)?\b"),
    "fine-tuning": re.compile(
        r"\bfine[- ]{0,2}tun(?:e|es|ed|ing)\b|\bsft\b|\binstruction[- ]tun(?:e|ed|ing)\b"
    ),
    "post-training": re.compile(r"\bpost[- ]{0,2}train(?:ing|ed)?\b"),
    "context extension": re.compile(
        r"\bcontext(?:[- ]length| window)?[- ]extension\b"
        r"|\bextend(?:s|ed|ing)? (?:the |its )?context\b"
        r"|\blong[- ]context (?:pre[- ]{0,2}train\w*|train\w*|stage|phase)\b"
    ),
    "reward modeling": re.compile(r"\breward[- ]model(?:l?ing|s)?\b|\brm\b"),
    "reinforcement learning": re.compile(r"\breinforcement[- ]learning\b|\brlhf\b|\bppo\b|\bgrpo\b|\brl\b"),
}
# Post-training includes the supervised, reward-model and reinforcement-learning steps.
POST_TRAINING_PARTS = ("fine-tuning", "reward modeling", "reinforcement learning")
OVERLAPPING = {("post-training", part) for part in POST_TRAINING_PARTS} | {
    (part, "post-training") for part in POST_TRAINING_PARTS
}
# 'The pre-trained model' describes a model; 'pre-trained on' names the stage.
ADJECTIVE_FORM = re.compile(
    r"\b(?:pre[- ]{0,2}train|fine[- ]{0,2}tun)ed\b(?!\s+(?:on|for|with|from|using|at|in)\b)"
)
# A stage named only to compare or to place in time ("the same as in pre-training", "after
# pre-training") does not set the scope of the figures around it.
COMPARISON = re.compile(
    r"\b(?:than|unlike|versus|vs\.?|compared (?:to|with)|in contrast to|instead of|rather than"
    r"|same as|as in|as for|after|before|following|prior to|subsequent to)\b"
    r"(?:\s+[\w-]+){0,3}"
)
# The whole run rather than one stage: "2.788M GPU hours for its full training". The entire
# training dataset or set is the data of a run, not the run.
WHOLE_RUN = re.compile(
    r"\b(?:full|entire|whole|complete|total)\s+(?:training|cost|costs|compute)\b"
    r"(?!\s+(?:data|dataset|datasets|corpus|corpora|set|sets)\b)|\bin total\b"
)
# Clauses end at punctuation or a conjunction; a stage word binds the numbers of its own clause.
CLAUSE_BREAK = re.compile(r"\s*[,;:()•]\s*|\s+(?:and|while|whereas|but|with|which|then)\s+", re.IGNORECASE)
SENTENCE_END = re.compile(r"(?<=[.!?;])\s+")
# A value, not the digits of a name such as L2 or H800; 1e-4 is one value.
NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?:e[-+]?\d+)?")
UNIT_STOPWORDS = {
    "on", "in", "of", "for", "to", "with", "per", "each", "every", "and", "or", "the", "a", "an",
    "its", "their", "our", "it", "is", "are", "was", "were", "only", "about", "approximately",
}


def stage_tags(text: str) -> set[str]:
    """Stages named in already lowercased text."""
    return {stage for stage, pattern in STAGES.items() if pattern.search(text)}


def mentions_text(text: str) -> str:
    """Lowercased text without the words that merely describe a model or compare with a stage."""
    return ADJECTIVE_FORM.sub(" ", COMPARISON.sub(" ", text))


def requested_stage(question: str) -> str | None:
    """The one stage a question asks about, or None when it names none or several."""
    found = stage_tags(mentions_text(question.casefold()))
    return next(iter(found)) if len(found) == 1 else None


def names_stage(tags: set[str], asked: str) -> bool:
    """Whether the tags include the asked stage or one that contains it."""
    return any(tag == asked or (asked, tag) in OVERLAPPING for tag in tags)


def numbers_in(text: str) -> set[str]:
    """Values stated in normalized text, without thousands separators."""
    return set(NUMBER.findall(text.casefold()))


def number_pattern(number: str) -> re.Pattern:
    """The number as a whole token, with an optional scale such as K or million."""
    return re.compile(
        rf"(?<![\w.]){re.escape(number)}(?:k\b|m\b|b\b|t\b|\s?(?:thousand|million|billion|trillion)\b)?(?![\w]|\.\d)"
    )


def unit_after(clause: str, end: int) -> set[str]:
    """The words just after a number, its unit: 'gpu hours' in '119k gpu hours for ...'."""
    words = re.findall(r"[a-z][\w-]*", clause[end:])[:3]
    return {word for word in words if word not in UNIT_STOPWORDS}


class Clause(NamedTuple):
    """A clause of a passage and what its figures are bound to."""

    text: str  # lowercased
    stages: set[str]
    from_text: bool  # the passage text names the stage, not only the section heading
    whole_run: bool
    named: bool  # the clause itself names the stage
    quote: str  # the clause as written
    sentence: str  # its sentence as written


def bind_clauses(passage: tuple[str, str]) -> list[Clause]:
    """Split a (text, location) passage into clauses with the stages their figures belong to.

    A clause names its own stage, or is a whole-run figure, or else takes the scope set by
    the sentence: clauses that state no number ("For fine-tuning,"), the one stage the
    sentence names, and finally the section heading. Stages that come from the section
    heading are marked as not from the text. A stage named in a neighbouring sentence is
    not carried over: it is often only an adjective ("the pre-trained weights").
    """
    text, location = passage
    heading = stage_tags(location.casefold())
    records = []
    for sentence in SENTENCE_END.split(text):
        quotes = [part for part in CLAUSE_BREAK.split(sentence) if part.strip()]
        parts = [quote.casefold() for quote in quotes]
        own = [stage_tags(mentions_text(part)) for part in parts]
        setting = set().union(*(tags for part, tags in zip(parts, own) if not re.search(r"\d", part)))
        named = set().union(*own)
        sentence_total = any(
            WHOLE_RUN.search(part) for part in parts if not re.search(r"\d", part)
        )
        scoped = setting or (named if len(named) == 1 else set())
        for quote, part, tags in zip(quotes, parts, own):
            if tags:
                stages, from_text, whole_run = tags, True, False
            elif WHOLE_RUN.search(part) or (sentence_total and not setting):
                stages, from_text, whole_run = set(), False, True
            elif scoped:
                stages, from_text, whole_run = scoped, True, False
            else:
                stages, from_text, whole_run = heading, False, False
            records.append(Clause(part, stages, from_text, whole_run, bool(tags), quote, sentence))
    return records


def label_occurrences(number: str, passages: list[tuple[str, str]], asked: str) -> list[tuple[str, set[str]]]:
    """Label every place the number appears as support, other, total or neutral, with its unit."""
    pattern = number_pattern(number)
    labels = []
    for passage in passages:
        for clause in bind_clauses(passage):
            found = pattern.search(clause.text)
            if not found:
                continue
            if clause.stages:
                label = "support" if names_stage(clause.stages, asked) else "other"
            else:
                label = "total" if clause.whole_run else "neutral"
            labels.append((label, unit_after(clause.text, found.end())))
    return labels


def stage_values(passages: list[tuple[str, str]], asked: str) -> list[tuple[str, set[str]]]:
    """Numbers (with their unit) that the text itself states for the asked stage."""
    values = []
    for passage in passages:
        for clause in bind_clauses(passage):
            if clause.from_text and names_stage(clause.stages, asked):
                values.extend(
                    (found[0], unit_after(clause.text, found.end())) for found in NUMBER.finditer(clause.text)
                )
    return values


def find_stage_mismatch(
    asked: str,
    numbers: set[str],
    cited: list[tuple[str, str]],
    available: list[tuple[str, str]],
) -> str:
    """Return why the cited text ties every number of the answer to another stage, or ''.

    A number is contradicted when some cited place binds it to a different stage and no
    place supports the asked one; places that say nothing about stages neither support
    nor contradict it, because a passage fragment often lacks the sentence that names the
    stage. The answer is rejected only if every number is contradicted, so one supported
    or unexplained number accepts it. A whole-run total contradicts a number only when
    the available passages also give the asked stage a value of its own in the same
    unit; a paper with a single stage calls its only value "total".
    """
    own_values = None
    reasons = []
    for number in numbers:
        places = label_occurrences(number, cited, asked)
        labels = {label for label, _ in places}
        if "support" in labels:
            return ""
        if "other" in labels:
            reasons.append(f"The cited text states the answer's value for a different stage, not for {asked}.")
            continue
        if "total" not in labels:
            return ""
        if own_values is None:
            own_values = stage_values(available, asked)
        units = set().union(*(unit for label, unit in places if label == "total"))
        competing = {value for value, unit in own_values if unit & units and value not in numbers}
        if not competing:
            return ""
        reasons.append(
            f"The cited text gives the answer's value for the whole training, but {asked} has its own "
            f"value ({', '.join(sorted(competing, key=float))})."
        )
    return reasons[0] if reasons else ""


SCALES = {
    "k": 10**3, "thousand": 10**3, "m": 10**6, "million": 10**6,
    "b": 10**9, "billion": 10**9, "t": 10**12, "trillion": 10**12,
}
SCALE_AFTER = re.compile(r"(?:(k|m|b|t)\b|\s?(thousand|million|billion|trillion)\b)")
# A figure per unit ("180K GPU hours per trillion tokens") is a rate, not what a stage costs in all.
RATE = re.compile(r"\b(?:per|each|every)\b|/")
# The stages of a breakdown add up to its total, give or take the rounding of a headline figure.
SUM_TOLERANCE = Decimal("0.01")
# A longer sentence is usually a flattened table, not a statement worth quoting.
MAX_QUOTE = 300


def figures(clause: str) -> list[tuple[str, Decimal, frozenset[str]]]:
    """Each value of a lowercased clause as (digits, size, unit): '119k gpu hours' is 119000, {gpu, hours}."""
    found = []
    for match in NUMBER.finditer(clause):
        if "e" in match[0]:
            continue
        scale = SCALE_AFTER.match(clause, match.end())
        size = Decimal(match[0]) * (SCALES[scale[1] or scale[2]] if scale else 1)
        words = re.findall(r"[a-z][\w-]*", clause[(scale or match).end():])
        # Names such as H800 are not part of the unit.
        unit = [word for word in words if not re.search(r"\d", word)][:3]
        found.append((match[0], size, frozenset(unit) - UNIT_STOPWORDS))
    return found


def find_stage_repair(
    question: str,
    numbers: set[str],
    cited: list[tuple[str, str]],
    available: list[tuple[str, str]],
) -> tuple[int, str] | None:
    """Find the sentence that gives the asked stage its own figure when the answer gave the whole run's.

    Only a plain whole-run total is repaired: its number sits in a whole-run clause and no cited
    place binds it to a stage. The stage's figure must be in the unit the question names ('GPU
    hours'), must not be a rate, and must be the only value the passages bind to the stage. It
    must also belong to a breakdown: the figures of the stages together add up to the total,
    so a model-size variant or an ablation never passes for a stage. Returns the index of the
    passage in `available` and the clause that states it, or the whole sentence when the stage
    is named elsewhere in it.
    """
    asked = requested_stage(question)
    if not asked or len(numbers) != 1:
        return None
    (number,) = numbers
    pattern = number_pattern(number)
    totals = []
    for passage in cited:
        for clause in bind_clauses(passage):
            if not pattern.search(clause.text):
                continue
            if clause.stages:
                return None
            if clause.whole_run:
                totals += [(size, unit) for digits, size, unit in figures(clause.text) if digits == number]
    named_words = {word.removesuffix("s") for word in re.findall(r"[a-z]+", question.casefold())}
    totals = [
        (size, unit) for size, unit in totals if unit and {word.removesuffix("s") for word in unit} <= named_words
    ]
    units = {unit for _, unit in totals}
    own: dict[str, dict[Decimal, list[tuple[int, Clause]]]] = {}
    for index, passage in enumerate(available):
        for clause in bind_clauses(passage):
            if clause.from_text and len(clause.stages) == 1 and not RATE.search(clause.text):
                (stage,) = clause.stages
                for _, size, unit in figures(clause.text):
                    if unit in units:
                        own.setdefault(stage, {}).setdefault(size, []).append((index, clause))
    if len(own.get(asked, {})) != 1:
        return None
    ((asked_size, places),) = own[asked].items()
    others = {
        stage: sizes for stage, sizes in own.items() if stage != asked and (stage, asked) not in OVERLAPPING
    }
    for total, _ in totals:
        for picks in product(*([None, *sizes] for sizes in others.values())):
            chosen = {stage: size for stage, size in zip(others, picks) if size is not None}
            overlapping = any((first, second) in OVERLAPPING for first in chosen for second in chosen)
            if chosen and not overlapping and abs(asked_size + sum(chosen.values()) - total) <= total * SUM_TOLERANCE:
                index, clause = min(places, key=lambda place: (not place[1].named, place[0]))
                quote = (clause.quote if clause.named else clause.sentence).strip()
                return (index, quote) if len(quote) <= MAX_QUOTE else None
    return None
