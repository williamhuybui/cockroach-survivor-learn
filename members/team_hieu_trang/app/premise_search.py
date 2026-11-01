"""Find the sentence in a question's paper that rules out what the question assumes.

"Why does the Transformer use recurrent layers?" takes recurrent layers for granted.
If the paper says it dispenses with recurrence, that sentence answers the question as
a correction. The search is plain text matching over the requested paper's own
sections: no model call, and only the question and the indexed paper are read.
"""

import re

from .grounding import normalize_evidence_text
from .paper_catalog import catalog_identity_names, name_pattern
from .paper_store import open_parent_section_store
from .retrieval import load_scoped_chunks

# The question credits the paper's model with an action on something. Finite verbs
# only, so the noun in "reinforcement learning" or "pre-training" does not qualify.
ASSUMING_VERBS = (
    r"use[sd]?|utili[sz]e[sd]?|employ(?:s|ed)?|adopt(?:s|ed)?|appl(?:y|ies|ied)|"
    r"rel(?:y|ies|ied)|depend(?:s|ed)?|require[sd]?|need(?:s|ed)?|include[sd]?|"
    r"contain(?:s|ed)?|train(?:s|ed)?|fine-?tune[sd]?|incorporate[sd]?|leverage[sd]?|"
    r"implement(?:s|ed)?|add(?:s|ed)?"
)
ASSUMING_VERB = re.compile(r"\b(?:" + ASSUMING_VERBS + r")\b", re.IGNORECASE)
# Question shapes that take their object for granted. Each captures that object: the
# thing the verb acts on, not the context around it ("in its encoder", "alongside ...").
OBJECT_FRAMES = (
    # Why does the Transformer use recurrent layers in its encoder?
    re.compile(
        r"\bwhy\s+(?:does|do|did)\s+.+?\s+(?:" + ASSUMING_VERBS + r")(?:\s+(?:on|upon|in|with))?"
        r"\s+(?P<object>.+)",
        re.IGNORECASE,
    ),
    # Which reward model does Orion train ...? What proprietary datasets did Sirius use ...?
    re.compile(
        r"\b(?:which|what)\s+(?P<object>.+?)\s+(?:does|do|did)\s+.+?\s+(?:" + ASSUMING_VERBS + r")\b",
        re.IGNORECASE,
    ),
    # How large is the critic model that GRPO trains ...? How many experts does Mixtral use?
    re.compile(
        r"\bhow\s+(?:large|big|many|much|long|deep|wide)\s+(?:(?:is|are|was|were)\s+)?"
        r"(?:the\s+|a\s+|an\s+)?(?P<object>.+?)\s+(?:that|which|does|do|did)\s+.+?\s+(?:"
        + ASSUMING_VERBS
        + r")\b",
        re.IGNORECASE,
    ),
)
# Where the object ends and the context around it begins.
CONTEXT_START = re.compile(
    r"\s+(?:in|on|for|with|during|before|after|at|by|from|as|while|when|if|because|"
    r"alongside|along|within|across|under|over|into)\b.*",
    re.IGNORECASE,
)
# A question that already negates or excludes something does not assume it is present:
# "What does DPO eliminate ...?" asks about the exclusion itself.
QUESTION_NEGATION = re.compile(
    r"\b(?:without|dispens\w*|forg\w*|forego\w*|eliminat\w*|avoid\w*|instead|rather|"
    r"no|not|never|lack\w*|omit\w*|remov\w*|free|absen\w*)\b|n['’]t\b",
    re.IGNORECASE,
)

# Words that name no assumed thing: function words and quantity words.
QUESTION_STOPWORDS = set(
    "the a an of in on for to with by from as at into over under between among about "
    "against before after during alongside along while than then that this these those "
    "its it their them they his her our we you your and or but if so also only very "
    "more most less many much large big long high low deep wide often".split()
)
# Words every paper has, so a match on them alone says nothing about the assumed thing.
GENERIC_TERMS = set(
    "model models data dataset datasets training learning network networks layer layers "
    "method methods approach approaches system systems task tasks performance result "
    "results size sizes number parameter parameters language input inputs output outputs "
    "architecture component components technique techniques algorithm algorithms step "
    "steps process feature features information type types level stage part set way time "
    "value values function functions setting settings experiment experiments paper work "
    "study tokens token cost costs dollars dollar hours days gpus gpu hardware throughput "
    "accuracy score scores rate rates amount budget epochs scale setup kind".split()
)

# Verbs whose negation says the thing is absent. "Does not train X" is left out: a model
# can use X without training it.
NEGATED_USE_VERBS = (
    r"use|utili[sz]e|employ|adopt|apply|rely\s+on|depend\s+on|require|need|involve|"
    r"include|contain|incorporate|leverage|implement"
)
# Wording by which a paper rules something out. The words just after a cue (its scope)
# are what the cue excludes. "No" governs only the next few words ("no recurrence").
RULING_CUE = re.compile(
    r"\b(?:(?P<long>without|dispens(?:e|es|ed|ing)\s+with|for(?:e)?go(?:es|ing)?|"
    r"eliminat(?:e|es|ed|ing)|avoid(?:s|ed|ing)?|free\s+(?:of|from)|no\s+longer|"
    r"(?:does|do|did)(?:\s+not|n['’]t)\s+(?:" + NEGATED_USE_VERBS + r"))"
    r"|(?P<short>no))\b",
    re.IGNORECASE,
)
SCOPE_WORDS = {"long": 6, "short": 3}
# The excluded thing is the cue's object, so a specific term must follow it closely:
# "without explicit reward modeling" rules out a reward model, but "without being
# blocked by subsequently arriving tokens" does not rule out tokens.
ADJACENT_WORDS = 3
# A preposition ends the object ("supervision on retrieved evidence" excludes
# supervision) unless it completes a verb: "resorting to", "relying on".
PREPOSITIONS = {
    "on", "of", "for", "in", "to", "from", "by", "with", "at", "over", "under", "into",
    "across", "during", "via", "per", "than", "as", "within", "between", "among", "about",
}
# "Without using X" excludes X, but "without compromising X" or "without being X" keeps it.
EXCLUDING_GERUNDS = {
    "using", "resorting", "relying", "requiring", "needing", "involving", "employing",
    "utilizing", "utilising", "having", "adopting", "applying", "including", "depending",
}
# A cue that is itself negated, or that names both options, excludes nothing:
# "do not eliminate X" keeps X and "with and without X" describes both settings.
NEGATED_BEFORE = re.compile(
    r"\b(?:with\s+(?:and|or)|(?:not|never|cannot)(?:\s+\w+)?|(?:fail(?:s|ed)?|unable)\s+to)\s*$",
    re.IGNORECASE,
)
# Ablations and result tables describe variants of the model, not its design.
EXPERIMENT_SECTION = re.compile(
    r"experiment|ablation|result|evaluat|analysis|effect of|empirical|appendix|"
    r"training details|implementation details|hyperparameter|setup|benchmark",
    re.IGNORECASE,
)
# A cue stops governing the text after punctuation or a new clause.
SCOPE_END = re.compile(
    r"[.;:!?()—]|\b(?:but|while|whereas|which|that|because|since|although)\b",
    re.IGNORECASE,
)
# The sentence speaks for the paper's own model, not for a baseline or earlier work.
SELF_REFERENCE = re.compile(
    r"\b(?:we|our|ours|us)\b|\bthis\s+(?:[\w-]+\s+)?(?:paper|work|study|article|approach|"
    r"method|model|architecture|technique)\b|\bthe\s+(?:proposed|present)\b",
    re.IGNORECASE,
)
# Sentences about variants and comparisons describe other systems than the paper's own.
# Only the plural "baselines" counts: "estimating the baseline" is a reinforcement-learning term.
VARIANT_WORDS = re.compile(r"\b(?:ablat\w*|variants?|baselines)\b", re.IGNORECASE)
HEADLINE_SECTION = re.compile(r"abstract|introduction|conclusion|discussion|summary|overview", re.IGNORECASE)
# Where authors state their own design. A discussion also speculates about other systems.
CLAIMS_SECTION = re.compile(r"abstract|introduction|conclusion", re.IGNORECASE)
# The title page carries the abstract; its section number is at most this.
FRONT_MATTER_SECTIONS = 2
# References and acknowledgements name other works, never the paper's own design.
SKIPPED_SECTIONS = {"references", "bibliography", "acknowledgements", "acknowledgments"}
SENTENCE_BREAK = re.compile(
    r"(?<=[.!?])(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\bet al\.)(?<!\bFig\.)(?<!\bvs\.)"
    r"(?<!\bEq\.)(?<!\bSec\.)\s+(?=[A-Z(\[“\"])"
)

# A ruling passage is a section window that starts a little before its sentence.
PASSAGE_CHARACTERS = 2400
PASSAGE_LEAD_CHARACTERS = 350
QUOTE_CHARACTERS = 400


# Longest first, so "ization" is removed before "ation" and "ations" before "s".
SUFFIXES = (
    "ization", "isation", "ations", "ation", "izing", "ized", "izes", "ize", "ings", "ing",
    "ions", "ion", "ences", "ence", "ents", "ent", "ally", "als", "al", "ives", "ive",
    "ers", "er", "ed", "es", "s", "ly",
)


def stem_word(word: str) -> str:
    """Strip up to two inflection endings, keeping at least four letters.

    recurrent and recurrence both become recurr; convolutional and convolutions both
    become convolut; special and specific stay apart, as do extra and extraction.
    """
    for _ in range(2):
        for suffix in SUFFIXES:
            if word.endswith(suffix) and len(word) - len(suffix) >= 4:
                word = word[: -len(suffix)]
                break
        else:
            break
    return word


def same_word(first: str, second: str) -> bool:
    """Match inflections such as recurrent/recurrence and convolutional/convolutions.

    Short words, such as acronyms, are never changed, so they must be identical.
    """
    return first == second or stem_word(first) == stem_word(second)


def chunk_number(key: str) -> int:
    """Stored ids end in their position in the paper: 'doc:1a2b:section:3' or 'doc:1a2b:17'."""
    return int(key.rpartition(":")[2])


def extract_assumed_terms(question: str, names: list[str]) -> tuple[list[str], list[str]]:
    """Split the words of the object the question takes for granted into specific and generic.

    A question outside the known shapes, or one that negates something itself, assumes
    nothing and returns no terms. The paper's own names are removed because they say
    whose design is asked about, not what it is assumed to contain.
    """
    if QUESTION_NEGATION.search(question):
        return [], []
    for frame in OBJECT_FRAMES:
        match = frame.search(question)
        if match:
            assumed = CONTEXT_START.sub("", match["object"])
            break
    else:
        return [], []
    for name in sorted(names, key=len, reverse=True):
        assumed = re.sub(name_pattern(name), " ", assumed, flags=re.IGNORECASE)
    specific, generic = [], []
    for raw in re.findall(r"[A-Za-z][A-Za-z0-9]+", assumed):
        word = raw.lower()
        # Acronyms such as PPO stay although they are short.
        if word in QUESTION_STOPWORDS or ASSUMING_VERB.fullmatch(word):
            continue
        if len(word) < 4 and not raw.isupper():
            continue
        group = generic if word in GENERIC_TERMS else specific
        if word not in group:
            group.append(word)
    return specific, generic


def load_paper_sections(paper: str) -> list[dict]:
    """Read a paper's complete sections in order, with the child chunks that cite them.

    Child chunks overlap and end mid-sentence, so sentences are read from the stored
    section text. References and acknowledgements are left out.
    """
    children: dict[str, list[dict]] = {}
    for row in load_scoped_chunks(paper):
        children.setdefault(row["parent_id"], []).append(row)
    if not children:
        return []
    parents = open_parent_section_store().get(ids=list(children), include=["documents", "metadatas"])
    sections = []
    for parent_id, text, meta in zip(parents["ids"], parents["documents"], parents["metadatas"]):
        heading = re.sub(r"^[\d.\s]+", "", meta.get("heading", "")).strip()
        if heading.casefold() in SKIPPED_SECTIONS:
            continue
        sections.append(
            {
                "parent_id": parent_id,
                "number": chunk_number(parent_id),
                "heading": heading,
                "path": meta.get("section_path") or heading,
                "text": normalize_evidence_text(text),
                "children": sorted(children[parent_id], key=lambda row: chunk_number(row["id"])),
            }
        )
    return sorted(sections, key=lambda row: row["number"])


# Words that may follow an excluded term without starting a different compound.
CONNECTIVES = PREPOSITIONS | {"and", "or", "nor", "also", "even", "altogether"}


def names_excluded_object(term: str, scope: list[list[str]], question_terms: list[str]) -> bool:
    """Whether a cue's scope contains the term as the thing it excludes.

    "Without activation checkpointing" does not rule out an activation function: the term
    is followed by another noun that the question never used, so it belongs to a different
    compound. A connective, an adverb ("convolutions entirely"), a question word or a plain
    head noun such as "layers" after the term leaves it the excluded object.
    """
    for index, chunk in enumerate(scope[:ADJACENT_WORDS]):
        if any(same_word(term, word) for word in chunk):
            follower = scope[index + 1][0] if index + 1 < len(scope) else ""
            return (
                not follower
                or follower in CONNECTIVES
                or follower.endswith("ly")
                or any(same_word(other, follower) for other in [*question_terms, *GENERIC_TERMS])
            )
    return False


def find_ruling_cues(sentence: str, specific: list[str], generic: list[str]) -> list[dict]:
    """List cues in the sentence whose scope contains the assumed object.

    A term counts only after the cue, so "A convolutional layer does not connect all
    positions" does not rule out convolution. At least half of the specific terms must
    be among the first few words after the cue, so a sentence that excludes only part
    of the object ("without access to the set of possible matches" for "access to the
    unknown source encoder") does not count. Generic terms only add to the score.
    """
    found = []
    for cue in RULING_CUE.finditer(sentence):
        if NEGATED_BEFORE.search(sentence[: cue.start()]):
            continue
        rest = sentence[cue.end() :]
        end = SCOPE_END.search(rest)
        # A hyphenated compound such as "left-to-right" is one word of the scope.
        chunks = re.findall(
            r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*", rest[: end.start()] if end else rest
        )
        scope = []
        for chunk in chunks[: SCOPE_WORDS["long" if cue["long"] else "short"]]:
            if chunk.lower() in PREPOSITIONS and not (scope and scope[-1][-1].endswith("ing")):
                break
            scope.append(chunk.lower().split("-"))
        if (
            cue.group().casefold() == "without"
            and scope
            and scope[0][0].endswith("ing")
            and scope[0][0] not in EXCLUDING_GERUNDS
        ):
            continue
        matched_specific = [t for t in specific if names_excluded_object(t, scope, specific + generic)]
        if 2 * len(matched_specific) >= len(specific):
            matched_generic = [
                t for t in generic if any(same_word(t, word) for chunk in scope for word in chunk)
            ]
            found.append(
                {
                    "start": cue.start(),
                    "end": cue.end(),
                    "terms": matched_specific + matched_generic,
                    "specific_count": len(matched_specific),
                }
            )
    return found


def quote_sentence(sentence: str, cue_start: int, cue_end: int) -> str:
    """Quote the whole sentence, or a window around the cue when it is very long."""
    if len(sentence) <= QUOTE_CHARACTERS:
        return sentence
    start = max(0, cue_start - QUOTE_CHARACTERS // 3)
    end = min(len(sentence), cue_end + 2 * QUOTE_CHARACTERS // 3)
    # Snap to word boundaries so the quote does not begin or end inside a word.
    start = sentence.find(" ", start) + 1 if start else 0
    end = sentence.rfind(" ", 0, end) if end < len(sentence) else end
    return ("... " if start else "") + sentence[start:end] + (" ..." if end < len(sentence) else "")


def build_ruling_passage(section: dict, sentence: str, fragment: str) -> dict:
    """Turn the section holding a ruling sentence into an evidence passage.

    Cite the child chunk that holds the start of the cue and its object, falling back
    to the section's first chunk, and give the model the section text around the sentence.
    """
    cited = next(
        (row for row in section["children"] if fragment in normalize_evidence_text(row["content"])),
        section["children"][0],
    )
    text = section["text"]
    if len(text) > PASSAGE_CHARACTERS:
        start = max(0, min(text.find(sentence) - PASSAGE_LEAD_CHARACTERS, len(text) - PASSAGE_CHARACTERS))
        text = text[start : start + PASSAGE_CHARACTERS]
    return {**cited, "content": text}


def find_premise_ruling(question: str, paper: str, papers: list[dict] | None = None) -> dict | None:
    """Return the paper sentence that best rules out what the question assumes, if any.

    The sentence must speak for the paper's own model (naming it, or its authors in the
    abstract, introduction or conclusion), contain a ruling cue such as "without" or
    "dispensing with", and name the assumed object right after that cue. Experiment and
    ablation sections are skipped because they describe variants of the model. Sentences
    in the abstract, introduction and conclusion score higher than the same wording
    elsewhere. Returns the quoted sentence, the matched terms and the evidence passage.
    """
    identity = next((row for row in papers or [] if row["title"] == paper), {"title": paper})
    names = [
        name
        for name in catalog_identity_names(identity)
        if len(name) >= 3 and "_" not in name and not name.endswith(".pdf")
    ]
    specific, generic = extract_assumed_terms(question, names)
    if not specific:
        return None
    paper_name = re.compile("|".join(name_pattern(name) for name in names), re.IGNORECASE) if names else None
    best = None
    for section in load_paper_sections(paper):
        if EXPERIMENT_SECTION.search(section["path"]):
            continue
        headline = bool(HEADLINE_SECTION.search(section["heading"]))
        authors = (
            bool(CLAIMS_SECTION.search(section["heading"])) or section["number"] <= FRONT_MATTER_SECTIONS
        )
        for sentence in SENTENCE_BREAK.split(section["text"]):
            if VARIANT_WORDS.search(sentence):
                continue
            named = bool(paper_name and paper_name.search(sentence))
            # Only the paper's own name, or its authors speaking where they state their
            # design, show the sentence is about the paper's own model.
            if not (named or (authors and SELF_REFERENCE.search(sentence))):
                continue
            for cue in find_ruling_cues(sentence, specific, generic):
                score = cue["specific_count"] + len(cue["terms"]) + named + headline
                # Strictly greater keeps the earliest sentence among equal scores.
                if best is None or score > best["score"]:
                    best = {
                        "score": score,
                        "sentence": quote_sentence(sentence, cue["start"], cue["end"]),
                        "terms": cue["terms"],
                        "passage": build_ruling_passage(
                            section, sentence, sentence[cue["start"] : cue["start"] + 60]
                        ),
                    }
    return best


def format_premise_correction(sentence: str) -> str:
    """Answer a false-premise question with the paper's own ruling-out sentence."""
    return f'This question assumes something the paper rules out. The paper states: "{sentence}"'


def add_premise_corrections(verification: dict, corrections: list[dict]) -> dict:
    """Replace a task's drafted answer with its correction and keep the totals consistent.

    Corrections are shaped like accepted answers: task index, answer and source labels.
    A drafted answer to the same task accepted the false premise, so it is dropped.
    """
    corrected = {row["task_index"] for row in corrections}
    kept = [row for row in verification["accepted"] if row["task_index"] not in corrected]
    accepted = sorted(kept + corrections, key=lambda row: row["task_index"])
    return {
        **verification,
        "accepted": accepted,
        "source_ids": sorted({label for row in accepted for label in row["source_ids"]}),
        "missing_tasks": [i for i in verification["missing_tasks"] if i not in corrected],
    }
