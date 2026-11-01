"""Check that a question's false premise is corrected by its own paper's wording.

A temporary index holds the invented paper "Orion". The tests cover which sentences
count as ruling something out, which look similar but must not, and how a correction
replaces a drafted answer without another model call.
"""

import pytest
from answer_fixtures import cited_draft

from app import grounding, premise_search, question_planning, services
from app.paper_store import open_chunk_store, open_parent_section_store

PAPERS = [{"title": "Orion.pdf", "aliases": ["Orion"]}]
ABSTRACT = (
    "We present Orion, a sequence model built on gated memory cells, dispensing with "
    "recurrent layers entirely. Orion also avoids hand-built feature extractors. "
    "Our experiments cover translation."
)
SECTIONS = [
    ("Abstract", "Abstract", ABSTRACT),
    (
        "3 Method",
        "3 Method",
        "A recurrent layer does not use gated memory cells. Orion reads tokens through "
        "gated memory cells. Orion does not use supervision on retrieved evidence.",
    ),
    (
        "5.1 Gating",
        "5 Ablation Studies > 5.1 Gating",
        "No Gating: Orion trained without gated memory cells to measure their effect.",
    ),
    ("References", "References", "Orion with no hypernetworks. Proceedings of Orion, 2020."),
]


@pytest.fixture
def orion_index():
    """Store each section once as parent text and once as the child chunk that cites it."""
    ids = [f"orion:section:{number}" for number in range(len(SECTIONS))]
    open_parent_section_store().add(
        ids=ids,
        documents=[text for _, _, text in SECTIONS],
        metadatas=[{"heading": heading, "section_path": path} for heading, path, _ in SECTIONS],
        embeddings=[[1.0]] * len(SECTIONS),
    )
    open_chunk_store().add(
        ids=[f"orion:chunk:{number}" for number in range(len(SECTIONS))],
        documents=[
            f"Paper: Orion.pdf\nSection: {heading}\n{text}" for heading, _, text in SECTIONS
        ],
        embeddings=[[1.0, 0.0]] * len(SECTIONS),
        metadatas=[
            {
                "title": "Orion.pdf",
                "parent_id": parent_id,
                "source_url": "https://example.com/orion",
                "location": f"page {number + 1}",
                "document_type": "sectioned_paper",
            }
            for number, parent_id in enumerate(ids)
        ],
    )


def orion_passage(key, text, page):
    """A retrieved Orion chunk, shaped as the retriever returns it."""
    return {
        "id": f"orion:{key}",
        "content": text,
        "title": "Orion.pdf",
        "source_url": "https://example.com/orion",
        "location": f"page {page}",
        "document_type": "sectioned_paper",
    }


def test_assumed_terms_come_from_the_object_of_the_question():
    names = ["Orion"]
    extract = premise_search.extract_assumed_terms
    assert extract("Why does Orion use recurrent layers in its encoder?", names) == (
        ["recurrent"],
        ["layers"],
    )
    assert extract("Which reward model does Orion train before optimizing the policy?", names) == (
        ["reward"],
        ["model"],
    )
    assert extract("How large is the critic model that Orion trains alongside the policy?", names) == (
        ["critic"],
        ["model"],
    )


@pytest.mark.parametrize(
    "question",
    [
        "Does Orion use recurrent layers?",  # a yes/no question assumes nothing
        "Why does Orion not use recurrent layers?",  # already negated
        "What does Orion eliminate compared with recurrent layers?",  # asks about the exclusion
        "How many tokens did Orion use?",  # a quantity, not a component
        "Where does Orion store its recurrent layers?",
    ],
)
def test_questions_that_assume_nothing_yield_no_specific_terms(question):
    """Without a specific term the search does not run, whatever generic words remain."""
    assert premise_search.extract_assumed_terms(question, ["Orion"])[0] == []


def test_words_match_by_stem_not_by_prefix():
    same = premise_search.same_word
    assert same("recurrent", "recurrence")
    assert same("convolutional", "convolutions")
    assert same("modeling", "model")
    assert not same("extraction", "extra")
    assert not same("special", "specific")


@pytest.mark.parametrize(
    "sentence, terms",
    [
        ("We build it dispensing with recurrence and convolutions entirely.", ["recurrent"]),
        ("Orion works without explicit reward modeling or search.", ["reward"]),
        ("Orion foregoes the critic model, estimating a baseline from group scores.", ["critic"]),
        ("Orion is free of recurrent layers.", ["recurrent"]),
        ("Orion does not use a critic during training.", ["critic"]),
        ("Our model contains no recurrence and no convolution.", ["recurrent"]),
        ("We train without resorting to proprietary datasets.", ["proprietary"]),
    ],
)
def test_cue_rules_out_the_object_that_follows_it(sentence, terms):
    assert premise_search.find_ruling_cues(sentence, terms, [])


@pytest.mark.parametrize(
    "sentence, terms",
    [
        # The term is the subject, before the cue.
        ("A recurrent layer does not use attention.", ["recurrent"]),
        # "Without compromising X" keeps X.
        ("Orion keeps quality without compromising labeler scores.", ["labeler"]),
        # The term sits in a prepositional phrase after the excluded object.
        ("Orion does not use supervision on retrieved evidence.", ["retriever"]),
        # The cue itself is negated.
        ("Other interventions do not eliminate recurrence in practice.", ["recurrent"]),
        # Both settings are described.
        ("We compare Orion with and without recurrence.", ["recurrent"]),
        # "Without being X" keeps X, and the term is far from the cue anyway.
        ("Orion routes messages without being blocked by arriving tokens.", ["arriving"]),
        # The term is the cue's sixth word, too far to be what it excludes.
        ("Orion runs without any extra heavy slow arriving tokens.", ["tokens"]),
        # Only one of four specific terms is excluded.
        ("We locate keys without access to the set of matches.", ["access", "unknown", "source", "encoder"]),
    ],
)
def test_wording_that_keeps_the_object_is_not_a_ruling(sentence, terms):
    assert premise_search.find_ruling_cues(sentence, terms, []) == []


def test_term_that_starts_a_different_compound_is_not_the_excluded_object():
    """'Without activation checkpointing' does not say a model has no activation function."""
    sentence = "We pre-train on long sequences without activation checkpointing."
    assert premise_search.find_ruling_cues(sentence, ["activation"], ["function"]) == []
    # The same words in the excluded object still count once the question uses the next word too.
    assert premise_search.find_ruling_cues(sentence, ["activation"], ["checkpointing"])
    # An adverb or a connective after the term ends the object.
    assert premise_search.find_ruling_cues("We build it without activation entirely.", ["activation"], [])
    assert premise_search.find_ruling_cues("We build it without activation or search.", ["activation"], [])


def test_hyphenated_compound_stays_one_word_of_the_scope():
    sentence = "We do not use traditional left-to-right language models."
    cues = premise_search.find_ruling_cues(sentence, ["left", "right", "objective"], ["language", "models"])
    assert cues and cues[0]["terms"][:2] == ["left", "right"]


def test_long_sentence_is_quoted_around_the_cue():
    sentence = "Opening words " * 20 + "without recurrence" + " closing words" * 40
    cue = sentence.index("without")
    quote = premise_search.quote_sentence(sentence, cue, cue + len("without"))
    assert "without recurrence" in quote
    assert quote.startswith("... ") and quote.endswith(" ...")
    assert len(quote) <= premise_search.QUOTE_CHARACTERS + 20


def test_ruling_sentence_comes_from_the_abstract_and_cites_its_chunk(orion_index):
    ruling = premise_search.find_premise_ruling(
        "Why does Orion use recurrent layers in its encoder?", "Orion.pdf", PAPERS
    )
    assert ruling["sentence"] == (
        "We present Orion, a sequence model built on gated memory cells, dispensing with "
        "recurrent layers entirely."
    )
    assert ruling["passage"]["id"] == "orion:chunk:0"
    assert ruling["sentence"] in ruling["passage"]["content"]


def test_chunks_are_read_in_paper_order_without_a_stored_child_index(orion_index):
    """Ingest stores no child index; ids end in the chunk's position, so 9 precedes 12."""
    open_chunk_store().add(
        ids=["orion:chunk:12", "orion:chunk:9"],
        documents=["Later text.", "Earlier text."],
        embeddings=[[1.0, 0.0]] * 2,
        metadatas=[{"title": "Orion.pdf", "parent_id": "orion:section:0"}] * 2,
    )
    abstract = premise_search.load_paper_sections("Orion.pdf")[0]
    assert [row["id"] for row in abstract["children"]] == ["orion:chunk:0", "orion:chunk:9", "orion:chunk:12"]


def test_ruling_found_in_the_method_when_it_names_the_paper(orion_index):
    ruling = premise_search.find_premise_ruling(
        "Which feature extractors does Orion use?", "Orion.pdf", PAPERS
    )
    assert ruling["sentence"] == "Orion also avoids hand-built feature extractors."


@pytest.mark.parametrize(
    "question",
    [
        # The abstract names the cells before the cue, a baseline's sentence is not the
        # paper's own, and the ablation describes a variant.
        "Why does Orion use gated memory cells?",
        # The method excludes supervision, not retrieval.
        "Which retriever does Orion use?",
        # Only a reference entry mentions this.
        "Why does Orion use hypernetworks?",
    ],
)
def test_similar_wording_elsewhere_does_not_rule_out_a_supported_premise(orion_index, question):
    assert premise_search.find_premise_ruling(question, "Orion.pdf", PAPERS) is None


def test_unindexed_paper_has_no_ruling():
    assert premise_search.find_premise_ruling("Why does Orion use recurrent layers?", "Missing.pdf", PAPERS) is None


def test_correction_replaces_the_drafted_answer_and_fixes_the_totals():
    verification = {
        "accepted": [
            {"task_index": 0, "answer": "Orion uses recurrent layers.", "source_ids": ["S2"]},
            {"task_index": 1, "answer": "28.4 BLEU", "source_ids": ["S3"]},
        ],
        "source_ids": ["S2", "S3"],
        "rejected_count": 0,
        "rejections": [],
        "missing_tasks": [2],
    }
    corrections = [
        {"task_index": 0, "answer": "Correction.", "source_ids": ["S1"]},
        {"task_index": 2, "answer": "Another.", "source_ids": ["S4"]},
    ]
    result = premise_search.add_premise_corrections(verification, corrections)
    assert [(row["task_index"], row["answer"], row["source_ids"]) for row in result["accepted"]] == [
        (0, "Correction.", ["S1"]),
        (1, "28.4 BLEU", ["S3"]),
        (2, "Another.", ["S4"]),
    ]
    assert result["source_ids"] == ["S1", "S3", "S4"]
    assert result["missing_tasks"] == []


def stub_orion_graph(monkeypatch, question, retrieved=lambda query: []):
    """Plan one Orion task, retrieve nothing unless asked, and share the temporary index."""
    monkeypatch.setattr(services, "list_indexed_papers", lambda: PAPERS)
    monkeypatch.setattr(
        question_planning,
        "request_structured_response",
        lambda schema, instruction, payload: schema(
            tasks=[{"question": question, "paper": "Orion.pdf", "scope": "paper"}]
        ),
    )
    monkeypatch.setattr(
        services, "retrieve_relevant_passages", lambda query, **kwargs: retrieved(query)
    )


def test_false_premise_is_corrected_without_calling_the_answer_model(orion_index, monkeypatch):
    question = "Why does Orion use recurrent layers in its encoder?"
    stub_orion_graph(monkeypatch, question)
    monkeypatch.setattr(
        grounding, "generate_from_evidence", lambda *args: pytest.fail("No answer model call expected")
    )
    result = services.answer_research_question(question)
    assert result["status"] == "answered"
    assert result["answer"].startswith("This question assumes something the paper rules out.")
    assert "dispensing with recurrent layers entirely" in result["answer"]
    assert result["answer"].endswith("[S1]")
    assert [row["title"] for row in result["sources"]] == ["Orion.pdf"]
    assert result["retrieval_trace"]["tasks"][0]["trace"]["premise_ruling"]["terms"] == ["recurrent", "layers"]
    assert "generation" not in result["timings"]


def test_correction_replaces_only_the_task_that_assumed_the_premise(orion_index, monkeypatch):
    question = "Why does Orion use recurrent layers, and what BLEU does Orion reach?"
    bleu = orion_passage("bleu", "Orion reaches 28.4 BLEU on the translation task.", 4)
    # Explicit clauses are planned without the model, so only retrieval is stubbed.
    stub_orion_graph(monkeypatch, question, lambda query: [bleu] if "BLEU" in query else [])
    draft = cited_draft("Orion uses recurrent layers for speed. [S1]\n\nOrion reaches 28.4 BLEU. [S2]")
    monkeypatch.setattr(grounding, "generate_from_evidence", lambda *args: draft)
    result = services.answer_research_question(question)
    assert result["status"] == "answered"
    assert "dispensing with recurrent layers entirely" in result["answer"]
    assert "Orion reaches 28.4 BLEU." in result["answer"]
    assert "for speed" not in result["answer"]


def test_supported_question_is_answered_by_the_model_as_before(orion_index, monkeypatch):
    question = "Why does Orion use gated memory cells?"
    cells = orion_passage("cells", "Gated memory cells let Orion keep long-range context.", 3)
    stub_orion_graph(monkeypatch, question, lambda query: [cells])
    draft = cited_draft("Gated memory cells let Orion keep long-range context. [S1]")
    monkeypatch.setattr(grounding, "generate_from_evidence", lambda *args: draft)
    result = services.answer_research_question(question)
    assert result["answer"].startswith("Gated memory cells let Orion keep long-range context.")
    assert "rules out" not in result["answer"]
