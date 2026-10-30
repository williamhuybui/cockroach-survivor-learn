"""Check explicit availability and named scope using invented facts, not benchmark answers."""

import pytest
from pydantic import ValidationError

from app import grounding
from app.answer_scope import extract_named_constraints
from app.grounding import DetailAnswer, DraftAnswer


def evaluate(question, text, answer="7", availability="available", original=""):
    """Run the real verification and formatting functions without a model call."""
    tasks = [{"question": question, "paper": "Invented.pdf"}]
    evidence = [{"id": "S1", "title": "Invented.pdf", "content": text, "task_ids": [0]}]
    draft = DraftAnswer(
        answers=[
            DetailAnswer(availability=availability, answer=answer, source_ids=["S1"])
        ]
    )
    checked = grounding.verify_answer_citations(
        draft, evidence, tasks, question=original
    )
    return grounding.format_cited_answer(checked, evidence, tasks), checked


def test_availability_is_required_and_nullable_slots_are_rejected():
    """The model must make an explicit choice instead of confusing null with answer text."""
    with pytest.raises(ValidationError):
        DetailAnswer(answer="7", source_ids=["S1"])
    with pytest.raises(ValidationError):
        DetailAnswer(availability="maybe", answer="7", source_ids=["S1"])
    with pytest.raises(ValidationError):
        DraftAnswer(answers=[None])
    missing = DetailAnswer(availability="unavailable", answer="", source_ids=[])
    assert not missing.source_ids


def test_unavailable_record_cannot_make_an_answer_complete():
    """A missing flag wins even if the model adds plausible text and valid citations."""
    result, _ = evaluate(
        "Duration?",
        "Orion uses routing.",
        "Training duration is not reported.",
        "unavailable",
    )
    assert result["status"] == "abstained"
    assert result["sources"] == []
    assert result["detail_answers"][0]["availability"] == "unavailable"


@pytest.mark.parametrize("text", ["null", "NULL", ' "null" ', "", "   "])
def test_literal_missing_values_are_not_supported_answers(text):
    """An available flag cannot turn an empty/null placeholder into a real answer."""
    result, _ = evaluate("Result?", "Orion scores 7.", text)
    assert result["status"] == "abstained"


@pytest.mark.parametrize("question", ["Result on Trial 2023?", "Result on trial-2023?"])
def test_wrong_benchmark_year_is_rejected(question):
    """A copied number from another edition cannot answer the requested edition."""
    result, checked = evaluate(question, "Orion scores 7 on Trial 2020.")
    assert result["status"] == "abstained"
    assert "benchmark" in checked["rejections"][0]["reason"]


def test_matching_benchmark_year_and_following_sentence_are_accepted():
    """The requested result may follow its benchmark name within the same paragraph."""
    result, _ = evaluate(
        "Result on Trial 2020?", "We evaluate on Trial 2020. Orion scores 7."
    )
    assert result["status"] == "answered"


def test_reference_mention_does_not_support_number_from_another_paragraph():
    """An unrelated mention of the right edition cannot rescue a result for another one."""
    result, checked = evaluate(
        "Result on Trial 2023?",
        "Orion scores 7 on Trial 2020.\n\nReferences: Notes on Trial 2023.",
    )
    assert result["status"] == "abstained"
    assert "Numbers absent" in checked["rejections"][0]["reason"]


@pytest.mark.parametrize(
    "question,text",
    [
        ("Parameters in Orion 8x22B?", "Orion 8x7B has 47 billion parameters."),
        ("Parameters in orion-13b?", "Orion-7B uses 7 billion parameters."),
        ("Result from Orion v3?", "Orion v2 scores 7."),
        ("Result from Orion version 3.1?", "Orion version 2.1 scores 7."),
    ],
)
def test_wrong_named_model_variant_is_rejected(question, text):
    """Named sizes and explicit version labels are constraints, not ordinary quantities."""
    result, checked = evaluate(question, text)
    assert result["status"] == "abstained"
    assert "requested model" in checked["rejections"][0]["reason"]


@pytest.mark.parametrize(
    "question,text,answer",
    [
        (
            "Parameters in Orion 8x7B?",
            "Orion 8x7B has 47 billion parameters.",
            "47 billion",
        ),
        ("Result from Orion v2?", "Orion version 2 scores 7.", "7"),
        (
            "Parameters in orion-13b?",
            "Orion 13B uses 13 billion parameters.",
            "13 billion",
        ),
    ],
)
def test_matching_named_model_variants_are_accepted(question, text, answer):
    """Spacing, case and v/version spelling do not create a false mismatch."""
    result, _ = evaluate(question, text, answer)
    assert result["status"] == "answered"


def test_counts_and_general_dates_are_not_benchmark_constraints():
    """List lengths and ordinary dates must not create invented benchmark identities."""
    assert (
        extract_named_constraints("List 3 methods used in 2020 with 13B parameters")
        == set()
    )
    result, _ = evaluate(
        "List 3 methods",
        "Methods include routing, attention and retrieval.",
        "Routing, attention and retrieval.",
    )
    assert result["status"] == "answered"


def test_original_single_question_survives_planner_changed_edition():
    """A planner cannot evade the guard by changing or omitting an original edition."""
    result, _ = evaluate(
        "Result on Trial 2020?",
        "Orion scores 7 on Trial 2020.",
        original="Result on Trial 2023?",
    )
    assert result["status"] == "abstained"
    result, _ = evaluate(
        "Orion score?",
        "Orion scores 7 on Trial 2020.",
        original="Result on Trial 2023?",
    )
    assert result["status"] == "abstained"


def test_mismatched_part_does_not_remove_a_supported_part():
    """Two different requested editions keep their own scope and final availability."""
    tasks = [
        {"question": "Result on Trial 2020?", "paper": "Invented.pdf"},
        {"question": "Result on Trial 2023?", "paper": "Invented.pdf"},
    ]
    evidence = [
        {
            "id": "S1",
            "title": "Invented.pdf",
            "content": "Orion scores 7 on Trial 2020.",
            "task_ids": [0, 1],
        }
    ]
    draft = DraftAnswer(
        answers=[DetailAnswer(availability="available", answer="7", source_ids=["S1"])]
        * 2
    )
    checked = grounding.verify_answer_citations(
        draft,
        evidence,
        tasks,
        question="Give the results on Trial 2020 and Trial 2023.",
    )
    result = grounding.format_cited_answer(checked, evidence, tasks)
    assert result["status"] == "partial"
    assert result["missing_tasks"] == [tasks[1]["question"]]
    assert [part["availability"] for part in result["detail_answers"]] == [
        "available",
        "unavailable",
    ]


def test_generation_has_one_required_availability_record_per_part(monkeypatch):
    """The native model schema accepts empty missing records but requires the flag."""
    calls = []

    def generate(schema, instruction, evidence, request):
        """Inspect the actual generation contract without calling Ollama."""
        calls.append(request)
        with pytest.raises(ValidationError):
            schema(answers=[{"answer": "7", "source_ids": ["S1"]}])
        return schema(
            answers=[{"availability": "unavailable", "answer": "", "source_ids": []}]
        )

    monkeypatch.setattr(grounding, "generate_from_evidence", generate)
    result = grounding.generate_grounded_draft("Result?", [], [{"question": "Result?"}])
    assert len(calls) == 1
    assert result.answers[0].availability == "unavailable"


def test_generation_lists_only_allowed_sources_for_each_task(monkeypatch):
    """Gemma receives the same source-to-task restrictions enforced by Python."""
    captured = []
    monkeypatch.setattr(
        grounding,
        "generate_from_evidence",
        lambda schema, instruction, evidence, request: (
            captured.append(request)
            or schema(
                answers=[
                    {"availability": "unavailable", "answer": "", "source_ids": []}
                ]
                * 2
            )
        ),
    )
    tasks = [{"question": "A?", "paper": "A.pdf"}, {"question": "B?", "paper": "B.pdf"}]
    evidence = [
        {"id": "S1", "title": "A.pdf", "task_ids": [0, 1], "content": "A"},
        {"id": "S2", "title": "B.pdf", "task_ids": [1], "content": "B"},
        {"id": "S3", "title": "A.pdf", "task_ids": [0], "content": "A"},
    ]
    grounding.generate_grounded_draft("Compare A and B", evidence, tasks)
    assert [row["allowed_source_ids"] for row in captured[0]["requested_details"]] == [
        ["S1", "S3"],
        ["S2"],
    ]


def test_none_can_be_a_real_available_answer():
    """An explicit availability flag distinguishes a real 'None' answer from a null placeholder."""
    result, _ = evaluate(
        "Which extra tools are used?", "No extra tools are used.", "None"
    )
    assert result["status"] == "answered"


def test_currency_question_cannot_be_answered_with_a_different_measurement():
    """Compute time is not a monetary amount even if it occurs in the source."""
    result, checked = evaluate(
        "What was the training cost in dollars?",
        "The training required 80 compute hours.",
        "The training required 80 compute hours.",
    )
    assert result["status"] == "abstained"
    assert "currency" in checked["rejections"][0]["reason"]


@pytest.mark.parametrize(
    "question,source,answer",
    [
        ("What was the cost in dollars?", "The cost was $45.", "$45"),
        ("What was the price in euros?", "The price was 25 euros.", "25 EUR"),
        ("How much in yen?", "The price was 30 JPY.", "30 yen"),
    ],
)
def test_explicit_currency_remains_valid_when_answer_and_source_match(
    question, source, answer
):
    """Equivalent currency symbols and names are accepted when the amount is cited."""
    result, _ = evaluate(question, source, answer)
    assert result["status"] == "answered"


def test_currency_is_not_supported_by_a_source_without_that_currency():
    """An answer cannot claim a monetary value from evidence with unrelated units."""
    result, _ = evaluate(
        "What was the cost in dollars?", "Training took 45 hours.", "$45"
    )
    assert result["status"] == "abstained"


def test_explaining_a_currency_symbol_does_not_require_a_monetary_amount():
    """A question about a symbol itself should not trigger an amount check."""
    result, _ = evaluate(
        "What does the dollar symbol mean?",
        "The dollar symbol identifies US currency.",
        "It identifies US currency.",
    )
    assert result["status"] == "answered"


@pytest.mark.parametrize(
    "source",
    [
        "Our 65B-parameter model is competitive with the largest models.",
        "Orion-13B beats smaller baselines. Our 65B-parameter model is competitive.",
        "The Orion models range from 7B to 65B parameters, with competitive performance.",
    ],
)
def test_contextual_model_size_does_not_require_repeated_name(source):
    """A correctly scoped paper may refer to its own model without repeating its name."""
    result, _ = evaluate("How large is Orion-65B?", source, "65 billion parameters")
    assert result["status"] == "answered"


def test_another_named_models_size_cannot_excuse_an_explicit_conflict():
    """A matching number attached to a different named model is not contextual support."""
    result, checked = evaluate(
        "How large is Orion-65B?",
        "Orion-13B has 13 billion parameters. Sirius-65B has 65 billion parameters.",
        "65 billion parameters",
    )
    assert result["status"] == "abstained"
    assert "contradicts" in checked["rejections"][0]["reason"]


def test_contextual_version_reference_is_allowed():
    """The paper can introduce a later version through 'our version' wording."""
    result, _ = evaluate(
        "How does Orion v3 score?", "Orion v2 scored less. Our version 3 scores 7.", "7"
    )
    assert result["status"] == "answered"


def test_contextual_size_does_not_bypass_numeric_support():
    """Relaxing the name requirement cannot introduce numbers absent from the source."""
    result, checked = evaluate(
        "How large is Orion-65B?",
        "Our 65B-parameter model is competitive.",
        "99 billion parameters",
    )
    assert result["status"] == "abstained"
    assert "Numbers absent" in checked["rejections"][0]["reason"]


def test_supported_contextual_part_survives_an_invalid_citation_on_another_part():
    """A valid contextual answer remains useful when a separate part cites the wrong task."""
    tasks = [
        {"question": "How large is Orion-65B?", "paper": "A.pdf"},
        {"question": "How large is Sirius?", "paper": "B.pdf"},
    ]
    evidence = [
        {
            "id": "S1",
            "title": "A.pdf",
            "content": "Orion-13B is small. Our 65B model is competitive.",
            "task_ids": [0],
        }
    ]
    draft = DraftAnswer(
        answers=[
            DetailAnswer(
                availability="available",
                answer="65 billion parameters",
                source_ids=["S1"],
            ),
            DetailAnswer(
                availability="available",
                answer="65 billion parameters",
                source_ids=["S1"],
            ),
        ]
    )
    checked = grounding.verify_answer_citations(draft, evidence, tasks)
    result = grounding.format_cited_answer(checked, evidence, tasks)
    assert result["status"] == "partial"
    assert result["missing_tasks"] == [tasks[1]["question"]]
    assert [row["id"] for row in result["sources"]] == ["S1"]


@pytest.mark.parametrize("extra_source", [False, True])
def test_joint_detail_can_cite_both_verified_papers_but_not_a_third(extra_source):
    """A comparison can join its verified sources while retaining the paper allow-list."""
    tasks = [{"question": "Compare the two models", "paper": ["A.pdf", "B.pdf"]}]
    evidence = [
        {
            "id": "S1",
            "title": "A.pdf",
            "content": "The first model has 65 billion parameters.",
            "task_ids": [0],
        },
        {
            "id": "S2",
            "title": "B.pdf",
            "content": "The second model has 70 billion parameters.",
            "task_ids": [0],
        },
        {
            "id": "S3",
            "title": "C.pdf",
            "content": "Unrelated model facts.",
            "task_ids": [0],
        },
    ]
    ids = ["S1", "S2"] + (["S3"] if extra_source else [])
    draft = DraftAnswer(
        answers=[
            DetailAnswer(
                availability="available",
                answer="65 billion versus 70 billion parameters",
                source_ids=ids,
            )
        ]
    )
    checked = grounding.verify_answer_citations(draft, evidence, tasks)
    result = grounding.format_cited_answer(checked, evidence, tasks)
    assert result["status"] == "answered"
    assert result["detail_answers"][0]["source_ids"] == ["S1", "S2"]
