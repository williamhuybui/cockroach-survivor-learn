"""Check that answers state the requested property and come from a named source paper."""

from app import paper_catalog
from app.grounding import (
    DetailAnswer,
    DraftAnswer,
    check_requested_attribute,
    verify_answer_citations,
)


def test_value_from_another_property_cannot_answer_batch_size():
    """A context-size sentence does not state a batch size, although its number is cited."""
    reason = check_requested_attribute(
        "What batch size was used to pre-train Orion 4x2B?",
        "Orion 4x2B is pretrained with a context size of 16k tokens.",
        "Orion is pretrained using a context size of 16k tokens.",
    )
    assert "requested batch size" in reason


def test_training_setting_needs_a_value_beyond_the_model_name():
    """Digits repeated from the model name are not a batch-size value."""
    reason = check_requested_attribute(
        "What batch size was used to pre-train Orion 4x2B?",
        "Orion 4x2B reaches higher throughput at large batch sizes.",
        "Orion reaches higher throughput at large batch sizes.",
    )
    assert "needs an explicit value" in reason


def test_value_stated_for_the_requested_property_passes():
    """A cited sentence that names the property supports a bare value."""
    assert not check_requested_attribute(
        "What peak learning rate does Orion use?",
        "3e-4",
        "Orion uses a peak learning rate of 3e-4 with cosine decay.",
    )


def test_explanations_without_values_are_not_checked():
    """Asking why a setting is used does not require a number."""
    assert not check_requested_attribute(
        "Why does Orion use dropout?", "To reduce overfitting.", ""
    )


def test_result_on_another_benchmark_cannot_answer_the_named_one():
    """A WebShop success rate does not answer a GSM8K question."""
    reason = check_requested_attribute(
        "What does Delta achieve on GSM8K?",
        "Delta improves the success rate by 10%.",
        "On WebShop, Delta improves the success rate by 10%.",
    )
    assert "GSM8K" in reason


def test_only_an_explicit_paper_phrase_names_the_source():
    """'According to the ZETA paper' names ZETA; mentioning ORION-3 does not name its paper."""
    papers = [
        {"title": "04_ZETA_Representations.pdf", "aliases": []},
        {"title": "05_ORION-3_Few_Shot_Learners.pdf", "aliases": []},
    ]
    assert paper_catalog.find_named_source_papers(
        "According to the ZETA paper, how many routers does ORION-3 use?", papers
    ) == [papers[0]["title"]]
    assert not paper_catalog.find_named_source_papers(
        "How many routers does ORION-3 use?", papers
    )


def test_detail_citing_only_another_paper_is_rejected():
    """A fact from ORION-3 cannot answer what the ZETA paper reports."""
    evidence = [
        {
            "id": "S1",
            "title": "05_ORION-3_Few_Shot_Learners.pdf",
            "content": "ORION-3 uses four routers per layer.",
            "location": "page 2",
            "task_ids": [0],
        }
    ]
    draft = DraftAnswer(
        answers=[
            DetailAnswer(
                availability="available",
                source_ids=["S1"],
                answer="ORION-3 uses four routers per layer.",
            )
        ]
    )
    checked = verify_answer_citations(
        draft,
        evidence,
        [{"question": "How many routers does ORION-3 use?"}],
        question="According to the ZETA paper, how many routers does ORION-3 use?",
        source_papers=["04_ZETA_Representations.pdf"],
    )
    assert not checked["accepted"]
    assert "names its source paper" in checked["rejections"][0]["reason"]
