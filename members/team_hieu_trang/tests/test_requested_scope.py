"""Check that answers state the requested property and come from a named source paper."""

from app import paper_catalog
from app.grounding import (
    DetailAnswer,
    DraftAnswer,
    check_requested_attribute,
    find_unsupported_numbers,
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


def test_task_listing_several_properties_skips_the_property_check():
    """One sentence rarely names corpus, batch size and optimizer together."""
    assert not check_requested_attribute(
        "What setup does Orion use in terms of corpus, batch size and optimizer?",
        "OrionText, batches of 256 sequences, 100,000 steps.",
        "Orion trains on OrionText with batches of 256 sequences for 100,000 steps.",
    )


def test_equal_scaled_numbers_are_supported():
    """'2.5 billion' restates '2,500M'; a different value stays unsupported."""
    evidence = "Orion reads OrionText (2,500M words) and 1.4 trillion tokens."
    assert not find_unsupported_numbers("2.5 billion words and 1.4T tokens", evidence)
    assert find_unsupported_numbers("3.1 billion words", evidence) == {"3.1"}


def test_number_cited_from_the_wrong_label_is_found_in_the_same_task():
    """A later same-task passage can supply the number Gemma cited from S1."""
    evidence = [
        {"id": "S1", "title": "A.pdf", "content": "Orion evaluates several open models.", "task_ids": [0]},
        {"id": "S2", "title": "A.pdf", "content": "The open models are Kite-30B and Wren-13B.", "task_ids": [0]},
        {"id": "S3", "title": "B.pdf", "content": "Delta uses Kite-70B.", "task_ids": [1]},
    ]
    draft = DraftAnswer(
        answers=[DetailAnswer(availability="available", source_ids=["S1"], answer="Kite-30B and Wren-13B.")]
    )
    checked = verify_answer_citations(
        draft, evidence, [{"question": "Which open models does Orion evaluate?", "paper": "A.pdf"}]
    )
    assert checked["accepted"][0]["source_ids"] == ["S1", "S2"]


def test_number_found_only_in_another_task_is_still_rejected():
    """Another task's passage cannot supply a number."""
    evidence = [
        {"id": "S1", "title": "A.pdf", "content": "Orion evaluates several open models.", "task_ids": [0]},
        {"id": "S3", "title": "B.pdf", "content": "Delta uses Kite-70B.", "task_ids": [1]},
    ]
    draft = DraftAnswer(
        answers=[DetailAnswer(availability="available", source_ids=["S1"], answer="Kite-70B.")]
    )
    checked = verify_answer_citations(
        draft, evidence, [{"question": "Which model does Orion evaluate?", "paper": "A.pdf"}]
    )
    assert not checked["accepted"]


def test_et_al_names_the_paper_whose_header_lists_that_first_author():
    """A surname right after the title counts; a citation further down does not."""
    papers = [
        {"title": "01_Orion.pdf", "aliases": [], "author_header": "Orion Scaling Ada Kepler Bo Lin"},
        {"title": "02_Delta.pdf", "aliases": [], "author_header": "Delta Wren Cho " + "x " * 100 + "Kepler et al. 2021"},
    ]
    refs = paper_catalog.find_paper_references("What does Kepler et al. report?", papers)
    assert refs["candidates"] == ["01_Orion.pdf"]
    assert not paper_catalog.find_paper_references("What does Lin et al. report?", papers)["titles"]


def test_question_words_do_not_match_a_title_that_starts_with_them():
    """'How does dropout help?' is not about a paper titled 'Does Orion Work'."""
    papers = [{"title": "03_Does_Orion_Really_Work.pdf", "aliases": []}]
    assert not paper_catalog.find_paper_references("How does dropout help?", papers)["candidates"]


def test_every_abstention_says_i_dont_know():
    """The judge counts only explicit refusals as idk; 'No matching evidence' alone can read as an answer."""
    from app.grounding import build_abstention_response

    assert build_abstention_response("No matching research-paper evidence was found.")["answer"].startswith("I don't know")
    assert build_abstention_response("I don't know - x.")["answer"] == "I don't know - x."
