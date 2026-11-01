"""Check that answers state the requested property and come from a named source paper."""

import pytest

from app import paper_catalog
from app.grounding import (
    DetailAnswer,
    DraftAnswer,
    check_named_kind,
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


def test_value_written_as_a_word_counts_when_the_text_states_it():
    """'A single epoch' is an explicit value; the cited sentence says so too."""
    question = "For how many epochs was the Orion reward model trained?"
    assert not check_requested_attribute(
        question, "The reward model was trained for a single epoch.", "We trained for a single epoch over the data."
    )
    assert not check_requested_attribute(
        question, "It was trained for two epochs.", "We trained the reward model for 2 epochs."
    )


def test_value_written_as_a_word_must_be_stated_by_the_text():
    """A spelled-out value that the cited sentence contradicts is still rejected."""
    reason = check_requested_attribute(
        "For how many epochs was the Orion reward model trained?",
        "The reward model was trained for a single epoch.",
        "We trained the reward model for 16 epochs.",
    )
    assert "spelled-out value" in reason


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


TWO_CLAUSE_EVIDENCE = [
    {
        "id": "S1",
        "title": "05_ORION-3_Few_Shot_Learners.pdf",
        "content": "ORION-3 is fine-tuned with PPO.",
        "location": "page 2",
        "task_ids": [0],
    },
    {
        "id": "S2",
        "title": "04_ZETA_Representations.pdf",
        "content": "We introduce ZETA routing, a variant of PPO.",
        "location": "page 1",
        "task_ids": [1],
    },
]


def two_clause_check(question, second_paper):
    tasks = [
        {"question": "Which algorithm does ORION-3 use?", "paper": "05_ORION-3_Few_Shot_Learners.pdf", "scope": "paper"},
        {"question": "Which algorithm does the ZETA paper introduce?", "paper": second_paper, "scope": "paper"},
    ]
    evidence = [{**row, "title": second_paper} if row["id"] == "S2" else row for row in TWO_CLAUSE_EVIDENCE]
    draft = DraftAnswer(
        answers=[
            DetailAnswer(availability="available", source_ids=["S1"], answer="ORION-3 uses PPO."),
            DetailAnswer(availability="available", source_ids=["S2"], answer="ZETA routing."),
        ]
    )
    return verify_answer_citations(
        draft, evidence, tasks, question=question, source_papers=["04_ZETA_Representations.pdf"]
    )


def test_paper_named_in_one_clause_does_not_govern_another_clause():
    """'..., and which algorithm does the ZETA paper introduce?' lets the ORION-3 task cite ORION-3."""
    question = "Which algorithm does ORION-3 use, and which algorithm does the ZETA paper introduce?"
    checked = two_clause_check(question, "04_ZETA_Representations.pdf")
    assert [row["task_index"] for row in checked["accepted"]] == [0, 1]


def test_framing_source_paper_governs_every_task():
    """'According to the ZETA paper, ...' still needs a ZETA citation for each part."""
    question = "According to the ZETA paper, which algorithm does ORION-3 use and which does it introduce?"
    checked = two_clause_check(question, "04_ZETA_Representations.pdf")
    assert [row["task_index"] for row in checked["accepted"]] == [1]
    assert "names its source paper" in checked["rejections"][0]["reason"]


def test_value_of_another_setting_cannot_answer_an_architecture_question():
    """A 20,480-token generation limit is not a vocabulary size; '126 layers' is not a head count."""
    generation = "The maximum number of tokens for generation is set to 20,480 tokens."
    assert "vocabulary" in check_requested_attribute(
        "What vocabulary size does Orion's tokenizer have?", "20480", generation
    )
    assert "attention heads" in check_requested_attribute(
        "How many attention heads does Orion use?", "126", "Orion uses an architecture with 126 layers."
    )


@pytest.mark.parametrize(
    "question,answer,evidence",
    [
        ("What vocabulary size does Orion use?", "128K tokens", "We use a vocabulary with 128K tokens."),
        ("What context length does Orion support?", "up to 128K tokens", "Orion has a context window of up to 128K tokens."),
        ("How many layers does Orion have?", "126", "Orion uses an architecture with 126 layers."),
        ("How many attention heads does Orion use?", "128", "It has 126 layers and 128 attention heads."),
        ("What sampling temperature does Orion use?", "1", "We set the sampling temperature to 1 for rollout."),
    ],
)
def test_architecture_value_stated_for_its_setting_is_accepted(question, answer, evidence):
    assert check_requested_attribute(question, answer, evidence) == ""


@pytest.mark.parametrize(
    "question,answer,label",
    [
        ("Which optimizer does Orion use?", "Orion uses an attention mechanism without recurrence.", "optimizer"),
        ("What hardware was Orion trained on?", "Orion was trained as a bidirectional encoder.", "hardware"),
        ("Which tokenizer does Orion use?", "Orion uses a sparse mixture of experts.", "tokenizer"),
        ("What positional encoding does Orion use?", "Orion uses multi-head attention in three ways.", "positional encoding"),
        ("How does Orion encode token positions?", "Orion embeddings come from the 16th decoder layer.", "positional encoding"),
        ("Which normalization layer does Orion use?", "Orion's text normalization (TN) module reads speech.", "normalization"),
        ("Which activation function does Orion use?", "Orion uses a bidirectional Transformer.", "activation"),
    ],
)
def test_answer_must_name_the_asked_kind(question, answer, label):
    """An answer about something else names no optimizer, GPU, tokenizer and so on."""
    assert check_named_kind(question, answer) == f"The answer names no {label}."


@pytest.mark.parametrize(
    "question,answer",
    [
        ("On which GPU was Orion's latency measured?", "An NVIDIA Quadro RTX8000"),
        ("What kind of hardware did Orion's training run on?", "Many CPU cores in a data center"),
        ("Which optimizer does Orion use?", "Minibatch SGD together with Adadelta"),
        ("Which optimizer does Orion use?", "Stochastic gradient descent with backpropagation"),
        ("Which optimizer does Orion use?", "AdamW with weight decay 0.1"),
        ("What tokenizer does Orion use?", "The byte- pair encoding (BPE) algorithm from SentencePiece"),
        ("What positional encoding does Orion use?", "RoPE"),
        ("What positional encoding does Orion use?", "Sine and cosine functions of different frequencies"),
        ("Which normalization does Orion use?", "Pre-normalization with RMSNorm"),
        ("Which normalization does Orion use?", "Layer normalization after each residual connection"),
        ("Which activation function does Orion use?", "GELU rather than ReLU"),
    ],
)
def test_answer_naming_an_instance_of_the_asked_kind_passes(question, answer):
    """Aliases, acronyms and long forms all name the kind; this does not prove the citation supports it."""
    assert check_named_kind(question, answer) == ""


@pytest.mark.parametrize(
    "question,answer",
    [
        ("Which optimizer and learning rate does Orion use?", "A learning rate of 1e-6"),
        ("Which optimizer does Orion use, and how many GPUs does Sirius use?", "Sirius uses 8 GPUs."),
        ("What beta2 value does the Adam optimizer use in Orion?", "0.95"),
        ("How many GPUs was Orion trained on?", "8"),
    ],
)
def test_partial_answers_and_value_questions_are_not_kind_checked(question, answer):
    """A longer task may be answered in part, and a question about a setting's value needs no name."""
    assert check_named_kind(question, answer) == ""


def test_measurement_values_must_be_stated_for_the_measurement():
    """13B active parameters is not a latency; a 1.3B model size is not a number of raters."""
    assert "latency" in check_requested_attribute(
        "What is Orion's inference latency per token?",
        "13B active parameters per token",
        "Orion uses only 13B active parameters during inference.",
    )
    assert "raters" in check_requested_attribute(
        "How many raters evaluated Orion?",
        "1.3B parameters",
        "Outputs from the 1.3B parameter Orion model are preferred.",
    )
    assert not check_requested_attribute(
        "How many labelers did the Orion authors hire?", "About 40", "We hired a team of about 40 contractors."
    )
    assert "needs an explicit value" in check_requested_attribute(
        "What random seed did the Orion authors use?", "Orion sweeps the KL coefficient.", "We sweep the KL coefficient."
    )


def test_latency_answer_without_a_value_and_gpu_without_the_word_gpu_pass():
    """'No additional inference latency' is a full answer; 'Quadro RTX8000' names a GPU."""
    assert not check_requested_attribute(
        "How much additional inference latency does Orion add?",
        "No additional inference latency",
        "Unlike adapters, Orion adds no additional inference latency.",
    )
    assert not check_requested_attribute(
        "On which GPU did the Orion authors measure throughput?",
        "An NVIDIA Quadro RTX8000",
        "We use an NVIDIA Quadro RTX8000.",
    )


def test_task_listing_several_properties_skips_the_property_check():
    """One sentence rarely names corpus, batch size and optimizer together."""
    assert not check_requested_attribute(
        "What setup does Orion use in terms of corpus, batch size and optimizer?",
        "OrionText, batches of 256 sequences, 100,000 steps.",
        "Orion trains on OrionText with batches of 256 sequences for 100,000 steps.",
    )


def test_value_for_none_of_several_requested_properties_is_rejected():
    """A context size cannot answer a 'batch size and learning rate' task."""
    reason = check_requested_attribute(
        "What batch size and peak learning rate were used to pre-train Orion 4x2B?",
        "Orion retrieves information from its context window of 32k tokens.",
        "Orion retrieves information from its context window of 32k tokens.",
    )
    assert "any requested property" in reason


def test_rate_given_as_a_total_is_rejected_but_rates_and_totals_pass():
    """'each trillion tokens requires 90K GPU hours' is not the stage's 1200K total."""
    from app.grounding import check_rate_given_as_total

    evidence = "Training Orion on each trillion tokens requires only 90K GPU hours. Pre-training costs 1200K GPU hours."
    question = "How many GPU hours did Orion's pre-training take?"
    assert "per unit" in check_rate_given_as_total(question, "Pre-training took 90K GPU hours.", evidence)
    assert not check_rate_given_as_total(question, "Pre-training took 1200K GPU hours.", evidence)
    assert not check_rate_given_as_total(question, "90K GPU hours per trillion tokens.", evidence)
    assert not check_rate_given_as_total("How many GPU hours per trillion tokens?", "90K GPU hours.", evidence)
    # A rate with no competing total, such as '37B activated for each token', passes.
    assert not check_rate_given_as_total("How many parameters are activated?", "37B activated.",
                                         "Orion has 600B parameters with 37B activated for each token.")


def test_words_like_over_do_not_make_a_shared_unit():
    """'from 70M to over 10B, each for four schedules' is no rate of '70 million to over 16 billion'.

    Only a shared unit word ('GPU hours') ties a per-unit value to a total; a preposition
    such as 'over' that follows both numbers names no unit.
    """
    from app.grounding import check_rate_given_as_total

    evidence = (
        "We launched Orion models going from 70M to over 10B parameters, each for four cosine cycles. "
        "Across approaches we trained models from 70 million to over 16 billion parameters. "
        "Approach 2 used models up to 10B."
    )
    question = "What range of model sizes did the Orion paper train?"
    assert not check_rate_given_as_total(question, "Orion trained models from 70M to 10B parameters.", evidence)


def test_equal_scaled_numbers_are_supported():
    """'2.5 billion' restates '2,500M'; a different value stays unsupported."""
    evidence = "Orion reads OrionText (2,500M words) and 1.4 trillion tokens."
    assert not find_unsupported_numbers("2.5 billion words and 1.4T tokens", evidence)
    assert find_unsupported_numbers("3.1 billion words", evidence) == {"3.1"}
    assert not find_unsupported_numbers("1,400,000,000,000 tokens", evidence)
    assert find_unsupported_numbers("1,500,000,000,000 tokens", evidence) == {"1500000000000"}


def test_spelled_out_numbers_in_the_source_support_digits():
    """'eight P100 GPUs' supports an answer that writes 8; a different value does not."""
    evidence = "Orion trains for twelve hours on eight P100 GPUs."
    assert not find_unsupported_numbers("Orion used 8 P100 GPUs for 12 hours.", evidence)
    assert find_unsupported_numbers("Orion used 16 P100 GPUs.", evidence) == {"16"}
    # Answer words are not read as values, so 'one of' needs no support.
    assert not find_unsupported_numbers("Orion is one of the larger models.", "Orion is large.")


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


def test_no_evidence_abstention_uses_the_gateway_wording():
    """With nothing retrieved, answer exactly as the evaluation gateway's contract asks."""
    from app import services

    state = {"evidence": [], "tasks": [], "retrieval_trace": {"tasks": []}}
    result = services.build_chat_response(state)["result"]
    assert result["answer"] == "I don't know - this is not in the documents."
    assert result["status"] == "abstained" and result["sources"] == []
