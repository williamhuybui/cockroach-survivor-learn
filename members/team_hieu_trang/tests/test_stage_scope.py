"""Check stage binding with invented facts: a value must belong to the stage the question names."""

import pytest

from app import grounding
from app.grounding import DetailAnswer, DraftAnswer
from app.stage_scope import find_stage_mismatch, find_stage_repair, requested_stage

COSTS = (
    "Combined with 12K GPU hours for the context length extension and 3K GPU hours for "
    "post-training, Orion costs only 90K GPU hours for its full training."
)
# A breakdown, 75K + 12K + 3K = 90K, with a rate that must not pass for a stage figure.
BREAKDOWN = (
    "During pre-training, each trillion tokens requires only 5K GPU hours, i.e., 3 days on our cluster. "
    "Consequently, our pre-training stage is completed in less than two months and costs 75K GPU hours. "
    + COSTS
)
PRE_TRAINING = "How many GPU hours did Orion's pre-training take?"


def verify(question, answer, rows, cite=(0,)):
    """Run the real verifier on (text, location) passages; cite the passages at these positions."""
    tasks = [{"question": question, "paper": "Orion.pdf"}]
    evidence = [
        {"id": f"S{number}", "title": "Orion.pdf", "content": text, "location": location, "task_ids": [0]}
        for number, (text, location) in enumerate(rows, 1)
    ]
    detail = DetailAnswer(availability="available", answer=answer, source_ids=[f"S{i + 1}" for i in cite])
    checked = grounding.verify_answer_citations(DraftAnswer(answers=[detail]), evidence, tasks)
    return grounding.format_cited_answer(checked, evidence, tasks), checked


@pytest.mark.parametrize(
    "question,stage",
    [
        ("What batch size did Orion use for pre-training?", "pre-training"),
        ("How many tokens was Orion pretrained on?", "pre-training"),
        ("How many tokens was Orion pre-trained on?", "pre-training"),
        ("What learning rate did Orion use when fine-tuning?", "fine-tuning"),
        ("How many epochs of SFT did Orion run?", "fine-tuning"),
        ("How many GPU hours did Orion's post-training take?", "post-training"),
        ("How many GPU hours did Orion's context extension take?", "context extension"),
        ("How many GPU hours did Orion's context length extension take?", "context extension"),
    ],
)
def test_question_names_one_stage(question, stage):
    assert requested_stage(question) == stage


@pytest.mark.parametrize(
    "question",
    [
        "How many parameters does the pre-trained Orion model have?",
        "What is the context window of the fine-tuned model?",
        "How many GPU hours did Orion take?",
        "What batch size did Orion use for pre-training and for fine-tuning?",
    ],
)
def test_question_without_exactly_one_stage_is_not_checked(question):
    """A describing adjective, no stage, or two stages leave nothing to bind a number to."""
    assert requested_stage(question) is None


def test_whole_run_total_cannot_answer_a_stage_that_has_its_own_value():
    """The headline total sits in the same sentence as the stage's own figure."""
    reason = find_stage_mismatch("context extension", {"90"}, [(COSTS, "")], [(COSTS, "")])
    assert "whole training" in reason and "12" in reason


def test_stage_value_in_its_own_clause_is_accepted():
    assert find_stage_mismatch("context extension", {"12"}, [(COSTS, "")], [(COSTS, "")]) == ""
    assert find_stage_mismatch("post-training", {"3"}, [(COSTS, "")], [(COSTS, "")]) == ""


def test_neighbouring_stage_value_in_the_same_sentence_is_rejected():
    """Each figure belongs to its own clause: 12K is context extension, not post-training."""
    reason = find_stage_mismatch("post-training", {"12"}, [(COSTS, "")], [(COSTS, "")])
    assert "different stage" in reason


def test_total_announced_in_its_own_clause_is_still_a_total():
    """'In total,' ends in a comma, so the marker is in the clause before the number."""
    text = "Context extension took 12K GPU hours. In total, Orion costs 90K GPU hours."
    assert "whole training" in find_stage_mismatch("context extension", {"90"}, [(text, "")], [(text, "")])


def test_total_is_accepted_when_the_paper_has_no_separate_stage_value():
    """A paper with one stage calls its only value 'total'; that is the pre-training value."""
    text = "In total, Orion was trained on 1.4T tokens."
    assert find_stage_mismatch("pre-training", {"1.4"}, [(text, "")], [(text, "")]) == ""


def test_value_from_another_stage_clause_is_rejected():
    text = "For fine-tuning, we use a batch size of 32."
    assert "different stage" in find_stage_mismatch("pre-training", {"32"}, [(text, "")], [(text, "")])
    assert find_stage_mismatch("fine-tuning", {"32"}, [(text, "")], [(text, "")]) == ""


def test_post_training_includes_fine_tuning():
    text = "Supervised fine-tuning ran for 3 epochs."
    assert find_stage_mismatch("post-training", {"3"}, [(text, "")], [(text, "")]) == ""


def test_one_place_that_supports_the_stage_is_enough():
    passages = [("Pre-training used a batch size of 256.", ""), ("Fine-tuning used a batch size of 256.", "")]
    assert find_stage_mismatch("pre-training", {"256"}, passages, passages) == ""


def test_place_that_names_no_stage_is_not_a_mismatch():
    """Nothing ties the number to another stage, so the check stays silent."""
    text = "The batch size is 256."
    assert find_stage_mismatch("pre-training", {"256"}, [(text, "")], [(text, "")]) == ""


def test_fragment_without_a_stage_does_not_rescue_a_number_bound_elsewhere():
    """A split-off bullet list says nothing about stages; another passage says it is fine-tuning."""
    fragment = "Batch size: 16, 32. Learning rate: 5e-5."
    bound = "For fine-tuning we use a batch size of 32."
    reason = find_stage_mismatch("pre-training", {"32"}, [(fragment, ""), (bound, "")], [(bound, "")])
    assert "different stage" in reason


def test_one_unexplained_number_accepts_the_answer():
    """Every number must be contradicted: the 512 below is stated without any stage."""
    bound = "For fine-tuning we use a batch size of 32."
    unexplained = "The sequence length is 512."
    cited = [(bound, ""), (unexplained, "")]
    assert find_stage_mismatch("pre-training", {"32", "512"}, cited, cited) == ""
    assert "different stage" in find_stage_mismatch("pre-training", {"32"}, cited, cited)


def test_stage_named_in_a_neighbouring_sentence_does_not_scope_a_figure():
    """A stage word in the sentence before is often only an adjective, so nothing is carried over."""
    text = "Orion starts from the pre-trained weights. It reduces the trainable parameters by 10000 times."
    assert find_stage_mismatch("fine-tuning", {"10000"}, [(text, "")], [(text, "")]) == ""
    text = "For pre-training we use the following values. The batch size is 32."
    assert find_stage_mismatch("fine-tuning", {"32"}, [(text, "")], [(text, "")]) == ""


def test_adjective_use_of_a_stage_word_names_no_stage():
    """'The pre-trained weights' describes the model; it does not make the figure a pre-training value."""
    text = "Orion freezes the pre-trained weights and trains 10000 times fewer parameters."
    assert find_stage_mismatch("fine-tuning", {"10000"}, [(text, "")], [(text, "")]) == ""
    text = "Orion is pre-trained on 10000 documents."
    assert "different stage" in find_stage_mismatch("fine-tuning", {"10000"}, [(text, "")], [(text, "")])


def test_comparison_baseline_in_the_question_names_no_stage():
    """'Compared with X fine-tuned with Adam' describes the baseline, not the stage asked about."""
    question = "Compared with GPT-3 fine-tuned with Adam, how much does Orion reduce the parameters?"
    assert requested_stage(question) is None


def test_digits_inside_a_name_are_not_values():
    """The 2 of 'L2' is a name; counting it would let any '2 heads' elsewhere accept the answer."""
    rows = [("For pre-training we use L2 weight decay of 0.01.", ""), ("The model has 2 heads.", "")]
    result, checked = verify("What weight decay did Orion use during fine-tuning?", "L2 weight decay of 0.01", rows, cite=(0, 1))
    assert result["status"] == "abstained"
    assert "different stage" in checked["rejections"][0]["reason"]


def test_scientific_notation_is_one_value():
    """1e-4 must not be read as the unrelated digits 1 and 4."""
    rows = [("For fine-tuning the learning rate is 1e-4.", ""), ("Table 1 has 4 rows.", "")]
    result, _ = verify("What learning rate did Orion use for pre-training?", "1e-4", rows, cite=(0, 1))
    assert result["status"] == "abstained"


def test_stage_verb_in_a_later_clause_scopes_the_whole_sentence():
    """'We use a batch size of 32 and fine-tune for 3 epochs' names one stage for both figures."""
    text = "We use a batch size of 32 and fine-tune for 3 epochs over the data."
    assert "different stage" in find_stage_mismatch("pre-training", {"32"}, [(text, "")], [(text, "")])
    assert find_stage_mismatch("fine-tuning", {"32"}, [(text, "")], [(text, "")]) == ""


def test_scope_clause_binds_the_figures_of_a_sentence_that_names_two_stages():
    """'For fine-tuning,' scopes the batch size; the pre-training figure keeps its own stage."""
    text = "For fine-tuning, we use a batch size of 32, whereas pre-training used 256."
    assert "different stage" in find_stage_mismatch("pre-training", {"32"}, [(text, "")], [(text, "")])
    assert find_stage_mismatch("pre-training", {"256"}, [(text, "")], [(text, "")]) == ""


def test_a_value_in_another_unit_does_not_compete_with_the_total():
    """Tokens in context extension say nothing about the GPU hours of the whole run."""
    text = "Orion costs 90K GPU hours for its full training. Context extension used 1.2T tokens."
    assert find_stage_mismatch("context extension", {"90"}, [(text, "")], [(text, "")]) == ""


def test_stage_named_only_to_compare_does_not_set_the_scope():
    """'The same as in pre-training' compares; the figure belongs to fine-tuning."""
    text = "Fine-tuning is the same as in pre-training, except the batch size is 32."
    assert find_stage_mismatch("pre-training", {"32"}, [(text, "")], [(text, "")]) != ""


def test_table_row_naming_every_stage_binds_no_number():
    row = "Cost Pre-Training Context Extension Post-Training Total 80K 12K 3K 95K"
    assert find_stage_mismatch("context extension", {"3"}, [(row, "")], [(row, "")]) == ""


def test_fine_tuning_value_cannot_answer_a_pre_training_request():
    rows = [
        ("We use a batch size of 32.", "4 Training > 4.2 Fine-tuning"),
        ("We train with a batch size of 256.", "4 Training > 4.1 Pre-training"),
    ]
    result, checked = verify("What batch size did Orion use for pre-training?", "32", rows)
    assert result["status"] == "abstained"
    assert "different stage" in checked["rejections"][0]["reason"]


def test_section_heading_binds_a_sentence_that_never_names_the_stage():
    question = "What batch size did Orion use for pre-training?"
    text = "We train with a batch size of 256 sequences for 1000 steps."
    assert verify(question, "256", [(text, "A.2 Pre-training Procedure")])[0]["status"] == "answered"
    result, checked = verify(question, "256", [(text, "A.3 Fine-tuning Procedure")])
    assert result["status"] == "abstained"
    assert "different stage" in checked["rejections"][0]["reason"]


def test_headline_total_is_rejected_through_the_verifier():
    question = "How many GPU hours did Orion's context extension take?"
    result, checked = verify(question, "Orion costs 90K GPU hours.", [(COSTS, "1 Introduction")])
    assert result["status"] == "abstained"
    assert "whole training" in checked["rejections"][0]["reason"]
    assert verify(question, "12K GPU hours.", [(COSTS, "1 Introduction")])[0]["status"] == "answered"


def test_passage_without_a_location_does_not_crash_the_check():
    """Study passages and older indexes may carry no section heading, or a null one."""
    tasks = [{"question": "What batch size did Orion use for pre-training?", "paper": "Orion.pdf"}]
    for evidence_row in ({"content": "We train with a batch size of 256."}, {"content": "We train with a batch size of 256.", "location": None}):
        evidence = [{"id": "S1", "title": "Orion.pdf", "task_ids": [0], **evidence_row}]
        detail = DetailAnswer(availability="available", answer="256", source_ids=["S1"])
        checked = grounding.verify_answer_citations(DraftAnswer(answers=[detail]), evidence, tasks)
        assert grounding.format_cited_answer(checked, evidence, tasks)["status"] == "answered"


def test_answer_without_a_number_or_a_question_without_a_stage_is_not_checked():
    text = "For fine-tuning, we use a batch size of 32."
    assert verify("Which optimizer did Orion use for pre-training?", "Adam", [("Adam is used for fine-tuning.", "")])[0]["status"] == "answered"
    assert verify("What batch size did Orion use?", "32", [(text, "")])[0]["status"] == "answered"


def test_stage_named_only_to_place_in_time_does_not_set_the_scope():
    """The 0.1M below belongs to the stages after pre-training, not to pre-training itself."""
    text = "The stages after pre-training require only 0.1M GPU hours."
    assert find_stage_mismatch("pre-training", {"0.1"}, [(text, "")], [(text, "")]) == ""
    assert find_stage_mismatch("post-training", {"0.1"}, [(text, "")], [(text, "")]) == ""


def test_entire_training_dataset_is_not_the_whole_run():
    """LLaMA: 'our entire training dataset contains 1.4T tokens' is the pre-training data, not a total."""
    text = (
        "Overall, our entire training dataset contains roughly 1.4T tokens. "
        "The pre-training runs on 1T tokens have the same sampling proportion."
    )
    assert find_stage_mismatch("pre-training", {"1.4"}, [(text, "")], [(text, "")]) == ""
    run = text.replace("entire training dataset", "entire training")
    assert "whole training" in find_stage_mismatch("pre-training", {"1.4"}, [(run, "")], [(run, "")])
    result, _ = verify("How many tokens was Orion pre-trained on?", "1.4T tokens", [(text, "2.1 Pre-training Data")])
    assert result["status"] == "answered"


def repair(question, number, text=BREAKDOWN, extra=()):
    """The repair for an answer that gave `number` from these passages, with extra passages available."""
    rows = [(text, "")]
    return find_stage_repair(question, {number}, rows, rows + list(extra))


def test_whole_run_total_is_replaced_by_the_clause_that_names_the_stage():
    assert repair("How many GPU hours did Orion's context extension take?", "90") == (
        0, "12K GPU hours for the context length extension"
    )
    assert repair("How many GPU hours did Orion's post-training take?", "90") == (0, "3K GPU hours for post-training")


def test_stage_named_elsewhere_in_the_sentence_quotes_the_whole_sentence():
    assert repair(PRE_TRAINING, "90") == (
        0, "Consequently, our pre-training stage is completed in less than two months and costs 75K GPU hours."
    )


def test_repair_needs_the_unit_the_question_names():
    """GPU hours say nothing about days, or about 'compute' in general."""
    assert repair("How many days did Orion's context extension take?", "90") is None
    assert repair("How much compute did Orion's context extension take?", "90") is None
    assert repair("How many GPU-hours did Orion's context extension take?", "90") is not None


def test_a_rate_is_not_a_stage_figure():
    """5K GPU hours per trillion tokens would compete with the 75K of the pre-training stage."""
    assert repair(PRE_TRAINING, "90") is not None
    rate_only = (
        "During pre-training, each trillion tokens requires only 5K GPU hours. "
        "Orion costs 90K GPU hours for its full training."
    )
    assert repair(PRE_TRAINING, "90", rate_only) is None


def test_the_same_figure_written_two_ways_is_one_value():
    again = "At a cost of only 0.075M GPU hours, we complete the pre-training of Orion."
    assert repair(PRE_TRAINING, "90", extra=[(again, "")]) is not None


def test_two_different_figures_for_the_stage_are_not_guessed_between():
    other = "A second pre-training run cost 80K GPU hours."
    assert repair(PRE_TRAINING, "90", extra=[(other, "")]) is None


def test_a_breakdown_that_does_not_add_up_is_not_trusted():
    """One stage figure beside a total that is something else, as for a paper with model sizes."""
    single = "Pre-training used 1T tokens. Overall, Orion was trained on 1.4T tokens in total."
    assert repair("How many tokens was Orion pre-trained on?", "1.4", single) is None
    assert repair(PRE_TRAINING, "90", BREAKDOWN.replace("75K", "60K")) is None


def test_stages_that_contain_each_other_are_not_added_together():
    """Fine-tuning is part of post-training: 75 + 12 + 3 + 2 = 92 would count it twice."""
    text = (
        "Pre-training cost 75K GPU hours. Context extension cost 12K GPU hours. "
        "Post-training cost 3K GPU hours, of which fine-tuning took 2K GPU hours. "
        "Orion costs 92K GPU hours for its full training."
    )
    assert repair(PRE_TRAINING, "92", text) is None
    parts = (
        "Pre-training cost 75K GPU hours. Post-training cost 3K GPU hours. "
        "Fine-tuning cost 2K GPU hours. Orion costs 80K GPU hours for its full training."
    )
    assert repair("How many GPU hours did Orion's post-training take?", "80", parts) is None


def test_only_a_plain_whole_run_total_is_repaired():
    """12K is context extension's own figure: giving it for post-training is a different-stage answer."""
    assert repair("How many GPU hours did Orion's post-training take?", "12") is None
    assert find_stage_repair(PRE_TRAINING, {"90", "999"}, [(BREAKDOWN, "")], [(BREAKDOWN, "")]) is None
    assert find_stage_repair("How many GPU hours did Orion take?", {"90"}, [(BREAKDOWN, "")], [(BREAKDOWN, "")]) is None
    # A number that the cited text states without calling it the whole run is not a total.
    assert find_stage_repair(PRE_TRAINING, {"90"}, [("Orion used 90K GPU hours.", "")], [(BREAKDOWN, "")]) is None


def test_a_total_that_a_cited_passage_also_gives_for_a_stage_is_not_repaired():
    stage = "Pre-training alone took 90K GPU hours."
    rows = [(BREAKDOWN, ""), (stage, "")]
    assert find_stage_repair("How many GPU hours did Orion's context extension take?", {"90"}, rows, rows) is None


def test_only_figures_the_text_binds_to_one_stage_compete():
    """A heading-bound sentence or a clause naming two stages says nothing certain about one stage."""
    heading_only = ("The later stages require only 0.1M GPU hours.", "4. Pre-Training")
    assert repair(PRE_TRAINING, "90", extra=[heading_only]) is not None
    two_stages = ("Pre-training plus context extension cost 87K GPU hours.", "")
    assert repair("How many GPU hours did Orion's context extension take?", "90", extra=[two_stages]) is not None


def test_breakdown_may_differ_from_a_rounded_total():
    assert repair(PRE_TRAINING, "90.5", BREAKDOWN.replace("90K", "90.5K")) is not None
    assert repair(PRE_TRAINING, "92", BREAKDOWN.replace("90K", "92K")) is None


def test_a_hardware_name_is_not_part_of_the_unit():
    assert repair(PRE_TRAINING, "90", BREAKDOWN.replace("90K GPU hours", "90K H800 GPU hours")) is not None


def test_a_long_sentence_is_not_quoted():
    padded = "Consequently, our pre-training stage is completed " + "in a very long time " * 20 + "and costs 75K GPU hours."
    assert repair(PRE_TRAINING, "90", padded + " " + COSTS) is None


def test_whole_run_answer_is_repaired_through_the_verifier():
    question = "How many GPU hours did Orion's context extension take?"
    result, checked = verify(question, "Orion costs 90K GPU hours.", [(BREAKDOWN, "1 Introduction")])
    assert result["status"] == "answered"
    assert result["answer"] == "12K GPU hours for the context length extension [S1]"
    assert checked["rejections"] == [] and checked["repairs"][0]["detail_index"] == 0
    assert "whole training" in checked["repairs"][0]["reason"]


def test_a_repair_that_fails_a_check_leaves_the_rejection(monkeypatch):
    """The quoted sentence is verified like a draft: a number the passage does not state sinks it."""
    monkeypatch.setattr(
        grounding, "repair_requested_stage", lambda *args: ("S1", "13K GPU hours for the context length extension")
    )
    question = "How many GPU hours did Orion's context extension take?"
    result, checked = verify(question, "Orion costs 90K GPU hours.", [(BREAKDOWN, "1 Introduction")])
    assert result["status"] == "abstained" and checked["repairs"] == []
    assert "whole training" in checked["rejections"][0]["reason"]


def test_repair_can_be_switched_off():
    question = "How many GPU hours did Orion's context extension take?"
    tasks = [{"question": question, "paper": "Orion.pdf"}]
    evidence = [{"id": "S1", "title": "Orion.pdf", "content": BREAKDOWN, "location": "", "task_ids": [0]}]
    draft = DraftAnswer(answers=[DetailAnswer(availability="available", answer="90K GPU hours.", source_ids=["S1"])])
    checked = grounding.verify_answer_citations(draft, evidence, tasks, repair=False)
    assert checked["accepted"] == [] and checked["repairs"] == []
    assert "whole training" in checked["rejections"][0]["reason"]
