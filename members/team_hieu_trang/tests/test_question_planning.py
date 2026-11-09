"""Test decomposition and paper-scope resolution for questions and follow-ups.

Controlled planner outputs check direct, multipart and comparison routes, including
ambiguous or missing paper references.
"""

import pytest
from answer_fixtures import cited_draft

from app import grounding, services
from app import question_planning as planning
from app.question_planning import QuestionPlan, SearchTask


def test_empty_catalog_skips_planner_model(monkeypatch):
    monkeypatch.setattr(
        planning,
        "request_structured_response",
        lambda *args: pytest.fail("Unexpected model"),
    )
    result = planning.plan_question_tasks("What is attention?", [], [])
    assert len(result.tasks) == 1


@pytest.mark.parametrize("question", [
    "What is LoRA?",
    "Which optimizer does LoRA use?",
    "How many parameters does LoRA train?",
])
def test_simple_definition_searches_known_paper_without_model(monkeypatch, question):
    monkeypatch.setattr(planning, "request_structured_response", lambda *args: pytest.fail("Unexpected model"))
    papers = [{"title": "16_LoRA.pdf", "aliases": ["LoRA"]}]
    plan = planning.plan_question_tasks(question, [], papers)
    assert [(task.question, task.paper, task.scope) for task in plan.tasks] == [
        (question, "16_LoRA.pdf", "paper")
    ]


@pytest.mark.parametrize("question,history", [
    ("What is LoRA and its limitations?", []),
    ("How does LoRA compare with BERT?", []),
    ("What are LoRA's strengths, weaknesses?", []),
    ("Which language models does LoRA evaluate?", []),
    ("What is LoRA?", [{"role": "user", "content": "Explain the earlier paper."}]),
])
def test_complex_or_followup_questions_still_use_planner(monkeypatch, question, history):
    calls = []
    def plan_with_model(*args):
        calls.append(True)
        return QuestionPlan(tasks=[SearchTask(question=question, paper="16_LoRA.pdf", scope="paper")])
    monkeypatch.setattr(planning, "request_structured_response", plan_with_model)
    planning.plan_question_tasks(question, history, [{"title": "16_LoRA.pdf", "aliases": ["LoRA"]}])
    assert calls


def test_explicit_independent_parts_keep_the_original_requests(monkeypatch):
    """Separate original clauses about one paper without inventing new tasks."""
    monkeypatch.setattr(planning, "request_structured_response", lambda *args: pytest.fail("No model rewrite needed"))
    papers = [{"title": "01_ORION_Research.pdf"}]
    plan = planning.plan_question_tasks("What are the Orion assumptions, and how does Orion train?", [], papers)
    assert len(plan.tasks) == 2
    assert plan.tasks[0].question.startswith("What are the Orion assumptions?")
    assert plan.tasks[1].question.startswith("how does Orion train?")
    assert all(task.paper == papers[0]["title"] for task in plan.tasks)


def no_model(*args):
    pytest.fail("Explicit clauses need no planner call")


def test_pronoun_clause_about_the_one_named_paper_gets_its_own_task(monkeypatch):
    """'how many tokens was it trained on' is a second request about the same paper."""
    monkeypatch.setattr(planning, "request_structured_response", no_model)
    papers = [{"title": "01_Orion.pdf", "aliases": ["Orion"]}]
    plan = planning.plan_question_tasks("How many parameters does Orion have and how many tokens was it trained on?", [], papers)
    assert [task.question for task in plan.tasks] == ["How many parameters does Orion have?", "how many tokens was it trained on?"]
    assert all(task.paper == "01_Orion.pdf" for task in plan.tasks)


def test_clauses_about_two_papers_search_each_paper(monkeypatch):
    """Each clause names its own paper, so each paper is the subject of one task."""
    monkeypatch.setattr(planning, "request_structured_response", no_model)
    papers = [{"title": "01_Orion.pdf", "aliases": ["Orion"]}, {"title": "02_Sirius.pdf", "aliases": ["Sirius"]}]
    plan = planning.plan_question_tasks("How many experts does Orion have, and how many tokens was Sirius trained on?", [], papers)
    assert [task.paper for task in plan.tasks] == ["01_Orion.pdf", "02_Sirius.pdf"]


def test_clause_about_an_unindexed_model_does_not_borrow_the_named_paper(monkeypatch):
    """'how many parameters does GPT-4 have' must not search Orion's paper."""
    monkeypatch.setattr(planning, "request_structured_response", no_model)
    papers = [{"title": "01_Orion.pdf", "aliases": ["Orion"]}]
    plan = planning.plan_question_tasks("How many tokens was Orion trained on, and how many parameters does GPT-4 have?", [], papers)
    assert plan.tasks[0].paper == "01_Orion.pdf"
    assert plan.tasks[1].paper is None and plan.tasks[1].scope == "unresolved"


def test_context_phrase_or_passing_mention_is_not_a_separate_clause():
    """'Compared with Sirius, how much ...' has one question; a pronoun with two papers stays whole."""
    assert planning.split_explicit_question_clauses("Compared with Sirius 7B, how much does Orion reduce memory?") == []
    assert planning.split_explicit_question_clauses("How large is Orion, and how does it compare with Sirius?") == []


def test_ambiguous_clause_stays_together_for_context():
    """Do not split a clause that depends on a missing subject or a pronoun."""
    assert planning.split_explicit_question_clauses("What is Orion, and how does it train?") == []
    assert planning.split_explicit_question_clauses("How many experts are there, and how many are used?") == []


TWO_PAPERS = [{"title": "01_Orion.pdf", "aliases": ["Orion"]}, {"title": "02_Sirius.pdf", "aliases": ["Sirius"]}]


def test_possessive_about_the_clause_own_paper_splits_per_clause(monkeypatch):
    """'how many GPUs does Sirius use for its fine-tuning' is only about Sirius."""
    monkeypatch.setattr(planning, "request_structured_response", no_model)
    plan = planning.plan_question_tasks(
        "Which optimizer does Orion use, and how many GPUs does Sirius use for its fine-tuning?", [], TWO_PAPERS
    )
    assert [(task.question, task.paper) for task in plan.tasks] == [
        ("Which optimizer does Orion use?", "01_Orion.pdf"),
        ("how many GPUs does Sirius use for its fine-tuning?", "02_Sirius.pdf"),
    ]


@pytest.mark.parametrize(
    "question",
    [
        # 'it' is the optimizer from the first clause.
        "Which optimizer does Orion use, and what learning rate does Sirius use with it?",
        # A comparison needs both papers.
        "How large is Orion, and how does Sirius compare with its size?",
        # The possessive comes before the clause's own paper, so it may mean Orion.
        "Which dataset does Orion use, and how large is its test split in Sirius?",
        # 'how many are used' leaves out what is counted.
        "How many experts does Orion have, and how many are used by Sirius?",
    ],
)
def test_clause_that_depends_on_another_clause_is_left_to_the_planner(question):
    """These clauses cannot be searched on their own, so no explicit tasks are made."""
    requested = planning.find_paper_references(question, TWO_PAPERS)
    assert planning.plan_explicit_clauses(question, requested, TWO_PAPERS) == []


def test_repeated_preposition_asks_about_each_paper(monkeypatch):
    """'in the Orion paper and in the Sirius paper' gets one focused task per paper."""
    monkeypatch.setattr(planning, "request_structured_response", lambda *args: QuestionPlan())
    plan = planning.plan_question_tasks(
        "Which activation function is used in the Orion paper and in the Sirius paper?", [], TWO_PAPERS
    )
    assert [task.paper for task in plan.tasks] == ["01_Orion.pdf", "02_Sirius.pdf"]
    assert "Sirius" not in plan.tasks[0].question and "Orion" not in plan.tasks[1].question


def test_task_word_key_keeps_different_requested_details():
    """Duplicate normalization preserves question words, values and negation."""
    assert planning.task_word_key("What does Orion train on?") == planning.task_word_key("What Orion does train on?")
    assert planning.task_word_key("What does Orion train on?") != planning.task_word_key("How does Orion train?")
    assert planning.task_word_key("What is Orion 7B?") != planning.task_word_key("What is Orion 70B?")
    assert planning.task_word_key("Does Orion beat Sirius?") != planning.task_word_key("Does Sirius beat Orion?")
    assert planning.task_word_key("What did Orion use?") != planning.task_word_key("What does Orion use?")


def test_two_named_subjects_get_separate_focused_tasks(monkeypatch):
    """A question about each of two subjects should not let one answer cover both."""
    papers = [{"title": "01_ORION.pdf"}, {"title": "02_SIRIUS.pdf"}]
    monkeypatch.setattr(
        planning,
        "request_structured_response",
        lambda schema, *args: schema(tasks=[{"question": "Which methods do ORION and SIRIUS use?", "paper": None, "scope": "broad"}]),
    )
    plan = planning.plan_question_tasks("Which methods do ORION and SIRIUS use?", [], papers)
    assert [task.paper for task in plan.tasks] == ["01_ORION.pdf", "02_SIRIUS.pdf"]
    assert "SIRIUS" not in plan.tasks[0].question
    assert "ORION" not in plan.tasks[1].question
    assert "and use" not in plan.tasks[0].question


def test_unknown_comparison_preserves_resolvable_part(monkeypatch):
    """An unknown paper cannot stop a separately identified paper's task."""
    monkeypatch.setattr(
        planning,
        "request_structured_response",
        lambda *args: QuestionPlan(
            route="comparison",
            tasks=[
                SearchTask(question="Explain Unknown", scope="unresolved"),
                SearchTask(
                    question="Explain Orion", paper="02_Orion.pdf", scope="paper"
                ),
            ],
        ),
    )
    result = planning.plan_question_tasks(
        "Compare Unknown and Orion", [], [{"title": "02_Orion.pdf"}]
    )
    assert not result.clarification
    assert result.tasks[0].scope == "unresolved"
    assert result.tasks[1].paper == "02_Orion.pdf"


def test_clarification_skips_search(monkeypatch):
    monkeypatch.setattr(
        services,
        "plan_question_tasks",
        lambda *args: QuestionPlan(clarification="Which paper?"),
    )
    monkeypatch.setattr(
        services,
        "retrieve_relevant_passages",
        lambda *args, **kwargs: pytest.fail("Must skip search"),
    )
    result = services.answer_research_question(
        "Compare these papers", require_review=True
    )
    assert result["status"] == "clarification"
    assert not result["context"]


def test_comparison_scopes_tasks_and_marks_missing_coverage(monkeypatch):
    monkeypatch.setattr(
        services,
        "plan_question_tasks",
        lambda *args: QuestionPlan(
            route="comparison",
            tasks=[
                SearchTask(question="Method A?", paper="A"),
                SearchTask(question="Method B?", paper="B"),
            ],
        ),
    )
    calls = []

    def search(query, paper=None, diagnostics=None):
        calls.append((query, paper))
        diagnostics.update(
            candidate_sources=[paper], reranked_sources=[paper], candidate_count=1
        )
        return [{"id": paper, "title": paper, "content": "This uses 8 GPUs."}]

    monkeypatch.setattr(services, "retrieve_relevant_passages", search)
    monkeypatch.setattr(
        grounding,
        "generate_from_evidence",
        lambda *args: cited_draft("Uses 8 GPUs. [S1]"),
    )
    result = services.answer_research_question("Compare A and B")
    assert calls == [("Method A?", "A"), ("Method B?", "B")]
    assert result["retrieval_trace"]["candidate_sources"] == ["A", "B"]
    assert result["status"] == "partial"
    assert result["missing_tasks"] == ["Method B?"]
    assert result["context"][0]["id"] == result["sources"][0]["id"]


def test_shared_budget_and_provenance(monkeypatch):

    def search(query, **kwargs):
        return [{"id": "shared", "content": "x" * 2000}] + [
            {"id": f"{query}{n}", "content": "x" * 2400} for n in range(5)
        ]

    monkeypatch.setattr(services, "retrieve_relevant_passages", search)
    tasks = [{"question": q, "paper": None} for q in "abc"]
    result = services.retrieve_task_evidence({"tasks": tasks, "retrieval_trace": {}})
    assert (
        sum(len(row["content"]) for row in result["evidence"])
        <= services.CONTEXT_CHARACTERS
    )
    assert result["evidence"][0]["task_ids"] == [0, 1, 2]


def test_empty_task_cannot_borrow_another_tasks_evidence(monkeypatch):
    """Matching paper names alone do not mean a passage was selected for a task."""
    def search(query, **kwargs):
        return [{"id": "hit", "title": "A.pdf", "content": "Some evidence."}] if query == "First" else []

    monkeypatch.setattr(services, "retrieve_relevant_passages", search)
    tasks = [
        {"question": "First", "paper": "A.pdf", "scope": "paper"},
        {"question": "Second", "paper": "A.pdf", "scope": "paper"},
        {"question": "Other", "paper": "B.pdf", "scope": "paper"},
    ]
    result = services.retrieve_task_evidence({"tasks": tasks, "retrieval_trace": {}})
    assert result["evidence"][0]["task_ids"] == [0]


def test_same_paper_search_keeps_original_question_as_context(monkeypatch):
    """Short tasks retain the original subject without changing verification tasks."""
    queries = []
    monkeypatch.setattr(services, "retrieve_relevant_passages", lambda query, **kwargs: queries.append(query) or [])
    tasks = [
        {"question": "How many experts?", "paper": "A.pdf", "scope": "paper"},
        {"question": "How many are used?", "paper": "A.pdf", "scope": "paper"},
    ]
    result = services.retrieve_task_evidence({"message": "How many experts are used per token?", "tasks": tasks, "retrieval_trace": {}})
    assert all("How many experts are used per token?" in query for query in queries)
    assert result["tasks"][1]["question"] == "How many are used?"


def test_empty_task_keeps_its_best_passages_when_a_sibling_found_evidence(monkeypatch):
    """A sub-question worded unlike its paper is searched again, keeping two passages."""
    calls = []

    def search(query, paper=None, diagnostics=None, keep_best=0):
        calls.append((paper, keep_best))
        if paper == "B.pdf" and not keep_best:
            return []
        return [{"id": f"{paper}-{keep_best}", "title": paper, "content": "Evidence."}]

    monkeypatch.setattr(services, "retrieve_relevant_passages", search)
    tasks = [
        {"question": "GPUs for A?", "paper": "A.pdf", "scope": "paper"},
        {"question": "GPUs for B?", "paper": "B.pdf", "scope": "paper"},
    ]
    result = services.retrieve_task_evidence({"tasks": tasks, "retrieval_trace": {}})
    assert calls == [("A.pdf", 0), ("B.pdf", 0), ("B.pdf", 2)]
    assert [row["chunk_id"] for row in result["evidence"]] == ["A.pdf-0", "B.pdf-2"]
    assert result["retrieval_trace"]["tasks"][1]["trace"]["kept_best_below_cutoff"]


def test_question_whose_tasks_all_find_nothing_gets_no_fallback(monkeypatch):
    """With no evidence for any task, the question still abstains without weak passages."""
    calls = []
    monkeypatch.setattr(
        services,
        "retrieve_relevant_passages",
        lambda query, paper=None, diagnostics=None, keep_best=0: calls.append(keep_best) or [],
    )
    tasks = [{"question": q, "paper": f"{q}.pdf", "scope": "paper"} for q in "AB"]
    result = services.retrieve_task_evidence({"tasks": tasks, "retrieval_trace": {}})
    assert calls == [0, 0]
    assert result["evidence"] == []


def test_shared_budget_keeps_a_strong_keyword_candidate_for_each_task(monkeypatch):
    """MiniLM's first hit and an accepted keyword hit can both reach the model."""

    def search(query, **kwargs):
        return [
            {"id": f"{query}-{i}", "content": "x" * 2200, "bm25_score": score}
            for i, score in enumerate([1, 2, 3, 9])
        ]

    monkeypatch.setattr(services, "retrieve_relevant_passages", search)
    tasks = [{"question": q, "paper": None} for q in "abc"]
    result = services.retrieve_task_evidence({"tasks": tasks, "retrieval_trace": {}})
    selected = {row["chunk_id"] for row in result["evidence"]}
    assert selected == {f"{q}-{i}" for q in "abc" for i in (0, 3)}


def test_source_labels_group_each_tasks_evidence(monkeypatch):
    """Present contiguous source labels per task after fair context selection."""
    monkeypatch.setattr(
        services,
        "retrieve_relevant_passages",
        lambda query, **kwargs: [
            {"id": f"{query}-{i}", "content": "Relevant text"} for i in (1, 2)
        ],
    )
    tasks = [{"question": q, "paper": None} for q in "ab"]
    result = services.retrieve_task_evidence({"tasks": tasks, "retrieval_trace": {}})
    assert [(row["id"], row["chunk_id"]) for row in result["evidence"]] == [
        ("S1", "a-1"),
        ("S2", "a-2"),
        ("S3", "b-1"),
        ("S4", "b-2"),
    ]


def test_multipart_plan_keeps_subquestions(monkeypatch):
    """The planner receives user context and catalog only, and preserves distinct tasks."""
    captured = []

    def structured(schema, instruction, payload):
        captured.append(payload)
        return QuestionPlan(
            route="multipart",
            tasks=[
                SearchTask(question="What is the method?"),
                SearchTask(question="What are its limitations?"),
            ],
        )

    monkeypatch.setattr(planning, "request_structured_response", structured)
    result = planning.plan_question_tasks(
        "Explain the method and discuss its limitations", [], [{"title": "A"}]
    )
    assert len(result.tasks) == 2
    assert set(captured[0]) == {
        "question",
        "verified_references",
        "recent_conversation",
        "indexed_papers",
    }


def test_focused_tasks_search_each_subject_in_its_own_paper(monkeypatch):
    """A shared measurement for two subjects becomes two scoped retrieval calls."""
    papers = [
        {"title": "01_Orion.pdf", "aliases": ["Orion"]},
        {"title": "02_Sirius.pdf", "aliases": ["Sirius"]},
    ]
    monkeypatch.setattr(
        planning, "request_structured_response", lambda *args: QuestionPlan()
    )
    plan = planning.plan_question_tasks(
        "Orion 20B and Sirius 40B were pre-trained. How many tokens did each use?",
        [],
        papers,
    )
    calls = []
    monkeypatch.setattr(
        services,
        "retrieve_relevant_passages",
        lambda query, paper, **kwargs: calls.append((query, paper)) or [],
    )
    services.retrieve_task_evidence(
        {"tasks": [task.model_dump() for task in plan.tasks], "retrieval_trace": {}}
    )
    assert [paper for _, paper in calls] == [paper["title"] for paper in papers]
    assert "Orion 20B" in calls[0][0] and "Sirius" not in calls[0][0]
    assert "Sirius 40B" in calls[1][0] and "Orion" not in calls[1][0]
    assert all("tokens" in query and "pre-trained" in query for query, _ in calls)


def test_duplicate_tasks_do_not_repeat_search_or_remove_other_aspects(monkeypatch):
    """Remove repeated wording while preserving distinct requests in their order."""
    monkeypatch.setattr(
        planning,
        "request_structured_response",
        lambda schema, *args: schema(
            tasks=[
                {"question": q, "paper": "Orion.pdf", "scope": "paper"}
                for q in (
                    "Explain Orion's method.",
                    "  Explain   ORION's method.  ",
                    "List Orion's limitations.",
                )
            ]
        ),
    )
    plan = planning.plan_question_tasks(
        "Explain Orion's method and list its limitations.",
        [],
        [{"title": "Orion.pdf", "aliases": ["Orion"]}],
    )
    assert [task.question for task in plan.tasks] == [
        "Explain Orion's method.",
        "List Orion's limitations.",
    ]
    assert plan.route == "multipart"


def test_multipart_followup_keeps_each_aspect_and_user_history_scope(monkeypatch):
    """Follow-up subquestions remain standalone and use the previous user's paper."""
    questions = ["What are Orion's assumptions?", "What are Orion's limitations?"]
    monkeypatch.setattr(
        planning,
        "request_structured_response",
        lambda schema, *args: schema(
            tasks=[
                {"question": q, "paper": "Orion.pdf", "scope": "paper"}
                for q in questions
            ]
        ),
    )
    result = planning.plan_question_tasks(
        "What are its assumptions and limitations?",
        [{"role": "human", "content": "Explain Orion's training method."}],
        [{"title": "Orion.pdf", "aliases": ["Orion"]}],
    )
    assert [task.question for task in result.tasks] == questions
    assert all(task.paper == "Orion.pdf" for task in result.tasks)


def test_single_followup_preserves_original_request_and_resolved_subject(monkeypatch):
    """A narrowed rewrite cannot erase the user's original multipart follow-up."""
    monkeypatch.setattr(
        planning,
        "request_structured_response",
        lambda schema, *args: schema(
            tasks=[
                {
                    "question": "What are Orion's assumptions?",
                    "paper": "Orion.pdf",
                    "scope": "paper",
                }
            ]
        ),
    )
    result = planning.plan_question_tasks(
        "What are its assumptions and limitations?",
        [{"role": "human", "content": "Explain Orion's training method."}],
        [{"title": "Orion.pdf", "aliases": ["Orion"]}],
    )
    assert result.tasks[0].question == (
        "What are its assumptions and limitations?\nWhat are Orion's assumptions?"
    )
    assert result.tasks[0].paper == "Orion.pdf"


def test_requested_list_stays_one_search_task(monkeypatch):
    """Unknown list members are answer content, not extra tasks to guess up front."""
    question = "Which baselines does Orion evaluate?"
    monkeypatch.setattr(
        planning,
        "request_structured_response",
        lambda schema, *args: schema(
            tasks=[{"question": question, "paper": "Orion.pdf", "scope": "paper"}]
        ),
    )
    result = planning.plan_question_tasks(
        question, [], [{"title": "Orion.pdf", "aliases": ["Orion"]}]
    )
    assert len(result.tasks) == 1
    assert result.tasks[0].question == question


def test_each_paper_gets_focused_search_without_guessing_other_papers(monkeypatch):
    """Each named paper gets its own query with the common requested subject."""
    papers = [
        {"title": "01_Orion.pdf", "aliases": ["Orion"]},
        {"title": "02_Sirius.pdf", "aliases": ["Sirius"]},
        {"title": "03_Vega.pdf", "aliases": ["Vega"]},
    ]
    monkeypatch.setattr(
        planning, "request_structured_response", lambda *args: QuestionPlan()
    )
    plan = planning.plan_question_tasks(
        "Which survey data does each of Orion, Sirius and Vega use, and how?",
        [],
        papers,
    )
    assert [task.paper for task in plan.tasks] == [row["title"] for row in papers]
    for task, name in zip(plan.tasks, ["Orion", "Sirius", "Vega"]):
        assert "survey data" in task.question
        assert name in task.question
        assert not any(
            other in task.question for other in {"Orion", "Sirius", "Vega"} - {name}
        )


def test_each_request_keeps_an_unidentified_subject_from_the_model_plan(monkeypatch):
    """A missing paper remains a separate unresolved part beside known subjects."""
    monkeypatch.setattr(
        planning,
        "request_structured_response",
        lambda *args: QuestionPlan(
            tasks=[
                SearchTask(
                    question="How many samples did Nova use?", scope="unresolved"
                )
            ]
        ),
    )
    plan = planning.plan_question_tasks(
        "How many samples did each of Orion, Sirius and Nova use?",
        [],
        [{"title": "Orion.pdf"}, {"title": "Sirius.pdf"}],
    )
    assert [task.paper for task in plan.tasks] == ["Orion.pdf", "Sirius.pdf", None]
    assert plan.tasks[2].scope == "unresolved"
    assert "Nova" in plan.tasks[2].question


def test_comparison_focus_preserves_target_variant_and_removes_other_variant():
    """Versions stay with their own subject instead of polluting another search."""
    papers = [
        {"title": "01_Orion.pdf", "aliases": ["Orion"]},
        {"title": "02_Sirius.pdf", "aliases": ["Sirius"]},
    ]
    question = "Compare training time for Orion-12B versus Sirius-30B."
    first = planning.focus_question_on_paper(question, papers[0]["title"], papers)
    second = planning.focus_question_on_paper(question, papers[1]["title"], papers)
    assert "12B" in first and "30B" not in first
    assert "30B" in second and "12B" not in second


def test_focus_removes_a_longer_overlapping_name_before_a_shorter_one():
    """Removing 'Orion' first must not leave the '3 405B' of another paper's 'Orion 3'."""
    papers = [
        {"title": "01_Orion.pdf", "aliases": ["Orion"]},
        {"title": "02_Orion_3_Herd.pdf", "aliases": ["Orion 3"]},
        {"title": "03_Vega.pdf", "aliases": ["Vega"]},
    ]
    question = "Which GPUs trained Orion-65B, Orion 3 405B and Vega?"
    focused = planning.focus_question_on_paper(question, "03_Vega.pdf", papers)
    assert "Vega" in focused
    assert "Orion" not in focused and "405B" not in focused and "65B" not in focused


def test_focus_removes_another_paper_name_with_its_attached_suffix():
    """'Vega-R1-Zero' must not leave '-Zero' in Orion's search."""
    papers = [
        {"title": "01_Orion.pdf", "aliases": ["Orion"]},
        {"title": "02_Vega-R1.pdf", "aliases": ["Vega-R1"]},
    ]
    question = "Which base models do Orion and Vega-R1-Zero start from?"
    orion = planning.focus_question_on_paper(question, "01_Orion.pdf", papers)
    vega = planning.focus_question_on_paper(question, "02_Vega-R1.pdf", papers)
    assert "Zero" not in orion and "Vega" not in orion and "Orion" in orion
    assert "Vega-R1-Zero" in vega and "Orion" not in vega


def test_list_question_left_as_one_joint_task_is_split_per_paper(monkeypatch):
    """A planner task naming five papers becomes five focused searches."""
    names = ["Orion", "Sirius", "Vega", "Lyra", "Deneb"]
    papers = [{"title": f"0{i}_{name}.pdf", "aliases": [name]} for i, name in enumerate(names, 1)]
    # "was" (not do/did/are/were) leaves the split to the planner, as in a real list question.
    question = "Which GPU hardware was used to train Orion, Sirius, Vega, Lyra and Deneb?"
    monkeypatch.setattr(
        planning,
        "request_structured_response",
        lambda schema, *args: schema(tasks=[{"question": question, "paper": None, "scope": "paper"}]),
    )
    plan = planning.plan_question_tasks(question, [], papers)
    assert sorted(task.paper for task in plan.tasks) == sorted(row["title"] for row in papers)
    for task in plan.tasks:
        name = task.paper[3:-4]
        assert name in task.question and "GPU hardware" in task.question
        assert not any(other in task.question for other in set(names) - {name})


FOCUS_PAPERS = [
    {"title": "01_Orion.pdf", "aliases": ["Orion"]},
    {"title": "02_Sirius.pdf", "aliases": ["Sirius", "original Sirius"]},
    {"title": "03_Vega.pdf", "aliases": ["Vega"]},
]


@pytest.mark.parametrize(
    "question,title,expected",
    [
        (
            "How many experts are activated per token in Orion and in Sirius?",
            "01_Orion.pdf",
            "For Orion: How many experts are activated per token in Orion?",
        ),
        (
            "How many experts are activated per token in Orion and in Sirius?",
            "02_Sirius.pdf",
            "For Sirius: How many experts are activated per token in Sirius?",
        ),
        (
            "Which normalization do Orion and the original Sirius use?",
            "01_Orion.pdf",
            "For Orion: Which normalization do Orion use?",
        ),
        (
            "What range of model sizes do the Orion paper and the Sirius paper cover?",
            "02_Sirius.pdf",
            "For Sirius: What range of model sizes do the Sirius paper cover?",
        ),
        (
            "What KL coefficient do Orion's RL training and Sirius's RL training use?",
            "02_Sirius.pdf",
            "For Sirius: What KL coefficient do Sirius's RL training use?",
        ),
        (
            "What is the largest model in the Vega experiments, in the Orion paper and in the Sirius paper?",
            "03_Vega.pdf",
            "For Vega: What is the largest model in the Vega experiments?",
        ),
        (
            "Which base models do Orion and Sirius start RL from?",
            "01_Orion.pdf",
            "For Orion: Which base models do Orion start RL from?",
        ),
    ],
)
def test_focus_removes_another_mention_with_its_joining_words(question, title, expected):
    """No 'in in Sirius', 'and in ?', 'the paper and the paper' or orphaned "'s" is left behind."""
    assert planning.focus_question_on_paper(question, title, FOCUS_PAPERS) == expected


def test_focus_keeps_the_users_question_when_removal_would_break_it():
    """'how many parameters does use' searches worse than the original two-clause question."""
    question = "How many parameters does Sirius use, and how much does Orion reduce memory?"
    assert planning.focus_question_on_paper(question, "01_Orion.pdf", FOCUS_PAPERS) == f"For Orion: {question}"


def test_focus_does_not_swallow_the_start_of_the_next_clause():
    """', and how does ...' begins a clause; it is not the list joint of Sirius."""
    question = "How does Orion compare in size with Sirius, and how does the Orion paper position it?"
    focused = planning.focus_question_on_paper(question, "01_Orion.pdf", FOCUS_PAPERS)
    assert "how does the Orion paper position it" in focused


def test_focused_question_preserves_number_punctuation():
    """List cleanup cannot change the value of a numeric condition."""
    papers = [{"title": "Orion.pdf"}, {"title": "Sirius.pdf"}]
    result = planning.focus_question_on_paper(
        "Which data did each of Orion and Sirius use for 1,000 training steps?",
        "Orion.pdf",
        papers,
    )
    assert "1,000" in result


def test_cross_paper_discovery_remains_broad(monkeypatch):
    """Finding other papers must not turn into two exact-paper searches."""
    monkeypatch.setattr(
        planning,
        "request_structured_response",
        lambda schema, *args: schema(
            tasks=[{"question": "Find other papers", "paper": None, "scope": "broad"}]
        ),
    )
    plan = planning.plan_question_tasks(
        "Which other papers compare Orion and Sirius?",
        [],
        [{"title": "Orion.pdf"}, {"title": "Sirius.pdf"}],
    )
    assert len(plan.tasks) == 1
    assert plan.tasks[0].scope == "broad"
