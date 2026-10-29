"""Keep MiniLM inputs understandable without benchmark-specific rewrite rules."""

import pytest

from app import question_planning as planning
from app import retrieval
from app.paper_store import open_chunk_store


def add_definitions(texts, titles=None):
    """Store invented acronym definitions in the disposable test index."""
    open_chunk_store().add(
        ids=[str(i) for i in range(len(texts))],
        documents=texts,
        embeddings=[[1.0] for _ in texts],
        metadatas=[{"title": title} for title in titles or ["A.pdf"] * len(texts)],
    )


def test_expansion_comes_from_documents_not_a_fixed_dictionary():
    """Any acronym with matching initials can expand, including an invented one."""
    add_definitions(["We study Adaptive Memory Routing (AMR)."])
    assert (
        retrieval.expand_defined_abbreviations("How does AMR work?")
        == "How does adaptive memory routing work?"
    )


def test_ambiguous_abbreviation_stays_unchanged_unless_paper_resolves_it():
    """Do not guess between conflicting definitions from different papers."""
    add_definitions(
        ["Adaptive Memory Routing (AMR)", "Automatic Model Review (AMR)"],
        ["A.pdf", "B.pdf"],
    )
    assert retrieval.expand_defined_abbreviations("Explain AMR") == "Explain AMR"
    assert (
        retrieval.expand_defined_abbreviations("Explain AMR", "B.pdf")
        == "Explain automatic model review"
    )


def test_expansion_handles_line_breaks_hyphens_and_function_words():
    """Definitions survive ordinary PDF line wrapping and words such as 'from'."""
    add_definitions(
        [
            "reinforcement learning from human feedback (RLHF)",
            "retrieval-augmented\ngeneration (RAG)",
        ]
    )
    assert (
        retrieval.expand_defined_abbreviations("Compare RLHF and RAG")
        == "Compare reinforcement learning from human feedback and retrieval augmented generation"
    )


def test_unknown_short_name_and_model_identifiers_are_preserved():
    """Leave unknown meanings and hyphenated model names intact."""
    add_definitions(["Generative Pretrained Transformer (GPT)"])
    assert (
        retrieval.expand_defined_abbreviations("Compare GPT-4 with GPT-3 and XYZ")
        == "Compare GPT-4 with GPT-3 and XYZ"
    )


def test_no_abbreviation_skips_definition_lookup(monkeypatch):
    """Ordinary questions do not add a database read."""
    monkeypatch.setattr(
        retrieval, "open_chunk_store", lambda: pytest.fail("No short name")
    )
    assert (
        retrieval.expand_defined_abbreviations("Explain the method")
        == "Explain the method"
    )


def test_expanded_query_reaches_both_retrieval_and_minilm(monkeypatch):
    """Use one search wording for candidate retrieval and reranking, with traceability."""
    add_definitions(["We introduce Sparse Feature Routing (SFR)."])
    calls = []
    candidate = {"id": "child", "title": "A.pdf", "content": "Original evidence"}
    monkeypatch.setattr(
        retrieval,
        "find_hybrid_candidates",
        lambda query, paper: calls.append(("retrieve", query, paper)) or [candidate],
    )
    monkeypatch.setattr(
        retrieval,
        "rerank_passage_candidates",
        lambda query, rows: calls.append(("rerank", query)) or rows,
    )
    trace = {}
    result = retrieval.retrieve_relevant_passages(
        "Explain SFR", paper="A.pdf", diagnostics=trace
    )
    assert calls == [
        ("retrieve", "Explain sparse feature routing", "A.pdf"),
        ("rerank", "Explain sparse feature routing"),
    ]
    assert result[0]["content"] == "Original evidence"
    assert trace["original_query"] == "Explain SFR"
    assert trace["search_query"] == "Explain sparse feature routing"


@pytest.mark.parametrize(
    "question",
    [
        "Which models does 'Memory in Context' evaluate?",
        "Which models does “Memory in Context” evaluate?",
        "Which models does Memory in Context evaluate?",
    ],
)
def test_title_scope_does_not_depend_on_quote_style(monkeypatch, question):
    """All quotation styles use the same catalog-constrained planner interface."""
    monkeypatch.setattr(
        planning,
        "request_structured_response",
        lambda schema, instruction, payload: schema(
            tasks=[
                {
                    "question": payload["question"],
                    "paper": "07_Memory_in_Context.pdf",
                    "scope": "paper",
                }
            ]
        ),
    )
    plan = planning.plan_question_tasks(
        question, [], [{"title": "07_Memory_in_Context.pdf"}]
    )
    assert plan.tasks[0].paper == "07_Memory_in_Context.pdf"
    assert plan.tasks[0].question == question


def test_catalog_schema_rejects_guessed_names_and_numbers(monkeypatch):
    """The actual planner schema permits indexed filenames or null, not repaired aliases."""
    from pydantic import ValidationError

    def plan(schema, instruction, payload):
        """Check the model-facing contract rather than a regex implementation."""
        for invalid in ("01", "Memory in Context", "Unknown.pdf"):
            with pytest.raises(ValidationError):
                schema(
                    tasks=[{"question": "Explain?", "paper": invalid, "scope": "paper"}]
                )
        return schema(
            tasks=[
                {
                    "question": "Explain?",
                    "paper": "01_Memory_in_Context.pdf",
                    "scope": "paper",
                }
            ]
        )

    monkeypatch.setattr(planning, "request_structured_response", plan)
    result = planning.plan_question_tasks(
        "Explain Memory in Context", [], [{"title": "01_Memory_in_Context.pdf"}]
    )
    assert result.tasks[0].paper == "01_Memory_in_Context.pdf"


def test_short_parent_keeps_the_start_even_when_later_child_ranks_first():
    """Keep all list items when the complete section fits the context allowance."""
    from app.paper_store import open_parent_section_store

    text = (
        "Open models: Model Alpha and Model Beta.\n\n"
        + "Training details. " * 30
        + "\n\nClosed models: Model Gamma."
    )
    open_parent_section_store().add(
        ids=["parent"], documents=[text], embeddings=[[1.0]]
    )
    child = {
        "id": "later",
        "parent_id": "parent",
        "content": "Paper: A\nSection: Models\nClosed models: Model Gamma.",
    }
    result = retrieval.expand_parent_section_windows([child], 5)
    assert result[0]["content"] == text
    assert "Model Alpha" in result[0]["content"]
    assert "Model Gamma" in result[0]["content"]


def test_reranker_orders_passages_by_score_without_heading_override(monkeypatch):
    """A heading cannot promote a low-scoring passage over stronger matches."""
    from types import SimpleNamespace

    monkeypatch.setattr(
        retrieval,
        "get_reranker_model",
        lambda: SimpleNamespace(predict=lambda pairs: [8.0, 3.0, -2.0, 9.0]),
    )
    rows = [
        {"id": "results", "content": "Results", "heading": "2.3 Results"},
        {
            "id": "methods",
            "content": "Methods",
            "heading": "2.2 Methods",
            "evidence_type": "prose",
        },
        {"id": "weak", "content": "Unrelated", "heading": "3 Methods"},
        {
            "id": "caption",
            "content": "Caption",
            "heading": "Methods",
            "evidence_type": "caption",
        },
    ]
    ranked = retrieval.rerank_passage_candidates(
        "Which methods does the experiment evaluate?", rows
    )
    assert [row["id"] for row in ranked] == ["caption", "results", "methods", "weak"]


def test_ranked_candidates_are_kept_without_heading_only_filter(monkeypatch):
    """A heading match alone does not remove higher-ranked evidence."""
    rows = [
        {
            "id": "result",
            "title": "A",
            "content": "Results",
            "heading_matches_request": False,
        },
        {
            "id": "method",
            "title": "A",
            "content": "Complete methods",
            "heading_matches_request": True,
        },
    ]
    monkeypatch.setattr(retrieval, "find_hybrid_candidates", lambda *args: rows)
    monkeypatch.setattr(retrieval, "rerank_passage_candidates", lambda *args: rows)
    trace = {}
    result = retrieval.retrieve_relevant_passages(
        "Which methods does it use?", diagnostics=trace
    )
    assert [row["id"] for row in result] == ["result", "method"]


def test_schema_field_descriptions_reach_model_separately_from_input(monkeypatch):
    """Give the model field semantics without putting source data into instructions."""
    from types import SimpleNamespace

    from app import llm
    from app.grounding import DraftAnswer

    calls = []
    client = SimpleNamespace(
        with_structured_output=lambda *args, **kwargs: SimpleNamespace(
            invoke=lambda messages: calls.append(messages)
        )
    )
    monkeypatch.setattr(llm, "get_chat_model", lambda: client)
    llm.request_structured_response(
        DraftAnswer, "Answer from evidence", {"question": "UNTRUSTED QUESTION"}
    )
    assert DraftAnswer.model_fields["answers"].description in calls[0][0][1]
    assert "UNTRUSTED QUESTION" not in calls[0][0][1]
    assert "UNTRUSTED QUESTION" in calls[0][1][1]


def test_mixed_case_word_split_rejoins_without_losing_real_model_hyphens():
    """A mixed-case term can wrap before a lowercase letter, unlike Model-7B."""
    from app.grounding import normalize_evidence_text

    assert (
        normalize_evidence_text("An AL-\niBi method uses Model-\n7B.")
        == "An ALiBi method uses Model-7B."
    )
