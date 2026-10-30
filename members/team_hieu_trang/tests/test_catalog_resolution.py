"""Check alias-free catalog names and evidence-based candidate resolution."""

import pytest

from app import paper_catalog, question_planning, retrieval, services
from app.paper_store import open_chunk_store


@pytest.fixture
def papers():
    """Use model identifiers embedded in titles, with no manually registered aliases."""
    return [
        {
            "title": "05_ORION-3_Language_Models_are_Few-Shot_Learners.pdf",
            "aliases": [],
        },
        {"title": "06_Orion_Observations.pdf", "aliases": []},
    ]


@pytest.mark.parametrize(
    "question", ["Explain ORION-3", "Explain orion 3", "Explain ORION3"]
)
def test_leading_model_identifier_resolves_without_alias(question, papers):
    """Case, hyphens and attached digits do not require an alias entry."""
    result = paper_catalog.find_paper_references(question, papers)
    assert result["titles"] == [papers[0]["title"]]
    assert papers[0]["aliases"] == []


def test_decimal_variant_does_not_match_a_shorter_identifier(papers):
    """ORION-3.5 is not an exact identity match for ORION-3."""
    result = paper_catalog.find_paper_references("Explain ORION-3.5", papers[:1])
    assert not result["titles"]
    assert not result["candidates"]


def test_uppercase_identifier_and_title_before_colon_need_no_alias():
    """Existing catalog text supplies standalone acronyms and named title prefixes."""
    papers = [
        {"title": "04_ZETA_Representations.pdf", "aliases": []},
        {
            "title": "05_Paper.pdf",
            "catalog_title": "Stellar Router: Sparse Routing",
            "aliases": [],
        },
    ]
    assert paper_catalog.find_paper_references("Explain ZETA", papers)["titles"] == [
        papers[0]["title"]
    ]
    assert paper_catalog.find_paper_references("Explain Stellar Router", papers)[
        "titles"
    ] == [papers[1]["title"]]


@pytest.mark.parametrize("question", [
    "What does the Chain-of-Thought paper find?",
    "What does the chain of thought paper find?",
])
def test_unique_title_prefix_resolves_without_an_alias(question):
    papers = [
        {"title": "11_Chain-of-Thought_Prompting.pdf", "aliases": []},
        {"title": "27_Reasoning_Capacity.pdf", "aliases": []},
    ]
    assert paper_catalog.find_paper_references(question, papers)["titles"] == [papers[0]["title"]]


def test_shared_title_prefix_is_ambiguous():
    papers = [{"title": "01_Scaling_Laws_for_Models.pdf"}, {"title": "02_Scaling_Laws_for_Data.pdf"}]
    refs = paper_catalog.find_paper_references("What do Scaling Laws report?", papers)
    assert refs["titles"] == []
    assert refs["ambiguous"] is True
    assert set(refs["candidates"]) == {paper["title"] for paper in papers}


def test_generic_title_prefix_does_not_identify_a_paper():
    papers = [{"title": "01_Language_Models_Learn_Reasoning.pdf"}]
    refs = paper_catalog.find_paper_references("What do language models do?", papers)
    assert refs["titles"] == []


def test_question_title_does_not_produce_incomplete_aliases():
    paper = {"title": "28_Why_Do_Multi-Agent_LLM_Systems_Fail.pdf", "aliases": ["MAST-Data"]}
    names = paper_catalog.catalog_identity_names(paper)
    assert "MAST-Data" in names
    assert "why do" not in names
    assert "why do multi agent llm systems" not in names
    assert paper_catalog.find_paper_references("What does the why do paper say?", [paper])["titles"] == []


def test_prefixes_end_at_natural_boundaries():
    names = paper_catalog.title_prefixes("Neural Machine Translation by Jointly Learning to Align and Translate")
    assert "neural machine translation" in names
    assert "neural machine" not in names
    assert "neural machine translation by jointly" not in names


def test_longer_specific_prefix_beats_shared_shorter_prefix():
    papers = [{"title": "01_Scaling_Laws_for_Models.pdf"}, {"title": "02_Scaling_Laws_for_Data.pdf"}]
    refs = paper_catalog.find_paper_references("What do Scaling Laws for Data report?", papers)
    assert refs["titles"] == [papers[1]["title"]]
    assert not refs["ambiguous"]


def test_distinctive_title_fragment_produces_candidates_without_stopping():
    """A weaker shortened name can reach retrieval rather than requiring an alias edit."""
    papers = [{"title": "01_Chimera_Compute_Efficient_Training.pdf", "aliases": []}]
    refs = paper_catalog.find_paper_references("What does Chimera conclude?", papers)
    assert refs["candidates"] == [papers[0]["title"]]
    assert not refs["titles"]


def test_author_finds_paper_without_matching_its_title():
    """An author reference suggests a paper without pretending the name is unique."""
    papers = [
        {"title": "01_Example.pdf", "authors": ["Ada Lovelace", "Grace Hopper"]},
        {"title": "02_Other.pdf", "authors": ["Alan Turing"]},
    ]
    for question in ("According to Lovelace et al., what changed?", "What did Grace Hopper find?"):
        refs = paper_catalog.find_paper_references(question, papers)
        assert refs["candidates"] == ["01_Example.pdf"]
        assert not refs["titles"]


def test_shared_author_keeps_all_candidate_papers():
    """A surname shared by papers never silently selects the first paper."""
    papers = [
        {"title": "01_Example.pdf", "authors": ["Ada Lovelace"]},
        {"title": "02_Other.pdf", "authors": ["Ada Lovelace"]},
    ]
    refs = paper_catalog.find_paper_references("What does Lovelace report?", papers)
    assert refs["candidates"] == ["01_Example.pdf", "02_Other.pdf"]
    assert not refs["titles"]


def test_et_al_prefers_first_author_over_coauthor():
    """An et al. citation normally refers to a paper's first author."""
    papers = [
        {"title": "01_Example.pdf", "authors": ["Ada Lovelace", "Grace Hopper"]},
        {"title": "02_Other.pdf", "authors": ["Grace Hopper", "Ada Lovelace"]},
    ]
    refs = paper_catalog.find_paper_references("What does Lovelace et al. report?", papers)
    assert refs["candidates"] == ["01_Example.pdf"]


def test_author_candidate_reaches_retrieval_when_planner_is_uncertain(monkeypatch):
    """The planner can search a catalog author match without a model guess."""
    monkeypatch.setattr(
        question_planning, "request_structured_response",
        lambda schema, *args: schema(tasks=[
            {"question": "According to Lovelace et al., what changed?", "paper": None, "scope": "unresolved"}
        ]),
    )
    papers = [{"title": "01_Example.pdf", "authors": ["Ada Lovelace"]}]
    task = question_planning.plan_question_tasks(
        "According to Lovelace et al., what changed?", [], papers
    ).tasks[0]
    assert task.candidate_papers == ["01_Example.pdf"]
    assert task.scope == "paper"


def test_alias_free_resolution_overrides_wrong_planner_choice(monkeypatch, papers):
    """The actual user identifier wins over a different valid catalog filename."""

    def plan(schema, instruction, payload):
        """The native schema keeps Python candidate lists out of model output."""
        assert (
            "candidate_papers"
            not in schema.model_json_schema()["$defs"]["CatalogSearchTask"][
                "properties"
            ]
        )
        return schema(
            tasks=[
                {
                    "question": "A proposed finding, not a search question.",
                    "paper": papers[1]["title"],
                    "scope": "paper",
                }
            ]
        )

    monkeypatch.setattr(question_planning, "request_structured_response", plan)
    result = question_planning.plan_question_tasks("Explain ORION-3", [], papers)
    assert result.tasks[0].paper == papers[0]["title"]
    assert result.tasks[0].question == "Explain ORION-3"


def test_partial_title_reaches_retrieval_even_if_planner_says_unresolved(monkeypatch):
    """Only absence of catalog candidates stops a specific-paper search immediately."""
    title = "01_Chimera_Compute_Efficient_Training.pdf"
    monkeypatch.setattr(
        question_planning,
        "request_structured_response",
        lambda schema, *args: schema(
            tasks=[{"question": "Chimera?", "paper": None, "scope": "unresolved"}]
        ),
    )
    task = question_planning.plan_question_tasks(
        "Explain Chimera", [], [{"title": title}]
    ).tasks[0]
    assert task.scope == "paper"
    assert task.paper is None
    assert task.candidate_papers == [title]


def test_duplicate_alias_keeps_all_candidates_not_the_planners_guess(monkeypatch):
    """An ambiguous alias must not silently select the first or model-preferred owner."""
    papers = [
        {"title": "A.pdf", "aliases": ["Shared Name"]},
        {"title": "B.pdf", "aliases": ["Shared Name"]},
    ]
    monkeypatch.setattr(
        question_planning,
        "request_structured_response",
        lambda schema, *args: schema(
            tasks=[
                {"question": "Explain Shared Name", "paper": "A.pdf", "scope": "paper"}
            ]
        ),
    )
    task = question_planning.plan_question_tasks(
        "Explain Shared Name", [], papers
    ).tasks[0]
    assert task.paper is None
    assert task.candidate_papers == ["A.pdf", "B.pdf"]


@pytest.mark.parametrize(
    "titles,expected", [(["A.pdf"], "A.pdf"), (["A.pdf", "B.pdf"], None), ([], None)]
)
def test_candidate_search_requires_one_accepted_paper(monkeypatch, titles, expected):
    """Multiple plausible papers yield no mixed evidence for a specific-paper task."""
    rows = [
        {"id": str(i), "title": title, "content": "Relevant evidence"}
        for i, title in enumerate(titles)
    ]
    calls = []
    monkeypatch.setattr(
        retrieval, "expand_defined_abbreviations", lambda query, paper: query
    )
    monkeypatch.setattr(
        retrieval,
        "find_hybrid_candidates",
        lambda query, paper: calls.append(paper) or rows,
    )
    monkeypatch.setattr(
        retrieval, "rerank_passage_candidates", lambda query, rows: rows
    )
    monkeypatch.setattr(
        retrieval, "expand_parent_section_windows", lambda rows, limit: rows
    )
    trace = {}
    evidence = retrieval.retrieve_relevant_passages(
        "Explain Shared Name",
        paper=["A.pdf", "B.pdf"],
        diagnostics=trace,
        resolve_candidates=True,
    )
    assert calls == [["A.pdf", "B.pdf"]]
    assert trace["resolved_paper"] == expected
    assert bool(evidence) == bool(expected)
    assert trace["scope_ambiguous"] == (len(titles) > 1)
    assert trace["accepted_count"] == len(rows)


def test_candidate_filter_excludes_unrelated_papers_before_search(monkeypatch):
    """BM25 and vector search use the same candidate-title restriction."""
    store = open_chunk_store()
    store.add(
        ids=["a", "b", "c"],
        documents=["routing"] * 3,
        embeddings=[[1.0]] * 3,
        metadatas=[{"title": title} for title in ["A.pdf", "B.pdf", "C.pdf"]],
    )
    calls = []
    monkeypatch.setattr(
        retrieval,
        "find_vector_candidates",
        lambda query, **kwargs: calls.append(kwargs["paper"]) or [],
    )
    candidates = retrieval.find_hybrid_candidates("routing", ["A.pdf", "B.pdf"])
    assert {row["title"] for row in candidates} == {"A.pdf", "B.pdf"}
    assert calls == [["A.pdf", "B.pdf"]]


def test_vector_candidate_filter_uses_one_embedding_call(monkeypatch):
    """Candidate discovery does not embed the same question once per paper."""
    from types import SimpleNamespace

    calls = []
    embedding_calls = []
    monkeypatch.setattr(
        retrieval,
        "embed_search_question",
        lambda query: embedding_calls.append(query) or [1.0],
    )
    monkeypatch.setattr(
        retrieval,
        "open_chunk_store",
        lambda: SimpleNamespace(
            count=lambda: 3,
            query=lambda **kwargs: (
                calls.append(kwargs)
                or {
                    "ids": [[]],
                    "documents": [[]],
                    "metadatas": [[]],
                    "distances": [[]],
                }
            ),
        ),
    )
    assert retrieval.find_vector_candidates("routing", paper=["A.pdf", "B.pdf"]) == []
    assert len(embedding_calls) == 1
    assert calls[0]["where"] == {"title": {"$in": ["A.pdf", "B.pdf"]}}


def test_resolved_candidate_becomes_the_final_task_scope(monkeypatch):
    """Generation and citation checks see the discovered paper's identity."""
    task = {
        "question": "Explain Chimera",
        "paper": None,
        "candidate_papers": ["A.pdf"],
        "scope": "paper",
    }

    def retrieve(question, paper, diagnostics, resolve_candidates=False):
        """Simulate one accepted candidate document, without a model call."""
        assert paper == ["A.pdf"]
        assert resolve_candidates
        diagnostics["resolved_paper"] = "A.pdf"
        return [{"id": "child", "title": "A.pdf", "content": "Relevant evidence"}]

    monkeypatch.setattr(services, "retrieve_relevant_passages", retrieve)
    result = services.retrieve_task_evidence({"tasks": [task], "retrieval_trace": {}})
    assert result["tasks"][0]["paper"] == "A.pdf"
    assert task["paper"] is None  # The saved plan was not silently mutated.


def test_broad_search_can_still_use_multiple_documents(monkeypatch):
    """Other-paper discovery is deliberately broad and is not treated as identity ambiguity."""
    rows = [
        {"id": title, "title": title, "content": "Relevant"}
        for title in ("A.pdf", "B.pdf")
    ]
    monkeypatch.setattr(
        retrieval, "expand_defined_abbreviations", lambda query, paper: query
    )
    monkeypatch.setattr(retrieval, "find_hybrid_candidates", lambda *args: rows)
    monkeypatch.setattr(retrieval, "rerank_passage_candidates", lambda *args: rows)
    monkeypatch.setattr(
        retrieval, "expand_parent_section_windows", lambda rows, limit: rows
    )
    assert (
        len(retrieval.retrieve_relevant_passages("Which other papers discuss routing?"))
        == 2
    )


def test_alias_cannot_steal_an_inferred_catalog_identifier():
    """Optional aliases cannot take ownership of another paper's existing short name."""
    paper_catalog.refresh_local_catalog(
        [
            {"id": "a", "title": "01_ORION-3_Research.pdf"},
            {"id": "b", "title": "02_Other_Research.pdf"},
        ]
    )
    with pytest.raises(ValueError, match="another paper"):
        paper_catalog.set_paper_aliases("b", ["ORION3"])


def test_ambiguous_evidence_returns_an_honest_abstention():
    """Several plausible documents are distinguished from an empty search result."""
    result = services.build_chat_response(
        {
            "evidence": [],
            "tasks": [],
            "retrieval_trace": {"tasks": [{"trace": {"scope_ambiguous": True}}]},
        }
    )["result"]
    assert result["status"] == "abstained"
    assert "Several indexed papers" in result["answer"]
    assert result["sources"] == []


def test_joint_task_keeps_both_explicitly_identified_papers(monkeypatch):
    """A joint comparison may use both named papers without identity ambiguity."""
    papers = [
        {"title": "01_ORION-3_Research.pdf"},
        {"title": "02_SIRIUS-4_Research.pdf"},
    ]
    monkeypatch.setattr(
        question_planning,
        "request_structured_response",
        lambda schema, *args: schema(
            tasks=[
                {
                    "question": "Compare both models",
                    "paper": None,
                    "scope": "unresolved",
                }
            ]
        ),
    )
    result = question_planning.plan_question_tasks(
        "Compare ORION-3 and SIRIUS-4", [], papers
    )
    assert result.tasks[0].paper == [row["title"] for row in papers]
    assert result.tasks[0].candidate_papers == []
    assert result.tasks[0].scope == "paper"


def test_verified_multi_paper_scope_does_not_trigger_ambiguity(monkeypatch):
    """Named comparison sources are allowed together; only uncertain identities need resolution."""
    rows = [
        {"id": title, "title": title, "content": "Relevant"}
        for title in ("A.pdf", "B.pdf")
    ]
    monkeypatch.setattr(
        retrieval, "expand_defined_abbreviations", lambda query, paper: query
    )
    monkeypatch.setattr(
        retrieval,
        "find_hybrid_candidates",
        lambda query, paper: [row for row in rows if row["title"] == paper],
    )
    monkeypatch.setattr(
        retrieval, "rerank_passage_candidates", lambda query, candidates: candidates
    )
    monkeypatch.setattr(
        retrieval, "expand_parent_section_windows", lambda rows, limit: rows
    )
    trace = {}
    result = retrieval.retrieve_relevant_passages(
        "Compare both", paper=["A.pdf", "B.pdf"], diagnostics=trace
    )
    assert len(result) == 2
    assert [row["title"] for row in result] == ["A.pdf", "B.pdf"]
    assert not trace["scope_ambiguous"]
    assert len(trace["paper_searches"]) == 2


def test_joint_search_interleaves_each_paper_and_retains_audit(monkeypatch):
    """A verified comparison searches each title and gives both an early result."""
    calls = []

    def find(query, paper):
        calls.append(paper)
        return [
            {"id": f"{paper}-{i}", "title": paper, "content": f"Evidence {i}"}
            for i in (1, 2)
        ]

    monkeypatch.setattr(
        retrieval, "expand_defined_abbreviations", lambda query, _: query
    )
    monkeypatch.setattr(retrieval, "find_hybrid_candidates", find)
    monkeypatch.setattr(
        retrieval, "rerank_passage_candidates", lambda query, candidates: candidates
    )
    monkeypatch.setattr(
        retrieval, "expand_parent_section_windows", lambda rows, limit: rows[:limit]
    )
    trace = {"capture_evidence": True}
    result = retrieval.retrieve_relevant_passages(
        "Compare both", paper=["A.pdf", "B.pdf"], diagnostics=trace
    )
    assert calls == ["A.pdf", "B.pdf"]
    assert [row["id"] for row in result] == ["A.pdf-1", "B.pdf-1", "A.pdf-2", "B.pdf-2"]
    assert trace["candidate_count"] == 4
    assert [row["id"] for row in trace["reranked_passages"]] == [
        row["id"] for row in result
    ]


def test_approved_verified_multi_paper_plan_is_validated(monkeypatch):
    """Saved plans can contain an explicit paper set without an unhashable-list error."""
    monkeypatch.setattr(
        services,
        "list_indexed_papers",
        lambda: [{"title": "A.pdf"}, {"title": "B.pdf"}],
    )
    result = services.plan_search_tasks(
        {
            "message": "Compare both",
            "approved_plan": {
                "tasks": [
                    {
                        "question": "Compare both",
                        "paper": ["A.pdf", "B.pdf"],
                        "scope": "paper",
                    }
                ]
            },
        }
    )
    assert not result["clarification"]
    assert result["tasks"][0]["paper"] == ["A.pdf", "B.pdf"]
def test_author_opening_is_a_candidate_not_a_verified_title(tmp_path, monkeypatch):
    import json
    from app import paper_catalog

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"placeholder")
    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps([{"title": "paper.pdf", "path": str(pdf)}]))
    monkeypatch.setattr(paper_catalog, "SOURCE_MANIFEST_PATH", manifest)
    monkeypatch.setattr(paper_catalog, "read_paper_opening", lambda *args: "title\nada lovelace\nabstract")
    result = paper_catalog.find_paper_references("What did Lovelace et al. find?", [{"title": "paper.pdf"}])
    assert result["candidates"] == ["paper.pdf"]
    assert result["titles"] == []
