"""Check paper identity and partial answers using invented papers, not test answers."""

import pytest

from app import grounding, paper_catalog, question_planning, services
from app.question_planning import QuestionPlan, SearchTask

PAPERS = [
    {
        "id": "drive-a",
        "title": "01_Stellar_Routing.pdf",
        "paper_number": 1,
        "aliases": ["Orion"],
        "catalog_title": "Stellar Routing",
    },
    {
        "id": "drive-b",
        "title": "02_Stellar_Extended.pdf",
        "paper_number": 2,
        "aliases": ["Orion Extended"],
    },
]


def test_alias_changes_invalidate_evaluation_resume_without_changing_index():
    """An identity change affects retrieval configuration, not stored PDF vectors."""
    from app.evaluate import build_evaluation_configuration

    paper_catalog.refresh_local_catalog(PAPERS)
    before = build_evaluation_configuration()
    paper_catalog.set_paper_aliases("drive-a", ["Orion"])
    after = build_evaluation_configuration()
    assert before["catalog_aliases_sha256"] != after["catalog_aliases_sha256"]
    assert before["index_sha256"] == after["index_sha256"]


def test_parallel_alias_reads_return_independent_values():
    """Concurrent requests only read aliases and cannot mutate another request's list."""
    from concurrent.futures import ThreadPoolExecutor

    paper_catalog.refresh_local_catalog(PAPERS)
    paper_catalog.set_paper_aliases("drive-a", ["Orion"])
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(lambda _: paper_catalog.load_local_aliases(), range(4)))
    rows[0]["drive-a"].append("Changed in one request")
    assert all(row["drive-a"] == ["Orion"] for row in rows[1:])
    assert paper_catalog.load_local_aliases()["drive-a"] == ["Orion"]


@pytest.mark.parametrize(
    "question,expected",
    [
        ("Explain ORION", [PAPERS[0]["title"]]),
        ("Read Stellar Routing", [PAPERS[0]["title"]]),
        ("Read 01_Stellar_Routing.pdf", [PAPERS[0]["title"]]),
        ("Explain paper #1", [PAPERS[0]["title"]]),
        ("Explain Orion Extended", [PAPERS[1]["title"]]),
        ("Compare Orion and Orion Extended", [p["title"] for p in PAPERS]),
        ("What happened in 2024 with 1 router?", []),
        ("Explain Orionism", []),
    ],
)
def test_names_numbers_and_overlaps(question, expected):
    """Whole names and labeled numbers resolve; overlapping aliases keep their owner."""
    result = paper_catalog.find_paper_references(question, PAPERS)
    assert result == {"titles": expected, "candidates": [], "ambiguous": False}


def test_ambiguous_alias_does_not_choose_an_owner():
    """Duplicate ownership must never silently select the first indexed paper."""
    papers = [dict(p, aliases=["Shared Name"]) for p in PAPERS]
    assert paper_catalog.find_paper_references("Explain Shared Name", papers) == {
        "titles": [],
        "candidates": [row["title"] for row in papers],
        "ambiguous": True,
    }


def test_local_refresh_preserves_aliases_and_excludes_removed_papers():
    """Refreshing CSV/index identity fields must preserve local aliases by document ID."""
    paper_catalog.refresh_local_catalog(PAPERS)
    paper_catalog.set_paper_aliases("drive-a", ["Orion"])
    paper_catalog.set_paper_aliases("drive-b", ["Orion Extended"])
    paper_catalog.refresh_local_catalog([dict(PAPERS[0], catalog_title="New Title")])
    assert paper_catalog.load_local_aliases() == {"drive-a": ["Orion"]}
    paper_catalog.refresh_local_catalog(PAPERS)
    assert paper_catalog.load_local_aliases()["drive-b"] == ["Orion Extended"]


@pytest.mark.parametrize(
    "owner,aliases",
    [
        ("absent", ["Unique"]),
        ("drive-a", ["x"]),
        ("drive-b", ["Stellar Routing"]),
    ],
)
def test_alias_maintenance_rejects_unknown_owner_short_and_colliding_names(
    owner, aliases
):
    """Maintenance cannot add an unindexed owner or steal another paper's identity."""
    paper_catalog.refresh_local_catalog(PAPERS)
    with pytest.raises(ValueError):
        paper_catalog.set_paper_aliases(owner, aliases)


def test_user_alias_overrides_wrong_model_paper_choice(monkeypatch):
    """An allowed filename is insufficient when it identifies the wrong subject."""
    monkeypatch.setattr(
        question_planning,
        "request_structured_response",
        lambda schema, *args: schema(
            tasks=[
                {
                    "question": "Explain Orion training",
                    "paper": PAPERS[1]["title"],
                    "scope": "paper",
                }
            ]
        ),
    )
    plan = question_planning.plan_question_tasks("Explain Orion training", [], PAPERS)
    assert plan.tasks[0].paper == PAPERS[0]["title"]


def test_planner_invented_reference_cannot_establish_scope(monkeypatch):
    """A title supplied only by the model is not an explicit user paper reference."""
    monkeypatch.setattr(
        question_planning,
        "request_structured_response",
        lambda schema, *args: schema(
            tasks=[
                {
                    "question": "Explain Orion",
                    "paper": PAPERS[0]["title"],
                    "scope": "paper",
                }
            ]
        ),
    )
    plan = question_planning.plan_question_tasks("Explain Unknown Model", [], PAPERS)
    assert plan.tasks[0].scope == "unresolved"
    assert plan.tasks[0].paper is None


def test_deliberately_broad_discovery_stays_broad(monkeypatch):
    """Mentioning a known model must not restrict a search for other papers."""
    monkeypatch.setattr(
        question_planning,
        "request_structured_response",
        lambda schema, *args: schema(
            tasks=[
                {
                    "question": "Which other papers discuss Orion?",
                    "paper": PAPERS[0]["title"],
                    "scope": "broad",
                }
            ]
        ),
    )
    task = question_planning.plan_question_tasks(
        "Which other papers discuss Orion?", [], PAPERS
    ).tasks[0]
    assert task.scope == "broad" and task.paper is None


@pytest.mark.parametrize(
    "role,expected", [("user", "paper"), ("assistant", "unresolved")]
)
def test_followup_identity_comes_from_user_history(monkeypatch, role, expected):
    """Generated answer text alone cannot establish a paper's identity for follow-ups."""
    monkeypatch.setattr(
        question_planning,
        "request_structured_response",
        lambda schema, *args: schema(
            tasks=[
                {
                    "question": "Explain Orion limitations",
                    "paper": PAPERS[0]["title"],
                    "scope": "paper",
                }
            ]
        ),
    )
    task = question_planning.plan_question_tasks(
        "What are its limitations?",
        [{"role": role, "content": "Explain Orion"}],
        PAPERS,
    ).tasks[0]
    assert task.scope == expected


def test_unresolved_part_keeps_valid_part_and_cannot_borrow_its_evidence(monkeypatch):
    """An unavailable identity stays missing while independent supported facts survive."""
    monkeypatch.setattr(
        question_planning,
        "request_structured_response",
        lambda schema, *args: schema(
            tasks=[
                {
                    "question": "Explain Unknown Model",
                    "paper": None,
                    "scope": "unresolved",
                },
                {
                    "question": "Explain Orion",
                    "paper": PAPERS[0]["title"],
                    "scope": "paper",
                },
            ]
        ),
    )
    plan = question_planning.plan_question_tasks(
        "Explain Unknown Model and Orion", [], PAPERS
    )
    tasks = [task.model_dump() for task in plan.tasks]
    calls = []
    monkeypatch.setattr(
        services,
        "retrieve_relevant_passages",
        lambda question, **kwargs: (
            calls.append(question)
            or [
                {
                    "id": "child-a",
                    "title": PAPERS[0]["title"],
                    "content": "Orion uses routing.",
                }
            ]
        ),
    )
    retrieved = services.retrieve_task_evidence({"tasks": tasks, "retrieval_trace": {}})
    assert calls == ["Explain Orion"]
    draft = grounding.DraftAnswer(
        answers=[
            grounding.DetailAnswer(
                availability="available",
                answer="Orion uses routing.",
                source_ids=["S1"],
            ),
            grounding.DetailAnswer(
                availability="available",
                answer="Orion uses routing.",
                source_ids=["S1"],
            ),
        ]
    )
    checked = grounding.verify_answer_citations(draft, retrieved["evidence"], tasks)
    result = grounding.format_cited_answer(checked, retrieved["evidence"], tasks)
    assert result["status"] == "partial"
    assert result["missing_tasks"] == ["Explain Unknown Model"]
    assert checked["accepted"][0]["task_index"] == 1


def test_unresolved_parts_do_not_consume_search_slots(monkeypatch):
    """Many missing identities cannot crowd a later resolvable task out of its batch."""
    tasks = [SearchTask(question=f"Missing {i}", scope="unresolved") for i in range(8)]
    tasks.append(
        SearchTask(question="Explain Orion", paper=PAPERS[0]["title"], scope="paper")
    )
    monkeypatch.setattr(services, "list_indexed_papers", lambda: PAPERS)
    monkeypatch.setattr(
        services, "plan_question_tasks", lambda *args: QuestionPlan(tasks=tasks)
    )
    result = services.plan_search_tasks({"message": "Explain these papers"})
    assert not result["deferred_tasks"]
    assert result["retrieval_trace"]["executed_task_count"] == 1
    assert result["retrieval_trace"]["unresolved_task_count"] == 8
