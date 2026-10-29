"""Check that independent evaluation questions never require another user turn."""

import ast
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from time import sleep
from types import SimpleNamespace

import pytest
from answer_fixtures import cited_draft
from fastapi.testclient import TestClient

from app import main, services
from app.db import open_database
from app.question_planning import QuestionPlan, SearchTask


def test_multipart_question_executes_without_approval(monkeypatch):
    """Planning still happens, but all three tasks run without a browser action."""
    monkeypatch.setattr(
        services,
        "plan_question_tasks",
        lambda *args: QuestionPlan(
            route="multipart",
            tasks=[SearchTask(question=f"Part {i}") for i in range(3)],
        ),
    )
    searches = []

    def search(question, **kwargs):
        """Provide independent evidence for each planned part."""
        searches.append(question)
        return [
            {"id": question, "title": "Paper", "content": "Training uses examples."}
        ]

    def generate(question, evidence, tasks):
        """Return one verifiable claim for each part without a live model."""
        return cited_draft(
            "\n\n".join(
                [f"{'Training uses examples.'} [{row['id']}]" for row in evidence]
            )
        )

    monkeypatch.setattr(services, "retrieve_relevant_passages", search)
    monkeypatch.setattr(services, "generate_grounded_draft", generate)
    result = services.answer_research_question(
        "Explain three parts", [], require_review=False
    )
    assert searches == ["Part 0", "Part 1", "Part 2"]
    assert result["status"] == "answered"
    assert "plan" not in result
    assert len(result["sources"]) == 3


def test_ambiguous_automatic_question_abstains_without_clarification(monkeypatch):
    """An unavailable evaluator cannot answer a follow-up clarification request."""
    monkeypatch.setattr(
        services,
        "plan_question_tasks",
        lambda *args: QuestionPlan(clarification="Which paper?"),
    )
    monkeypatch.setattr(
        services,
        "retrieve_relevant_passages",
        lambda *args, **kwargs: pytest.fail("Unresolved scope must not search"),
    )
    result = services.answer_research_question(
        "Explain that paper", [], require_review=False
    )
    assert result["status"] == "abstained"
    assert result["abstention_reason"] == "unresolved_scope"
    assert result["sources"] == []
    assert "plan" not in result
    assert "Which paper?" not in result["answer"]


@pytest.mark.parametrize("has_evidence", [True, False])
def test_large_automatic_plan_reports_unexecuted_parts(monkeypatch, has_evidence):
    """Keep execution bounded and expose every unfinished part without a proposal."""
    monkeypatch.setattr(
        services,
        "plan_question_tasks",
        lambda *args: QuestionPlan(
            route="multipart",
            tasks=[SearchTask(question=f"Part {i}") for i in range(8)],
        ),
    )
    calls = []

    def search(question, **kwargs):
        """Return either supported evidence or no matches for the same plan."""
        calls.append(question)
        return (
            [{"id": question, "title": "Paper", "content": "Evidence exists."}]
            if has_evidence
            else []
        )

    monkeypatch.setattr(services, "retrieve_relevant_passages", search)
    monkeypatch.setattr(
        services,
        "generate_grounded_draft",
        lambda *args: cited_draft("\n\n".join([f"{'Evidence exists.'} [{'S1'}]"])),
    )
    result = services.answer_research_question(
        "Explain eight parts", [], require_review=False
    )
    assert calls == [f"Part {i}" for i in range(5)]
    assert result["unexecuted_tasks"] == ["Part 5", "Part 6", "Part 7"]
    assert set(result["unexecuted_tasks"]) <= set(result["missing_tasks"])
    assert result["status"] == ("partial" if has_evidence else "abstained")
    assert result["retrieval_trace"]["unexecuted_task_count"] == 3
    assert "plan" not in result
    assert "approve" not in result["answer"].lower()


def load_team_gateway():
    """Load only our gateway function; avoid running other teams' imports and demos."""
    path = Path(__file__).resolve().parents[3] / "api_view.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "team1_agent"
    )
    namespace = {}
    exec(  # noqa: S102 -- Execute only our trusted gateway function from the repository.
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace["team1_agent"]


@pytest.fixture
def gateway_stores(monkeypatch):
    """Replace Chroma startup so adapter tests never touch the real index."""
    stores = SimpleNamespace(
        open_chunk_store=lambda: None,
        open_parent_section_store=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "members.team_hieu_trang.app.paper_store", stores)
    return stores


def test_gateway_passes_independent_questions_and_disables_review(
    monkeypatch, gateway_stores
):
    """The competition entry point calls the current function with no saved history."""
    seen = []

    def answer(question, **kwargs):
        """Record inputs and return repeated source cards to check output conversion."""
        seen.append((question, kwargs))
        return {
            "answer": "Supported answer",
            "sources": [{"title": "Paper.pdf"}, {"title": "Paper.pdf"}],
        }

    monkeypatch.setitem(
        sys.modules,
        "members.team_hieu_trang.app.services",
        SimpleNamespace(answer_research_question=answer),
    )
    gateway = load_team_gateway()
    for question in ("First question", "Unrelated question"):
        assert gateway(question) == {
            "answer": "Supported answer",
            "sources": ["Paper.pdf"],
        }
    assert seen == [
        (question, {"history": [], "capture_evidence": False, "require_review": False})
        for question in ("First question", "Unrelated question")
    ]


def test_gateway_handles_local_service_failure(monkeypatch, gateway_stores):
    """A normal model outage returns a plain answer instead of crashing the gateway."""

    def fail(*args, **kwargs):
        """Simulate a local runtime failure without contacting a model."""
        raise RuntimeError("private diagnostic")

    monkeypatch.setitem(
        sys.modules,
        "members.team_hieu_trang.app.services",
        SimpleNamespace(answer_research_question=fail),
    )
    result = load_team_gateway()("Question")
    assert "I don't know" in result["answer"]
    assert "private diagnostic" not in result["answer"]
    assert result["sources"] == []


def test_gateway_cold_start_initializes_once_and_answers_concurrently(
    monkeypatch, gateway_stores
):
    """Four first requests share initialization but can overlap in the agent."""
    calls = []
    ready = Barrier(4)
    answering = Barrier(4)

    def open_chunks():
        """Keep initialization open briefly so all requests contend for it."""
        calls.append("chunks")
        sleep(0.05)

    def answer(question, **kwargs):
        """Fail if the startup lock is still held while answering questions."""
        answering.wait(timeout=10)
        return {"answer": question, "sources": []}

    gateway_stores.open_chunk_store = open_chunks
    gateway_stores.open_parent_section_store = lambda: calls.append("sections")
    monkeypatch.setitem(
        sys.modules,
        "members.team_hieu_trang.app.services",
        SimpleNamespace(answer_research_question=answer),
    )
    gateway = load_team_gateway()

    def ask(question):
        """Start all four requests together to exercise the cold path."""
        ready.wait(timeout=10)
        return gateway(question)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(ask, ["First", "Second", "Third", "Fourth"]))
    assert calls == ["chunks", "sections"]
    assert [row["answer"] for row in results] == ["First", "Second", "Third", "Fourth"]


def test_gateway_retries_failed_index_initialization(monkeypatch, gateway_stores):
    """A failed first attempt must not mark Chroma ready or block the next call."""
    attempts = []

    def open_chunks():
        """Fail only the first initialization attempt."""
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("temporary initialization failure")

    gateway_stores.open_chunk_store = open_chunks
    monkeypatch.setitem(
        sys.modules,
        "members.team_hieu_trang.app.services",
        SimpleNamespace(
            answer_research_question=lambda *args, **kwargs: {"answer": "Recovered"}
        ),
    )
    gateway = load_team_gateway()
    assert "could not complete" in gateway("First")["answer"]
    assert gateway("Second")["answer"] == "Recovered"
    assert len(attempts) == 2


@pytest.mark.parametrize(
    "outcome", ["multipart", "ambiguous", "no_evidence", "large_plan"]
)
def test_local_swagger_runs_graph_to_completion_without_approval(monkeypatch, outcome):
    """Exercise the real HTTP route and graph for each branch that could pause a run."""
    count = 8 if outcome == "large_plan" else 2
    plan = (
        QuestionPlan(clarification="Which paper?")
        if outcome == "ambiguous"
        else QuestionPlan(
            route="multipart",
            tasks=[SearchTask(question=f"Part {i}") for i in range(count)],
        )
    )
    inputs = []

    def planning(question, history, papers):
        """Record independent inputs instead of making a live planning model call."""
        inputs.append((question, history))
        return plan

    def search(question, **kwargs):
        """Return source evidence unless the test needs the no-evidence branch."""
        return (
            []
            if outcome == "no_evidence"
            else [
                {
                    "id": question,
                    "title": "Paper.pdf",
                    "content": "Training uses examples.",
                }
            ]
        )

    def draft(question, evidence, tasks):
        """Give verification a supported claim for every executed task."""
        return cited_draft(
            "\n\n".join(
                [f"{'Training uses examples.'} [{row['id']}]" for row in evidence]
            )
        )

    monkeypatch.setattr(services, "plan_question_tasks", planning)
    monkeypatch.setattr(services, "retrieve_relevant_passages", search)
    monkeypatch.setattr(services, "generate_grounded_draft", draft)
    with TestClient(main.app) as client:
        assert client.get("/docs").status_code == 200
        assert "/team1/rag-agent" in client.get("/openapi.json").json()["paths"]
        for question in ("First multipart question", "Another independent question"):
            response = client.post("/team1/rag-agent", json={"question": question})
            assert response.status_code == 200
            result = response.json()
            assert set(result) == {"answer", "sources"}
            assert "approve" not in result["answer"].lower()
            assert "Which paper?" not in result["answer"]
            if outcome == "large_plan":
                assert "execution budget" in result["answer"]
            assert result["sources"] == (
                [] if outcome in {"ambiguous", "no_evidence"} else ["Paper.pdf"]
            )
    assert inputs == [
        ("First multipart question", []),
        ("Another independent question", []),
    ]
    with open_database() as database:
        assert database.execute("SELECT count(*) FROM chat_sessions").fetchone()[0] == 0
        assert database.execute("SELECT count(*) FROM pending_plans").fetchone()[0] == 0


@pytest.mark.parametrize("question", ["", "   ", "x" * 2001])
def test_swagger_rejects_invalid_question_before_model_call(monkeypatch, question):
    """Input validation remains active; it never starts a planning approval loop."""
    monkeypatch.setattr(
        main,
        "answer_research_question",
        lambda *args, **kwargs: pytest.fail("Invalid input must not reach the model"),
    )
    with TestClient(main.app) as client:
        assert (
            client.post("/team1/rag-agent", json={"question": question}).status_code
            == 422
        )


def test_four_swagger_requests_keep_separate_plans_and_evidence(monkeypatch):
    """Four graph executions can reach generation together without sharing state."""
    together = Barrier(4)
    monkeypatch.setattr(services, "list_indexed_papers", list)
    monkeypatch.setattr(
        services,
        "plan_question_tasks",
        lambda question, *args: QuestionPlan(tasks=[SearchTask(question=question)]),
    )
    monkeypatch.setattr(
        services,
        "retrieve_relevant_passages",
        lambda question, **kwargs: [
            {
                "id": question,
                "title": f"{question}.pdf",
                "content": f"Evidence for {question}.",
            }
        ],
    )

    def draft(question, evidence, tasks):
        """Wait for all four requests and verify that this request kept its own inputs."""
        together.wait(timeout=10)
        assert tasks[0]["question"] == question
        assert evidence[0]["title"] == f"{question}.pdf"
        text = f"Evidence for {question}."
        return cited_draft("\n\n".join([f"{text} [{'S1'}]"]))

    def ask(question):
        """Send an independent HTTP request from its own client thread."""
        with TestClient(main.app) as client:
            return client.post("/team1/rag-agent", json={"question": question})

    monkeypatch.setattr(services, "generate_grounded_draft", draft)
    questions = [f"Request {i}" for i in range(4)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(ask, questions))
    for question, response in zip(questions, responses):
        assert response.status_code == 200
        assert response.json() == {
            "answer": f"Evidence for {question}. [S1]",
            "sources": [f"{question}.pdf"],
        }
