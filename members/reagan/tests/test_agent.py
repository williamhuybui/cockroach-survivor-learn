from types import SimpleNamespace
import pytest
from app import services
from app.retrieval import fuse_results


def test_no_evidence_skips_generation(monkeypatch):
    monkeypatch.setattr(services, "search", lambda *args: [])
    monkeypatch.setattr(services, "generate", lambda *args: pytest.fail("No evidence must not call model"))
    result = services.respond("What is Cockroach?", [], "user")
    assert "No matching" in result["answer"]
    assert result["sources"] == []


def test_graph_returns_only_valid_citations(monkeypatch):
    evidence = [{"id": "s1", "content": "Alembic uses migrations", "title": "Notes"}]
    monkeypatch.setattr(services, "search", lambda *args: evidence)
    monkeypatch.setattr(services, "generate", lambda *args: SimpleNamespace(answer="Uses migrations [S1]", source_ids=["S1"]))
    result = services.respond("What is Alembic?", [], "user")
    assert result["sources"] == [{"id": "S1", "title": "Notes"}]


def test_invented_citation_is_rejected(monkeypatch):
    monkeypatch.setattr(services, "search", lambda *args: [{"id": "s1", "content": "Notes"}])
    monkeypatch.setattr(services, "generate", lambda *args: SimpleNamespace(answer="Invented", source_ids=["missing"]))
    with pytest.raises(ValueError, match="citation"):
        services.respond("A question", [], "user")


def test_evidence_uses_short_stable_source_ids():
    evidence = [{"id": "long-database-id", "content": "Notes"}, {"id": "another-id", "content": "More"}]
    assert [row["id"] for row in services.label_evidence(evidence)] == ["S1", "S2"]


def test_quiz_bypasses_chat_generation(monkeypatch):
    monkeypatch.setattr(services, "make_quiz", lambda request, user: {"count": request.count, "owner": user})
    assert services.respond('/quiz "Alembic" hard 3', [], "member") == {"count": 3, "owner": "member"}


def test_last_week_uses_date_retrieval(monkeypatch):
    calls = []
    monkeypatch.setattr(services, "week_materials", lambda start, end: calls.append((start, end)) or [])
    monkeypatch.setattr(services, "search", lambda *args: pytest.fail("Must use dated retrieval"))
    services.respond("What did we cover last week?", [], "user")
    assert len(calls) == 1


def test_followup_includes_recent_question(monkeypatch):
    queries = []
    monkeypatch.setattr(services, "search", lambda q: queries.append(q) or [])
    services.respond("Show an example", [{"role": "human", "content": "Explain Alembic"}], "user")
    assert "Alembic" in queries[0]


def test_hybrid_fusion_promotes_shared_result(monkeypatch):
    from app import retrieval
    monkeypatch.setattr(retrieval, "evidence_row", lambda chunk, doc: {"id": chunk.id})
    a, b, c = [SimpleNamespace(id=key) for key in ("a", "b", "c")]
    result = fuse_results([(a, None), (b, None)], [(c, None), (b, None)])
    assert result[0]["id"] == "b"
    assert len(result) == 3
