"""Check catalog/guide knowledge without embedding papers or accessing live Drive.

Fictional paper IDs and editable guide rows ensure reading priorities come from
the files, not benchmark-specific answers or hardcoded paper lists.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app import (
    evaluate,
    grounding,
    ingest,
    paper_store,
    question_planning,
    retrieval,
    services,
    study_materials,
)
from app.drive_sync import download_research_source


@pytest.fixture
def study_files(tmp_path):
    """Create a tiny catalog and guide using the production manifest format."""
    catalog = tmp_path / "index.csv"
    catalog.write_text(
        "#,Phase,Title,Year,arXiv link,File\n"
        "9,Foundations,First Model,2020,https://example.org/a,09_First.pdf\n"
        "42,Efficiency,Compact Model,2021,https://example.org/b,42_Compact.pdf\n"
        "43,Efficiency,Other Model,2022,https://example.org/c,43_Other.pdf\n",
        encoding="utf-8",
    )
    guide = tmp_path / "how_to_read_llm_papers.md"
    guide.write_text(
        "# How to Read\n\n## Reading depth\n\n"
        "| Depth | Papers | Why |\n|---|---|---|\n"
        "| **Pass 3 (deep)** | 9 | Core idea |\n"
        "| **Pass 2 (understand)** | 42–43 | Practical concepts |\n\n"
        "## Prerequisites\n\nRead First Model before Compact Model.\n",
        encoding="utf-8",
    )
    manifest = study_materials.SOURCE_MANIFEST_PATH
    rows = [
        {
            "id": "catalog",
            "title": catalog.name,
            "path": catalog.name,
            "source_url": "https://example.org/catalog",
        },
        {
            "id": "guide",
            "title": guide.name,
            "path": guide.name,
            "source_url": "https://example.org/guide",
        },
    ]
    manifest.write_text(json.dumps(rows), encoding="utf-8")
    return catalog, guide, manifest


def test_catalog_depths_are_loaded_from_guide_not_hardcoded(study_files):
    catalog = study_materials.load_paper_catalog()
    assert [row["reading_depth"] for row in catalog] == [
        "Pass 3 (deep)",
        "Pass 2 (understand)",
        "Pass 2 (understand)",
    ]
    assert catalog[1]["phase"] == "Efficiency"
    overview = next(
        row
        for row in study_materials.load_study_passages()
        if row["location"] == "Catalog overview"
    )
    assert "Efficiency: 2 papers" in overview["content"]
    priorities = next(
        row
        for row in study_materials.load_study_passages()
        if row["location"] == "Reading priority: Pass 2 (understand)"
    )
    assert (
        "Compact Model" in priorities["content"]
        and "Other Model" in priorities["content"]
    )
    _, guide, _ = study_files
    guide.write_text(guide.read_text().replace("Pass 2 (understand)", "Pass 1 (skim)"))
    assert study_materials.load_paper_catalog()[1]["reading_depth"] == "Pass 1 (skim)"


def test_references_keep_citations_and_change_ids_when_text_changes(study_files):
    before = study_materials.load_study_passages()
    assert {row["title"] for row in before} == study_materials.STUDY_FILENAMES
    assert all(len(row["content"]) <= 1800 and "parent_id" not in row for row in before)
    depth = next(row for row in before if row["location"] == "Reading depth, paper #42")
    assert "Compact Model" in depth["content"] and "Pass 2" in depth["content"]
    assert depth["source_url"] == "https://example.org/guide"
    _, guide, _ = study_files
    guide.write_text(
        guide.read_text().replace("Practical concepts", "Revised priority")
    )
    after = next(
        row
        for row in study_materials.load_study_passages()
        if row["location"] == depth["location"]
    )
    assert after["id"] != depth["id"]


def test_references_work_without_any_pdf_vectors_and_follow_shared_reranking(
    study_files, monkeypatch
):
    monkeypatch.setattr(retrieval, "find_vector_candidates", lambda *args, **kwargs: [])
    seen = []

    def score(query, candidates):
        """Observe real candidate construction without loading a neural model."""
        seen.extend(candidates)
        return [{**row, "rerank_score": 1.0} for row in candidates]

    monkeypatch.setattr(retrieval, "rerank_passage_candidates", score)
    evidence = retrieval.retrieve_relevant_passages("reading depth Compact Model")
    assert seen and evidence
    assert any(row["title"] == "how_to_read_llm_papers.md" for row in evidence)
    assert retrieval.load_scoped_chunks("42_Compact.pdf") == []


def test_catalog_enriches_only_actual_indexed_papers(study_files, monkeypatch):
    saved = {
        "metadatas": [
            {
                "document_id": "pdf-42",
                "title": "42_Compact.pdf",
                "source_url": "https://example.org/pdf",
                "indexed_at": "today",
            }
        ]
    }
    monkeypatch.setattr(
        paper_store,
        "open_chunk_store",
        lambda: SimpleNamespace(get=lambda **kwargs: saved),
    )
    papers = paper_store.list_indexed_papers()
    assert len(papers) == 1
    assert papers[0]["paper_number"] == 42 and papers[0]["phase"] == "Efficiency"
    assert papers[0]["reading_depth"] == "Pass 2 (understand)"
    assert papers[0]["title"] == "42_Compact.pdf"


def test_planner_receives_catalog_identity_and_reference_search_instructions(
    study_files, monkeypatch
):
    def plan(schema, instruction, payload):
        """Keep learning searches unscoped so guide evidence remains eligible."""
        assert "reading depth" in instruction and "paper=null" in instruction
        assert payload["indexed_papers"][0]["paper_number"] == 42
        return question_planning.QuestionPlan(
            tasks=[question_planning.SearchTask(question="Reading depth Compact Model")]
        )

    monkeypatch.setattr(question_planning, "request_structured_response", plan)
    paper = {**study_materials.load_paper_catalog()[1], "title": "42_Compact.pdf"}
    result = question_planning.plan_question_tasks(
        "How deeply should I read paper 42?", [], [paper]
    )
    assert result.tasks[0].paper is None


def test_supporting_source_changes_invalidate_evaluation_resume(study_files):
    before = evaluate.build_evaluation_configuration()
    _, guide, _ = study_files
    guide.write_text(guide.read_text() + "\n## Extra advice\nReview assumptions.\n")
    after = evaluate.build_evaluation_configuration()
    assert before["study_materials_sha256"] != after["study_materials_sha256"]
    assert before["index_sha256"] == after["index_sha256"]


def test_duplicate_catalog_numbers_are_rejected(study_files):
    catalog, _, _ = study_files
    catalog.write_text(catalog.read_text().replace("43,Efficiency", "42,Efficiency"))
    with pytest.raises(ValueError, match="unique"):
        study_materials.load_paper_catalog()


def test_missing_or_deleted_study_sources_are_not_silently_reused(study_files):
    _, guide, manifest = study_files
    guide.unlink()
    with pytest.raises(ValueError, match="missing"):
        study_materials.load_study_passages()
    manifest.unlink()
    assert study_materials.load_study_passages() == []


@pytest.mark.parametrize(
    "name,mime,export",
    [
        ("index.csv", "text/csv", None),
        ("how_to_read_llm_papers.md", "text/markdown", None),
        (
            "how_to_read_llm_papers",
            "application/vnd.google-apps.document",
            "text/plain",
        ),
    ],
)
def test_sync_downloads_the_two_supporting_sources(tmp_path, name, mime, export):
    class DriveFile(dict):
        """Capture downloads locally without authenticating to Google Drive."""

        def GetContentFile(self, path, **kwargs):
            self["export"] = kwargs.get("mimetype")
            Path(path).write_text("Reference content", encoding="utf-8")

    item = DriveFile(id="source", title=name, mimeType=mime)
    result = download_research_source(item, tmp_path)
    assert result.is_file() and item["export"] == export


def test_pdf_ingestion_ignores_reference_rows_without_changing_them(study_files):
    _, _, manifest = study_files
    pdf = manifest.parent / "paper.pdf"
    pdf.write_bytes(b"pdf fixture")
    rows = json.loads(manifest.read_text())
    rows.append(
        {
            "id": "paper",
            "title": "Paper",
            "path": pdf.name,
            "source_url": "https://example.org/pdf",
            "source_type": "google_drive",
        }
    )
    manifest.write_text(json.dumps(rows))
    assert [row["id"] for row in ingest.load_source_manifest(manifest)] == ["paper"]


def test_guide_answer_passes_normal_quote_verification_and_cites_guide(
    study_files, monkeypatch
):
    """Guide evidence follows the same claim/source checks as research passages."""
    passage = next(
        row
        for row in study_materials.load_study_passages()
        if row["location"] == "Reading depth, paper #42"
    )
    monkeypatch.setattr(
        services,
        "plan_question_tasks",
        lambda *args: question_planning.QuestionPlan(
            tasks=[question_planning.SearchTask(question="Reading depth Compact Model")]
        ),
    )
    monkeypatch.setattr(
        services, "retrieve_relevant_passages", lambda *args, **kwargs: [passage]
    )

    def draft(schema, instruction, evidence, request):
        """Generate a known supported claim without making an Ollama request."""
        assert "learning recommendations" in instruction
        assert evidence[0]["evidence_type"] == "study_guide"
        return grounding.DraftAnswer(
            status="answered",
            claims=[
                grounding.SupportedClaim(
                    text="The guide recommends Pass 2 for Compact Model.",
                    quote="The reading guide recommends: Pass 2 (understand).",
                    source_id=evidence[0]["id"],
                    task_ids=[0],
                )
            ],
        )

    monkeypatch.setattr(grounding, "generate_from_evidence", draft)
    result = services.answer_research_question(
        "How deeply should I read Compact Model?"
    )
    assert result["status"] == "answered"
    assert result["sources"][0]["source_url"] == "https://example.org/guide"
    assert result["sources"][0]["title"] == "how_to_read_llm_papers.md"
