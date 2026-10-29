"""Test model-worker lifecycle, region filtering and failure recovery.

Controlled worker processes and synthetic PDFs verify reuse, slow-progress warnings,
crash detection and cancellation cleanup. Errors produce uncertain page fallback;
elapsed time alone does not.
"""

import json
import multiprocessing
import time
from pathlib import Path
from types import SimpleNamespace

import pymupdf
import pytest

from app import document_processing, neural_pdf_extraction


def hanging_worker(connection):
    """Simulate an uninterruptible native page call after successful model startup."""

    connection.send({"ready": True})
    connection.recv()
    time.sleep(60)


def echo_worker(connection):
    """Serve repeated requests to prove that the same worker remains reusable."""

    connection.send({"ready": True})
    while True:
        request = connection.recv()
        if request is None:
            break
        connection.send({"page": {"metadata": {"page_number": request[1] + 1}}})
    connection.close()


def make_pdf(path: Path, pages: int = 2) -> None:
    """Create local prose/caption fixtures without model or network dependencies."""

    with pymupdf.open() as pdf:
        for _ in range(pages):
            page = pdf.new_page()
            page.insert_text((72, 72), "Paragraph text supports a fact.")
            page.insert_text((72, 110), "Table 2: Method comparison.")
        pdf.save(path)


def test_worker_is_reused_across_page_and_paper_requests(tmp_path, monkeypatch):
    monkeypatch.setattr(neural_pdf_extraction, "run_region_worker", echo_worker)
    reader = neural_pdf_extraction.NeuralPageReader()
    try:
        reader.read_page(tmp_path / "first.pdf", 0)
        process_id = reader.process.pid
        result = reader.read_page(tmp_path / "second.pdf", 2)
        assert reader.process.pid == process_id
        assert result["metadata"]["page_number"] == 3
    finally:
        reader.close()
    assert reader.process is None


def test_adapter_disables_grid_component_and_rejects_incompatible_versions(monkeypatch):
    from pymupdf.layout import DocumentLayoutAnalyzer

    model = SimpleNamespace(_model=SimpleNamespace(table_grid_extractor=object()))
    monkeypatch.setattr(neural_pdf_extraction, "version", lambda name: "1.28.2")
    monkeypatch.setattr(DocumentLayoutAnalyzer, "get_model", lambda **kwargs: model)
    assert (
        neural_pdf_extraction.create_region_model()._model.table_grid_extractor is None
    )
    monkeypatch.setattr(
        neural_pdf_extraction, "version", lambda name: "another-version"
    )
    with pytest.raises(RuntimeError, match="requires"):
        neural_pdf_extraction.create_region_model()


def test_region_conversion_omits_visual_table_and_equation_regions(tmp_path):
    path = tmp_path / "paper.pdf"
    make_pdf(path, 1)
    with pymupdf.open(path) as pdf:
        page = pdf[0]
        regions = [
            {"class_name": "text", "group_bbox": [60, 55, 420, 90]},
            {"class_name": "caption", "group_bbox": [60, 94, 420, 130]},
        ]
        for kind in ("table", "picture", "formula", "page-header", "page-footer"):
            regions.append({"class_name": kind, "group_bbox": [60, 55, 420, 130]})
        result = neural_pdf_extraction.extract_region_text(page, regions)
    assert [box["class"] for box in result["page_boxes"]] == ["text", "caption"]
    assert result["text"].count("Paragraph text") == 1


@pytest.mark.parametrize("error_type", [RuntimeError, OSError, EOFError])
def test_extraction_error_fallback_is_page_based_and_reported(tmp_path, error_type):
    path = tmp_path / "paper.pdf"
    make_pdf(path)

    class FailingReader:
        """Return an extraction error to exercise fallback and reporting."""

        def read_page(self, *args):
            """Simulate an actual failure without loading the model."""
            raise error_type("test extraction failure")

    kind, sections = document_processing.extract_pdf_sections(
        path, reader=FailingReader()
    )
    assert kind == "pdf_page_fallback"
    assert all(section.structure_confidence == "uncertain" for section in sections)
    assert all(not section.heading_path for section in sections)
    caption = next(section for section in sections if section.reference_label)
    assert caption.location in {"page 1", "page 2"}
    report = json.loads(
        next(document_processing.PDF_EXTRACTION_REPORT_PATH.glob("*.json")).read_text()
    )
    assert report["neural_pages"] == 0 and report["fallback_pages"] == 2
    assert all(error_type.__name__ in page["review_reason"] for page in report["pages"])
    assert "Paragraph text" not in json.dumps(report)


def test_slow_neural_pages_continue_without_fallback(tmp_path, monkeypatch):
    """Even a clock far beyond the former limits must not trigger fallback."""
    path = tmp_path / "paper.pdf"
    make_pdf(path)
    clock = iter(range(0, 10000, 100))
    monkeypatch.setattr(document_processing, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(
        document_processing,
        "extract_fast_page",
        lambda *args: pytest.fail("Slow is not failed"),
    )

    class SlowReader:
        """Provide complete neural pages while the test clock advances by minutes."""

        def read_page(self, path, number):
            """Return text regardless of how much elapsed time has been recorded."""
            return {
                "metadata": {"page_number": number + 1},
                "text": "Neural text",
                "page_boxes": [{"class": "text", "pos": (0, 11)}],
            }

    pages = document_processing.extract_text_pages(path, reader=SlowReader())
    assert len(pages) == 2
    assert all(
        page["metadata"]["extraction_method"] == "neural_regions" for page in pages
    )


def test_slow_worker_warns_and_returns_complete_result(monkeypatch, capsys):
    """Warnings after 30 and 60 seconds must not discard an eventual neural result."""
    reader = neural_pdf_extraction.NeuralPageReader(warning_seconds=30)
    replies = iter([False, False, True])
    reader.connection = SimpleNamespace(
        poll=lambda wait: next(replies), recv=lambda: {"page": "complete"}
    )
    reader.process = SimpleNamespace(is_alive=lambda: True)
    clock = iter(range(0, 1000, 31))
    monkeypatch.setattr(neural_pdf_extraction, "perf_counter", lambda: next(clock))
    assert reader.wait_for_worker_reply("test neural page") == {"page": "complete"}
    output = capsys.readouterr().err
    assert output.count("SLOW") == 2
    assert "continuing neural extraction" in output


def test_worker_crash_is_detected():
    reader = neural_pdf_extraction.NeuralPageReader()
    reader.connection = SimpleNamespace(poll=lambda wait: False)
    reader.process = SimpleNamespace(is_alive=lambda: False)
    with pytest.raises(RuntimeError, match="stopped before returning"):
        reader.wait_for_worker_reply("crashed page")


def test_blank_neural_output_recovers_available_native_text(tmp_path):
    path = tmp_path / "paper.pdf"
    make_pdf(path, pages=1)

    class BlankReader:
        """Simulate a completed model call that omitted readable PDF text."""

        def read_page(self, *args):
            """Return no regions so recovery depends on content, not elapsed time."""
            return {"text": "", "page_boxes": [], "metadata": {"page_number": 1}}

    pages = document_processing.extract_text_pages(path, reader=BlankReader())
    assert "Paragraph text" in pages[0]["text"]
    assert pages[0]["metadata"]["extraction_method"] == "fast_fallback"
    assert "omitted all available page text" in pages[0]["metadata"]["review_reason"]


def test_fallback_page_breaks_neural_hierarchy_inheritance():
    pages = [
        {
            "metadata": {"page_number": 1},
            "text": "2 Methods\n\nOverview",
            "page_boxes": [
                {"class": "section-header", "pos": (0, 9)},
                {"class": "text", "pos": (11, 19)},
            ],
        },
        {
            "metadata": {
                "page_number": 2,
                "extraction_method": "fast_fallback",
                "review_reason": "timeout",
            },
            "text": "Uncertain body",
            "page_boxes": [{"class": "text", "pos": (0, 14)}],
        },
        {
            "metadata": {"page_number": 3},
            "text": "2.1 Training\n\nDetails",
            "page_boxes": [
                {"class": "section-header", "pos": (0, 12)},
                {"class": "text", "pos": (14, 21)},
            ],
        },
    ]
    sections = document_processing.group_layout_pages_into_sections(pages)
    assert sections[1].structure_confidence == "uncertain"
    assert sections[1].heading == "" and sections[1].location == "page 2"
    assert sections[2].heading_path == ("2.1 Training",)


def test_unlabelled_neural_caption_retains_caption_type_and_uncertainty():
    section = document_processing.ExtractedSection(
        location="page 3",
        text="Method overview without a numbered figure label.",
        strategy="pdf-caption",
        heading="Caption",
        extraction_method="fast_fallback",
        structure_confidence="uncertain",
        review_reason="page layout timed out",
    )
    child = document_processing.split_sections_into_chunks([section], "Paper")[0]
    assert child.metadata["evidence_type"] == "caption"
    assert child.metadata["structure_confidence"] == "uncertain"
    assert child.metadata["review_reason"] == "page layout timed out"
    assert child.metadata["reference_label"] == ""


def test_cancellation_cleanup_stops_a_busy_worker(tmp_path, monkeypatch):
    """Explicit shutdown must clean up native work without limiting extraction time."""
    monkeypatch.setattr(neural_pdf_extraction, "run_region_worker", hanging_worker)
    reader = neural_pdf_extraction.NeuralPageReader()
    try:
        reader.start()
        process_id = reader.process.pid
        reader.connection.send((str(tmp_path / "paper.pdf"), 0))
        reader.close()
        assert reader.process is None and reader.connection is None
        assert all(
            child.pid != process_id for child in multiprocessing.active_children()
        )
    finally:
        reader.close()


@pytest.mark.parametrize("interval", [0, -1, float("inf"), float("nan")])
def test_invalid_warning_interval_is_rejected(interval):
    """Prevent invalid intervals from turning progress reporting into a busy loop."""
    with pytest.raises(ValueError, match="warning interval"):
        neural_pdf_extraction.NeuralPageReader(warning_seconds=interval)
