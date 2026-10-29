"""Run the PDF layout model in a separate process for document_processing.py.

The model finds text, headings and captions and suggests their reading order. One
worker serves many pages so the model is not loaded for every page. This file
manages that worker; section grouping and chunking happen in document_processing.py.
"""

import math
import multiprocessing
import sys
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from .config import PDF_LAYOUT_WARN_SECONDS

LAYOUT_VERSION = "1.28.2"


def create_region_model():
    """Load the model that locates paragraphs, headings and captions on PDF pages.

    Keep region detection and reading order, but disable the extra step that
    rebuilds table rows and cells because this agent indexes prose and captions. The
    library exposes this setting through an internal attribute, so check its
    version and expected attributes before changing the setting.
    """

    if version("pymupdf_layout") != LAYOUT_VERSION:
        raise RuntimeError(
            f"Region-only extraction requires pymupdf_layout {LAYOUT_VERSION}"
        )
    import pymupdf4llm
    from pymupdf.layout import DocumentLayoutAnalyzer

    # Free the automatically loaded model before creating our configured instance.
    pymupdf4llm.use_layout(False)
    model = DocumentLayoutAnalyzer.get_model(n_workers=1, use_sort=True)
    if not hasattr(model, "_model") or not hasattr(
        model._model, "table_grid_extractor"
    ):
        raise RuntimeError("The pinned layout adapter no longer matches this library")
    # Keep table-region detection, but skip rebuilding rows, columns and cells.
    model._model.table_grid_extractor = None
    return model


def extract_region_text(page, regions: list[dict]) -> dict:
    """Turn detected page regions into text blocks with positions and a page number.

    Read prose, headings, captions, lists, code and footnotes from the PDF text
    layer in the model's suggested order. Skip detected table, image, formula,
    header and footer regions. Record heading font sizes so document_processing.py
    can suggest heading depth.
    """

    import pymupdf

    flags = pymupdf.TEXTFLAGS_TEXT & ~pymupdf.TEXT_USE_CID_FOR_UNKNOWN_UNICODE
    textpage = page.get_textpage(flags=flags)
    spans = [
        span
        for block in page.get_text("dict", textpage=textpage)["blocks"]
        for line in block.get("lines", [])
        for span in line["spans"]
    ]
    text, boxes = "", []
    allowed = {
        "text",
        "title",
        "section-header",
        "caption",
        "list-item",
        "footnote",
        "code",
    }
    for region in regions:
        kind = region["class_name"]
        if kind not in allowed:
            continue
        rect = pymupdf.Rect(region["group_bbox"])
        content = page.get_textbox(rect, textpage=textpage).strip()
        if not content:
            continue
        start = len(text)
        text += content + "\n\n"
        box = {"class": kind, "pos": (start, len(text)), "bbox": tuple(rect)}
        if kind == "section-header":
            box["font_size"] = max(
                (
                    round(span["size"], 1)
                    for span in spans
                    if rect.contains(
                        pymupdf.Point(
                            (span["bbox"][0] + span["bbox"][2]) / 2,
                            (span["bbox"][1] + span["bbox"][3]) / 2,
                        )
                    )
                ),
                default=0,
            )
        boxes.append(box)
    return {
        "metadata": {"page_number": page.number + 1},
        "text": text,
        "page_boxes": boxes,
    }


def run_region_worker(connection) -> None:
    """Load one model and serve page requests from the indexing process.

    Each request contains a PDF path and a zero-based page number. Reply with
    extracted text and region metadata, or an error the caller can recover from.
    Reuse the model and open PDF between requests to avoid repeated setup.
    """

    pdf = None
    try:
        import pymupdf

        model = create_region_model()
        connection.send({"ready": True})
        while True:
            request = connection.recv()
            if request is None:
                break
            path, number = request
            try:
                if pdf is None or pdf.name != path:
                    if pdf is not None:
                        pdf.close()
                    pdf = pymupdf.open(path)
                page = pdf[number]
                regions = model.predict(page, return_raw=True)
                connection.send({"page": extract_region_text(page, regions)})
            except Exception as error:  # noqa: BLE001 - forward library errors across the process boundary
                # Send failures to the parent so they are recorded as explicit fallbacks.
                connection.send({"error": f"{type(error).__name__}: {error}"})
    except (EOFError, BrokenPipeError):
        pass  # The owning ingestion process has closed its pipe.
    except Exception as error:  # noqa: BLE001 - forward library errors across the process boundary
        try:
            connection.send({"error": f"{type(error).__name__}: {error}"})
        except (EOFError, OSError):
            pass
    finally:
        if pdf is not None:
            pdf.close()
        connection.close()


class NeuralPageReader:
    """Manage the separate process that reads neural PDF regions.

    The process keeps a model crash separate from the indexing command and lets
    cancellation stop native library work. Page requests are sequential; slow work
    reports progress and keeps running.
    """

    def __init__(self, warning_seconds: float = PDF_LAYOUT_WARN_SECONDS):
        """Prepare worker state and validate the progress-warning interval.

        Do not load the model yet: ingestion can skip unchanged papers without
        starting a worker.
        """

        if not math.isfinite(warning_seconds) or warning_seconds <= 0:
            raise ValueError(
                "PDF layout warning interval must be positive finite seconds"
            )
        self.warning_seconds = warning_seconds
        self.process = None
        self.connection = None

    def wait_for_worker_reply(self, label: str) -> dict:
        """Wait for a page or startup reply while checking that the worker is alive.

        Print elapsed waiting time at the configured interval so a slow model is
        visible. A warning does not cancel extraction or switch to fallback.
        """

        started = perf_counter()
        next_warning = started + self.warning_seconds
        while True:
            now = perf_counter()
            wait = min(1.0, max(0.0, next_warning - now))
            if self.connection.poll(wait):
                return self.connection.recv()
            if not self.process.is_alive():
                raise RuntimeError("layout worker stopped before returning a result")
            now = perf_counter()
            if now >= next_warning:
                print(
                    f"SLOW {label}: still waiting after {now - started:.1f}s; "
                    "continuing neural extraction",
                    file=sys.stderr,
                    flush=True,
                )
                next_warning = now + self.warning_seconds

    def start(self) -> None:
        """Start the model worker if it is not already running, then wait for readiness.

        The ready message ensures page requests are sent only after model setup
        succeeds. Close failed worker state before passing an error back to
        extraction.
        """

        if self.process is not None and self.process.is_alive():
            return
        self.close()
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(
            target=run_region_worker, args=(child,), daemon=True
        )
        try:
            self.process.start()
        except OSError:
            self.close()
            raise
        finally:
            child.close()
        try:
            reply = self.wait_for_worker_reply("layout worker startup")
            if not reply.get("ready"):
                raise RuntimeError(reply.get("error", "layout worker did not start"))
        except (EOFError, OSError, RuntimeError):
            self.close()
            raise

    def read_page(self, path: Path, number: int) -> dict:
        """Request one PDF page and return its complete neural extraction result.

        Start or reuse the worker, send the file path and page number, and wait for
        its reply. Report worker errors to document_processing.py so that file can
        decide how to recover.
        """

        self.start()
        try:
            self.connection.send((str(path.resolve()), number))
            reply = self.wait_for_worker_reply(
                f"{path.name}, page {number + 1}: page layout"
            )
            if "error" in reply:
                raise RuntimeError(reply["error"])
            return reply["page"]
        except (EOFError, OSError, RuntimeError):
            self.close()
            raise

    def close(self) -> None:
        """Release worker resources when indexing finishes, fails or is cancelled.

        Ask the worker to exit first. If native model code is still busy, terminate
        it so cancellation does not leave a background process. The short shutdown
        waits do not limit extraction time.
        """

        if self.process is not None:
            if self.process.is_alive():
                try:
                    self.connection.send(None)
                except (EOFError, OSError):
                    pass
                # These brief waits apply only to shutdown, never to page processing.
                self.process.join(timeout=0.2)
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join(timeout=1)
                if self.process.is_alive():
                    self.process.kill()
                    self.process.join(timeout=1)
            self.process.close()
            self.process = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None
