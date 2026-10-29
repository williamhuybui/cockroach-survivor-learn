# Validation â€” research RAG update, 2026-10-20

## Current checks

- 53 automated tests passed using `.venv-research`.
- Ruff formatting and lint checks passed for `app/` and `tests/`.
- The research index was rebuilt with pipeline
  `embeddinggemma:768:paper-parents-v4:1200:120`: 39 PDFs, 5,295 child chunks,
  zero skipped files. Parent sections are stored beside the child-vector index.
- Tests cover PDF classification and section chunking, unchanged/changed
  ingestion, Chroma parent expansion, BM25 ranking, named-paper handling,
  evidence quotation/number checks, prompt-injection boundaries, chat deletion,
  quizzes, the team endpoint, and generic evaluation reports.
- Two consecutive real local answers completed after replacing a large Chroma
  `$in` parent filter with a bounded dense pool plus Python-side filtering.

## Generic evaluation run

`data/evaluations/hybrid-grounded-final.json` contains the 15-case manager
dataset run against the rebuilt index. Its labels never enter the agent call.

| Measure | Result | Meaning |
| --- | ---: | --- |
| Cases | 15 | Six single-paper, six multi-paper, three unanswerable labels |
| Runtime errors | 0 | The full run completed after the Chroma query repair |
| Expected-source recall | 83.3% | 10/12 answerable cases had their expected paper represented in returned evidence |
| Unanswerable abstention recall | 66.7% | Two of three labelled unanswerable cases abstained |
| Abstention precision | 28.6% | Safety is conservative: several answerable questions also abstained |
| Human-reviewed accuracy | Not measured | No reviewer labels have been supplied |
| Human-reviewed unsupported-claim rate | Not measured | No reviewer claim labels have been supplied |

These are retrieval and abstention measurements, not a claim of model accuracy.
The final report deliberately leaves correctness and hallucination metrics empty
until a person reviews the answer and its quoted evidence. One observed example
is the Transformer paper: retrieved passages contain 41.0 in prose and 41.8 in
a table, so evaluation must inspect the original PDF rather than silently choose
a value.

## Learner benchmark checkpoint

The supplied `learner_rag_50_question_benchmark.csv` is treated as evaluation
data only. The evaluator reads its `paper_scope` and `expected_behavior` fields
only after an answer has been generated; `expected_answer`, source location, and
route never enter retrieval or prompting.

The local run checkpointed 37 of 50 questions with no application-level result
errors. After that, the Windows native local-model runtime terminated with a
memory allocation/access violation outside Python. This is not recorded as a
model failure or used to tune the RAG pipeline. The evaluator now supports
`--resume` and `--max-cases` so an interrupted evaluation can continue in small,
separate batches after the local runtime is stable.

## Remaining limits

- Exact quotation and numeric checks reduce unsupported claims but do not prove
  every paraphrase is entailed or resolve contradictions within a paper.
- The BM25 implementation is intentionally local and simple. Revisit it only if
  the corpus grows enough that scanning its chunk text is slow.
- The reranker cutoff is an uncalibrated raw score of zero. Tune it after human
  review, not against the supplied reference answers at runtime.
- PDF extraction is text-based. Scanned pages, figures, equations, tables and
  two-column reading order require manual review or a future OCR/table path.
- Local Gemma generation took roughly 6â€“25 seconds per case after model load.
  The application remains synchronous by design.
# Conditional workflow update checked on 2026-10-23

- Graph now uses plan, retrieve, generate, verify and format_response nodes.
  Conditional edges handle clarification and empty evidence without generation.
- Offline suite: **74 passed**. Ruff lint and formatting passed; diff whitespace
  checks passed. Tests use temporary storage and fake model outputs.
- A real local Chroma test verifies exact-title filtering before dense top-k:
  a paper excluded from global top one is still found by its scoped search.
- Tests cover missing-paper clarification, separate comparison subquestions,
  partial and complete task coverage, cross-paper claim rejection, shared context
  limits, deduplication provenance, private evaluation labels, and safe resume.
- Live Gemma planner smoke testing was attempted but blocked by the installed
  Ollama runtime: `llama-server.exe` was not found and generation returned HTTP
  500. No new live benchmark scores or answer-quality improvements are claimed.
- Existing evaluation reports describe their original workflow. Start a new
  report for this version; resume checks reject incompatible code/model/index
  fingerprints. Parent/child storage and existing indexed papers need no rebuild
  for this graph-only upgrade.
# Evidence evaluation instrumentation checked on 2026-10-23

- **82 offline tests passed**; lint, formatting and diff whitespace checks passed.
- Added reviewed-requirement candidate recall, final context coverage and all-part
  coverage. Exact paper titles alone do not satisfy a reviewed evidence span.
- Candidate, ordered accepted reranker and parent-expanded passages are retained
  for offline inspection, along with stable final chunk IDs and task associations.
  Full passage audits are disabled in browser responses.
- Timing reports include P50/P95 and sample counts by actual route and explicitly
  declared runtime condition. Failed requests retain total duration; unavailable
  stage timings stay absent. No new ranking or chunking settings were introduced.
- Tests cover private-label isolation, alternative evidence spans, stage losses,
  missing labels, latency calculations and changed-label resume rejection.
- This is instrumentation validation, not a new model benchmark. The previously
  recorded local Ollama missing-binary issue still blocks live quality evaluation.
  Human relevance and correctness reviews are needed before reporting accuracy.


## Plan approval update checked on 2026-10-23

- 88 offline tests passed; lint, formatting and JavaScript syntax checks passed.
- Headless browser smoke passed proposal display, refresh, revision, stale controls and approval with mocked responses.
- SQLite pending_plans was initialized without altering existing chat messages. Approval executes the saved plan without replanning; revisions replace the approval token.
- Removed the fixed three-task planner cap. Execution remains context-batched; unfinished tasks are listed and saved for another approval.
- Offline evaluation explicitly records automatic_offline and does not simulate the human approval interaction. Live model planning remains unvalidated because of the previously recorded Ollama runtime issue.
# Readability simplification on 24 October 2026

Reviewed all application Python modules and the browser script. Removed five
single-use wrappers; kept heading repair inside line cleanup, and kept manifest
hashing, Drive query escaping, evaluation paper-label parsing and review citation
formatting at their call sites. Consolidated daily challenge decoding and simplified
claim verification, timing collection, collection reset and approval-mode selection.
Transaction handling, model caching, validation and the LangGraph nodes remain.

Validation: all 89 offline tests passed on the completed verbose run. Ruff lint and
formatting checks passed. An initial quiet test run stalled and was stopped before
the successful rerun. These checks do not measure live model accuracy.

# Function naming review on 24 October 2026

Renamed application functions to describe their actions and data, including PDF
heading repair, Chroma stores, retrieval, model calls, graph nodes, route handlers,
evaluation and browser actions. Updated callers, imports, test mocks and the junior
developer guide. Standard Python CLI entry points retain main(). API URLs, JSON
keys, stored record formats and graph node labels remain unchanged; all string
constants were compared with the pre-rename versions across 19 Python modules.

Validation: 89 offline tests passed outside the sandbox after Windows socket
initialization blocked the sandboxed FastAPI test client. Ruff lint and formatting
passed. Node syntax checking passed. A headless Edge check with mocked API replies
passed chat proposal, refresh, revision, stale controls, approval, quiz creation and
grading. Live model answer quality was not evaluated.


## PyMuPDF4LLM extraction update on 25 October 2026

Replaced custom heading regex, heading-name lists, paper-signal classification and
per-page font checks with PyMuPDF4LLM 1.28.2 layout extraction. The matching PyMuPDF
and layout packages are installed in .venv-research. Library block positions are
used to group section text without reinterpreting Markdown code/list/table lines
as headings. Page spans, source links, parent-child IDs and bounded LangChain
splitting remain. Parents are flat sections, not a nested section tree.

Extraction-only checks ran on three synced PDFs: Chain-of-Thought Prompting,
Retrieval-Augmented Generation, and Neural Machine Translation by Jointly Learning
to Align and Translate. All produced section parents and bounded child chunks.
On the 43-page Chain-of-Thought PDF, References now spans pages 10-14 instead of
10-43; Checklist and lettered appendix headings are separated. It produced 61
parents and 202 children, with a maximum child length of 1197 characters.
The other two papers produced 35/97 and 35/82 parents/children respectively.
Local review outputs are in data/extraction_review/ (not benchmark fixtures).

These are extraction smoke checks, not a measured improvement in retrieval or
answer accuracy. The layout model can still label author names or example labels
as headings; tables/figures/formulas and heading quality need corpus review. OCR
remains disabled. No production embeddings or Chroma records were rebuilt.
The new index signature requires an explicit ingestion --rebuild before querying
the old index with this code. The guide documents the installation/rebuild steps.

Validation: all 89 tests passed on the final complete run (10.35 seconds),
including extraction-failure preservation. Ruff lint/formatting checks and pip
check passed. Tests exercise cross-page sections, unfamiliar
headings, page fallback, code/list/table boundaries, empty pages, extraction
errors, child limits, and source metadata through real Chroma ingestion.


## Text-only extraction and nested sections on 25 October 2026

The current extractor excludes blocks labeled picture, formula, table, caption,
page-header or page-footer in both section and page fallback paths. Text inside
ordinary paragraphs is retained, including inline notation; this is block-based
filtering, not perfect semantic removal of every mathematical expression.

Sections now retain heading paths, depth and containing-section indexes. Explicit
numbering (2, 2.1, A.1) refines library font levels only on detected heading blocks.
Unnumbered headings use library levels. A missing numbered ancestor is not invented
or replaced with a contradictory section number. Each section keeps its own text;
subsection bodies are not duplicated into all ancestors. Chroma parents and children
store section_path, heading_level and ancestor_section_id. Existing parent_id still
identifies a child's immediate section. Full paths appear in chunk prefixes and
citation locations; retrieval does not fetch every ancestor automatically.

Repeated continuation headings are suppressed across pages. Text cleanup removes
invisible characters and display-only soft hyphens and expands common ligatures.
Ordinary hyphens, technical punctuation and numbers are preserved. Blank/scanned
pages and replacement characters are reported. Missing layout blocks cause an
extraction error so unfiltered page text does not enter the index.

Validation: all 97 tests passed after the final readability changes (69.52 seconds).
Ruff lint, formatting and git diff whitespace checks passed. Tests cover
numbered and unnumbered hierarchy, sibling branches, missing parents, continuation
headings, excluded blocks on both fallback paths, text cleanup, code/list boundaries,
chunk budgets and hierarchy metadata stored in actual Chroma collections. An
extraction-only run on the 43-page Chain-of-Thought paper retained 61 sections and
produced 177 bounded children (maximum 1197 characters). References still ends on
page 14 and the appendices are separated. Output: data/extraction_review/text_hierarchy_trace.json.

The library can still mistake authors or example labels for headings; unnumbered
hierarchy remains inferred from styling. Ambiguous hard-hyphen repairs and exact
child-page citations are not implemented. No claim of perfect heading detection,
complete formula removal or improved answer accuracy is made. OCR remains off.
Index version is now v7-text-hierarchy-1.28.2; production Chroma was not rebuilt.


## Fast text/caption extraction on 25 October 2026

Replaced the neural layout route with PyMuPDF4LLM's font/column route via
use_layout(False). Explicitly disabled image/graphics processing, background
rendering and table reconstruction (table_strategy=None). Font sizes, bookmarks
and a short fully-bold-line fallback supply heading candidates. Heading lines
are separated from following body text; numeric page footers do not create
sections. Existing numbering-based hierarchy and text cleanup remain.

Explicit Table/Figure/Fig. caption paragraphs become separate searchable records,
with reference_label, evidence_type=caption, the paper URL and an exact PDF page.
Captions are removed from multi-page section bodies to avoid duplicate storage
and imprecise caption pages. Ordinary prose mentions are not treated as caption
labels, and code fences protect examples. Captions and prose share the normal
hybrid retrieval/reranking pipeline. The model receives caption/prose metadata
and instructions not to infer unread table cells or graph values.

Per-page extraction times and per-paper extraction/embedding/storage/total times
are printed with flushing. Embedding progress includes batch numbers. A separate
EXTRACTION_PROFILE is included in each paper fingerprint and Chroma metadata.
The current v7 collection storage signature remains compatible, allowing normal
ingestion without deleting collections. Old-profile papers refresh once; complete
unchanged papers using this profile skip extraction and embedding on resumes.
Incompatible embedding settings and older collection signatures remain rejected.
No production records were reset and no live ingestion process was interrupted.

Validation: 105 tests passed in the complete run (7.59 seconds). Ruff lint and
format checks and git diff whitespace checks passed. A real PDF test replaces
get_layout, find_tables, get_pixmap and get_drawings with failures, proving that
the fast path does not call them. Tests cover exact caption pages, Chroma metadata,
heading/body boundaries, code fences, ordinary prose mentions, displayed progress,
profile changes and unchanged-paper skips. Existing grounding and index-preservation
tests also passed.

Extraction-only smoke checks (including section grouping and child splitting):
LoRA, 26 pages: 5.58 seconds, 26 caption records, 151 children.
Chain-of-Thought, 43 pages: 1.60 seconds, 41 caption records, 237 children.
Retrieval-Augmented Generation: 0.80 seconds, 11 caption records, 120 children.
All children stayed within 1200 characters. These timings exclude embedding,
Chroma writes and process startup. Report: data/extraction_review/fast_text_captions.json.
No new answer-accuracy or heading-recall benchmark was performed.

Limits: fast mode cannot semantically remove every raw table/diagram fragment
from a PDF's text layer. No cell reconstruction or visual interpretation is
performed. Caption boundaries and font-based headings can be imperfect, native
text may contain replacement characters, and library Markdown can rewrap or
join hyphenated words. Review extraction before making answer-quality claims.

## 2026-10-25: bounded neural region extraction

This update supersedes the fast-only extractor described above. Neural heading,
region and reading-order detection are retained. Native text is read directly
from accepted regions, avoiding the full layout-to-Markdown conversion. Table
grid reconstruction is disabled; table, picture, formula, header and footer
regions are excluded. Captions keep exact PDF pages, with explicit Table/Figure
labels when present. PyMuPDF/PyMuPDF4LLM/layout stay pinned to 1.28.2. The isolated
adapter uses an internal table-grid attribute because this release has no public
regions-only switch; its version and attribute shape are checked before use.

One synchronous worker process reuses its model across papers. The defaults are
10 seconds per neural page request and 60 seconds per paper, including startup.
A timeout terminates native work, then reads that page using the fast extractor
with heading guesses disabled. Fallbacks are page-based and marked uncertain;
they break section ancestry. Coverage reports record every fallback reason.
Chroma parents and children retain these flags, and the grounded prompt limits
uncertain text to explicit quoted facts, rather than inferred section relations.

An initial 30-second paper budget fell back on 17 of 43 Chain-of-Thought pages.
The final 60-second default completed all 43 pages. Final extraction-only results
include process startup where needed, section grouping and child splitting:

| Paper | Pages | Neural / fallback | Seconds | Children |
| --- | --- | --- | --- | --- |
| Retrieval-Augmented Generation | 19 | 19 / 0 | 21.92 | 94 |
| LoRA | 26 | 25 / 1 | 39.17 | 119 |
| Chain-of-Thought | 43 | 43 / 0 | 44.03 | 207 |

LoRA page 26 exceeded the 10-second page limit; the worker was stopped and its
fallback was reported. RAG retained Methods > Models/Training nesting. LoRA
retained Our Method > Low-Rank-Parametrized Update Matrices, without promoting
the earlier false '175 billion trainable parameters' prose heading. These are
spot checks, not a heading-recall or answer-accuracy benchmark. RAG page 4 still
reports unreadable glyphs; warnings do not trigger expensive retries or OCR.
Local smoke output: data/extraction_review/bounded_neural_smoke.json. Per-paper
page reports: data/extraction_reports. Reports contain metadata, not passage text.

Validation: 113 tests passed in 13.50 seconds. Ruff lint/format and Git whitespace
checks passed. Tests cover terminating a stalled worker, worker reuse, disabling
table grids, excluding non-text regions, document deadline exhaustion, fallback
metadata, broken ancestry, unlabelled captions, exact caption pages and Chroma
flag preservation. Existing retrieval, grounding, API, history and plan tests pass.

No production ingestion, collection reset or live-process interruption was done.
The v7 storage signature remains compatible; the changed extraction fingerprint
refreshes old-profile papers once on normal ingestion. Completed unchanged papers,
including reported fallbacks, skip on resume. Time limits stop neural work only:
fallback extraction, cleanup, embedding and database writes can add time. Native
fallback text can contain unstructured table/figure fragments; neither extraction
mode interprets their values, and model region labels can still be wrong.

## 2026-10-25: flat sections with heading breadcrumbs

Removed unused section-to-section links (`parent_section_index` and
`ancestor_section_id`). Detected main headings and subsections still create
separate flat section records. A small heading stack holds only names and levels
to build breadcrumbs; it no longer tracks section indexes. Each section retains
its own text and page span. Missing numbered containing headings and uncertain
fallback pages still prevent misleading breadcrumb inheritance.

Child `parent_id` links remain: retrieval expands a matching child into its own
containing section. Tests now check nested and sibling breadcrumbs, separate
section text, absence of ancestor IDs in new Chroma records, and actual expansion
of a Training child into Training text. The developer guide and trace example
describe the flat representation. Existing unused ancestor metadata is ignored;
this cleanup does not change extracted text or vector dimensions.

Validation: focused document/ingestion/worker tests passed (38 tests), followed
by the complete suite (113 tests in 11.91 seconds). The sandbox FastAPI test
client stalled, so that run was stopped and the full suite passed outside the
sandbox. Ruff lint/format and Git whitespace checks passed. No production index
was rebuilt and no runtime performance or answer-accuracy improvement is claimed.

## 2026-10-25: consolidate the document-processing flow

Moved page extraction, fallback, caption recognition and coverage reporting into
document_processing.py. Its entry point is followed by the extraction helpers,
section grouping, heading/text cleanup and chunk creation. neural_pdf_extraction.py
remains responsible for the model worker, region-to-text conversion and hard
process timeouts. The redundant pdf_text_extraction.py module was removed rather
than retained as an import wrapper.

Updated ingestion imports, evaluation source fingerprinting, isolated test paths,
test imports and the developer guide. Extraction function bodies, caption rules,
time budgets, section text, chunking and the extraction profile are unchanged.
Existing index records therefore need no refresh solely for this file move.
Evaluation source hashes intentionally change, preventing an old evaluation run
from silently resuming against reorganized sources.

Validation: all 113 tests passed in 11.91 seconds outside the sandbox, including
real fixture PDF extraction, hard worker timeouts, Chroma metadata/expansion,
grounding and evaluation tests. Ruff lint and format checks passed. Git whitespace
checks passed. No production ingestion or index reset was performed.

## 2026-10-25: quality-first extraction with opt-in hard timeouts

The earlier 10-second page, 60-second paper and hidden 15-second model-startup
cutoffs were application choices, not PyMuPDF requirements or validated quality
thresholds. They are superseded by this policy: neural extraction waits for a
complete result by default. Slow work prints periodic elapsed-time warnings and
continues. Errors, worker crashes and blank model output despite available PDF
text still use explicit, structurally uncertain page fallback. Missing headings
and scanned pages without a text layer do not imply a timeout or recoverable text.

PDF_LAYOUT_PAGE_SECONDS and PDF_LAYOUT_DOCUMENT_SECONDS now default to 0, meaning
disabled. Positive finite values opt into hard timeouts and fallback; invalid
negative/non-finite values are rejected. There is no independent startup deadline.
PDF_LAYOUT_WARN_SECONDS defaults to 30 and controls warnings only. The synchronous
pipe wait checks for replies/crashes at most one second apart; a slow warning
does not discard a model result. Explicit hard timeouts still terminate the native
worker. Existing positive environment overrides remain an intentional opt-in;
the effective local settings were checked as page=0, paper=0, warning=30 seconds.

The v2 extraction profile refreshes papers built under the earlier policy once
on ingestion. No collection reset or production ingestion was performed. The
developer guide and example environment settings describe the new defaults.

Validation: 122 tests passed in 16.29 seconds. New tests cover simulated elapsed
times beyond the former limits without fallback, periodic warnings followed by
a complete result, worker-crash detection without timeouts, blank-output recovery,
invalid timeout settings and actual worker termination for explicitly configured
page/document limits. Ruff lint/format and Git whitespace checks passed.

Real extraction-only check: LoRA page 26, which previously exceeded its page
cutoff, completed neural extraction in 15.46 seconds including model startup.
It returned 1012 text characters in 3 accepted regions without fallback. A
3-second warning interval was used for this smoke check; warnings at 3/6/9/12
seconds did not stop extraction. Output: data/extraction_review/uncapped_neural_smoke.json.
This demonstrates a valid result can arrive after the old cutoff; it is not a
collection-wide quality benchmark. With no optional timeout configured, a hung
but living worker keeps waiting and warning until the operator cancels the run.

## 2026-10-25: remove extraction hard-timeout code entirely

Removed optional page/document timeout settings, deadline parameters and checks,
timeout-triggered fallback, and their configuration-specific tests. The worker now
waits for a complete result or actual failure, with periodic progress warnings.
Fallback remains for extraction errors, crashed workers and unusable neural output.
Model startup follows the same wait-and-warn policy. The v3 extraction profile
records this simplified policy; the storage/vector signature remains compatible.

Explicit cancellation/completion still closes the owning worker. Brief join waits
in close() allow graceful shutdown before termination and do not impose any page
processing limit. Old timeout environment variables are no longer read. The only
extraction timing setting is the positive finite progress-warning interval.

Updated README, developer guide, module comments and example settings. Validation:
122 tests passed in 11.68 seconds outside the sandbox. Tests cover slow pages and
warnings without fallback, real failure recovery, crash detection, cleanup of a
busy worker, invalid warning intervals and the existing retrieval/API pipeline.
Ruff lint/format and Git whitespace checks passed. No production index was rebuilt.

## 2026-10-25: explanatory source comments for reviewers and junior developers

Rewrote application file overviews and function documentation to describe their
role, important inputs/results and purpose in the complete flow. Explained neural
region extraction versus table reconstruction, page fallback, section breadcrumbs,
child chunks, hybrid retrieval, claim verification, graph branches, browser plan
approval and private offline evaluation. Removed stale timeout references and
corrected the model-request retry description to match its actual implementation.

Added class context using ordinary comments for Pydantic models, preserving their
existing model-facing schema descriptions. Updated example/test file overviews,
browser comments, SQL table explanations, dependency comments and the guide's
reading instructions. This was documentation work, not an extraction-policy update.

Verification: parsed all 38 Python source/example/test files before and after,
removed docstrings from their syntax trees and confirmed every executable tree
was identical. Model instructions and settings constants are included in that
comparison. JavaScript, HTML, CSS and SQL content was also compared with comment
lines removed. Dependency declarations were retained. All 122 tests passed in
14.24 seconds using temporary test storage; Ruff lint/format, JavaScript syntax
and Git whitespace checks passed. The running production rebuild was not stopped
or restarted, and extraction profiles/index signatures were unchanged.

## 2026-10-25: include catalog and reading-guide knowledge alongside PDFs

Registered the two already downloaded supporting files in sources.local.json,
keeping the existing PDF rows. Drive sync now accepts PDFs plus the exact named
index.csv and how_to_read_llm_papers.md sources (including text export for the named
native Google Doc), and continues to skip unrelated files. PDF ingestion still
processes only PDF rows; no extraction profile or vector-storage signature changed.

Added study_materials.py to read the manifest, validate catalog identities and
join default study depths/reasons from the guide's Markdown depth table. It reads
single numbers and ranges from the data rather than hardcoding priority lists.
Generated passages include phase overviews, phase groups, priority groups,
per-paper depth lookups and guide sections, with original source URLs and
content-based IDs. Reference text is searched through BM25 and the shared MiniLM
reranker without embedding or writing to the PDF Chroma collections.

Indexed-paper metadata now includes catalog identity/phase/depth fields when a
filename matches. Planner input includes compact metadata and instructs learning
searches to stay unscoped so references remain eligible. Exact-paper retrieval
still returns only that PDF. Grounding attributes recommendations to the guide,
keeps them separate from scientific findings and retains quote/source/task checks.
Raw guide sections preserve prerequisites and exceptions to default depth rows.
Evaluation configuration hashes reference contents so changed guidance cannot be
mixed into a resumed run unnoticed.

Real-file check: 38 catalog entries, all 38 with default guide depths, and 64 cited
reference passages; largest passage was 1299 characters. The parsed Pass 3 group
matches guide IDs 3, 5, 8, 11, 12 and 19. No vision processing or runtime reading-pass
policy was added. No production embeddings, index reset or live answer generation ran.
The user's running rebuild was not stopped or restarted.

Validation: 135 tests passed in 12.42 seconds. New tests use fictional papers to
verify live depth changes, ranges, phase/priority groups, reference-only retrieval,
exact-paper exclusion, metadata joins, planner context, quote-verified guide
citations, source hashes, malformed/deleted data, download/export handling and
PDF-only ingestion. Test manifests, SQLite, Chroma and reports are isolated.
The planner-payload contract test was updated for the added catalog metadata.
Ruff lint/format and Git whitespace checks passed. These checks establish wiring
and provenance, not measured live-model answer accuracy for learning questions.
