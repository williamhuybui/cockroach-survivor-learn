# Junior developer guide to the research agent

Evaluation instrumentation: read `evaluation_metrics.py` after `evaluate.py`.
It scores private reviewed evidence requirements at the candidate and final-context
stages and summarizes latency. `EVALUATION_GUIDE.md` explains the label format,
stage audits, cold/warm measurement and human review. Retrieval settings are unchanged.

## 1 Code tree

Each application file starts with a short explanation of its role in the flow.
Function docstrings explain what comes in, what comes out and why that step is
needed. Inline comments explain decisions such as keeping exact caption pages or
sharing evidence space between question parts. Start with the file overview, then
follow the entry function and its helpers; you do not need to read every helper
before understanding the main flow.

Start in `Team Hieu&Trang_code`. Files below are source code unless marked generated.

```text
README.md                    Setup and operating commands
IMPLEMENTATION_GUIDE.md       This guide
VALIDATION.md                Recorded checks and evaluation limitations
requirements.txt             Python dependencies
.env.example                 Example local settings
app/
  config.py                  Paths, model names, chunk settings and thresholds
  __init__.py                Makes app a Python package
  drive_sync.py              Downloads PDFs from one Drive folder and subfolders
  document_processing.py     Extracts pages, handles fallback/captions, groups sections and splits children
  neural_pdf_extraction.py   Reuses a stoppable worker for neural text regions
  ingest.py                  Embeds and updates papers, records index runs
  paper_store.py             Opens child and parent Chroma collections
  study_materials.py         Loads catalog phases, paper identities and cited reading guidance
  question_planning.py       Resolves paper references and proposes reviewable tasks
  retrieval.py               Dense search, Python BM25, reranking and parent expansion
  services.py                Conditional graph, command routing and quiz generation
  grounding.py               Structured claims, verification and answer formatting
  llm.py                     Shared model calls with task-specific instructions
  main.py                    Browser routes and saved conversations
  quiz_validation.py         Quiz data contracts, command parsing and grading
  challenge.py               One saved daily quiz question
  db.py                      SQLite transactions, cleanup and unique IDs
  schema.sql                 SQLite table definitions
  cli.py                     init-db command
  evaluate.py                Generic offline evaluation and checkpointed reports
  review_evaluations.py      Human review CSV export and import
  static/
    index.html               Chat interface and quiz dialogs
    style.css                Layout and appearance
    app.js                   Browser requests, messages, source links and quizzes
tests/                       Automated tests with temporary stores and fake models
sources.local.json           Generated source manifest
synced_sources/drive/        Downloaded PDFs
data/chroma/                 Local child vectors and parent sections
data/app.sqlite3             Chat, quizzes, settings and index-run summaries
data/evaluations/            Evaluation reports and human reviews
```

`.env`, Google OAuth files and virtual environments are local setup, not the
first files to read. Never put credentials into Git. No PostgreSQL or Docker
server is required. Chroma is accessed through its Python package; BM25 runs
in Python memory, not a separate search server.

## 2 Understand the three flows

```text
Prepare knowledge
Drive -> download PDFs -> extract sections -> split children -> embed -> Chroma

Answer questions
Browser -> FastAPI -> plan -> retrieve -> generate -> verify -> format -> browser

Evaluate
Question CSV -> same question workflow -> saved outputs -> human review -> metrics
```

Preparation is manual. A new file in Drive is not automatically searchable.
Run sync and ingestion again after sources change. Asking questions does not
download papers or rebuild their embeddings.

EmbeddingGemma creates search vectors. MiniLM scores question/passage pairs.
Gemma 3 plans selected questions and generates answer drafts. These are different
jobs. SQLite stores application records; Chroma stores research retrieval data.

## 3 Read the question path in this order

### Step 1 The browser

Open `static/index.html`, then `static/app.js`. Follow:

```text
chat-form submission -> submitChatMessage() -> runUiAction() -> requestApi('/api/chat', body)
```

The browser trims the question, displays a user bubble and sends `message` plus
`session_id` as JSON. `clearChatForNewConversation()` creates a UUID and retains it in
sessionStorage. `runUiAction()` manages loading and error messages. Browser async keeps
the interface responsive; backend model searches are sequential.

### Step 2 The request route

Read `ChatRequest` and `handle_chat_request()` in `main.py`. Pydantic validates a message of
1 to 2000 characters and an optional session ID. FastAPI routes the browser
request to `handle_chat_request()`. It reads the conversation from SQLite, closes the connection,
calls `handle_chat_message()`, then saves updated messages and returns JSON.

This class is a data contract, not an AI agent. Keep request validation and the
database cleanup wrapper: both replace error-prone repeated code.

### Step 3 Commands and the graph

Read `handle_chat_message()` in `services.py`:

- `/papers` calls `build_paper_catalog_response()` without searching or generating.
- `/quiz "topic" medium 3` parses a quiz request and calls `create_and_save_quiz()`.
- Other messages call `answer_research_question()`.

Read `ChatState` and graph construction next:

```text
START -> plan -> retrieve -> generate -> verify -> format_response -> END
           |        |
           |        +-- no evidence -----------> format_response
           +-- clarification or plan review ---> format_response
```

These are short LangGraph node labels. The functions behind them are:

| Node label | Function in services.py |
|---|---|
| plan | plan_search_tasks() |
| retrieve | retrieve_task_evidence() |
| generate | generate_answer_draft() |
| verify | verify_answer_draft() |
| format_response | build_chat_response() |


Nodes do work. Conditional edges choose the next node. There are no automatic
graph retry loops. `chat_graph.invoke()` runs the whole workflow; a later
`model.invoke()` inside generation calls Gemma. They are nested operations.

### Step 4 Planning the question

Read `plan_question_tasks()`, `SearchTask` and `QuestionPlan` in `question_planning.py`.
Ordinary questions produce one search task without a planning model call.
Explicit paper references and clearly complex wording trigger structured planning.
The planner sees indexed titles, the user question and one recent user question;
it never sees evaluation answers.

Each task has a focused question and an optional exact paper title. Plans contain
no fixed task-count cap. Unknown titles and incomplete comparison scopes lead to
clarification instead of silently selecting unrelated search hits. Complexity
detection is heuristic, and model-based title resolution needs reviewed evaluation.

Example: a comparison of CoT and ReAct becomes two method questions, each scoped
to an exact catalog title. A missing paper produces a request to identify it.

### Step 5 Retrieving evidence

In `services.retrieve_task_evidence()`, every task goes through the same `retrieval.retrieve_relevant_passages()`:

```text
retrieve_relevant_passages -> find_hybrid_candidates
       -> load_scoped_chunks
       -> find_vector_candidates -> llm.embed_search_question -> Chroma.query
       -> find_bm25_candidates
       -> deduplicate -> rerank_passage_candidates -> expand_parent_section_windows
```

`load_scoped_chunks()` loads child text and metadata, optionally scoped to a title.
`find_vector_candidates()` embeds the task question and requests a bounded dense
candidate pool. Exact-paper filtering happens inside Chroma before top-k selection.
`find_bm25_candidates()` calculates BM25 word counts, rarity and length-normalized
scores in Python. No keyword index is saved to SQLite or Chroma.

Candidate lists are unioned and deduplicated; incompatible dense and BM25 scores
are not averaged. A local cross-encoder assigns one relevance score per pair.
Candidates below `MIN_RERANK_SCORE` are dropped. Raw scores are not probabilities.
Parent expansion gives surrounding context, up to 2400 characters per window.

Across tasks, the graph selects full windows round-robin under one 14000-character
content budget, reserving a share for each task. Shared passages retain all task
associations. Citation labels S1, S2 and so on are assigned once after selection.
These are character limits, not exact model-token limits.

Diagnostics contain every task run, aggregated candidate/reranked source titles,
counts and timings. Inspect these when retrieval selects the wrong papers.

### Step 6 Generating and verifying

Read `DraftAnswer` and `SupportedClaim` in `grounding.py`, followed by
`generate_grounded_draft()`, `verify_answer_claims()` and `format_verified_answer()`.

`generate_answer_draft()` bounds recent conversation text and calls `generate_grounded_draft()`.
Gemma receives the original question, planned tasks and labeled evidence. It must
return separate claims with exact supporting quotes, source IDs and task IDs.
The instructions require comparing the same requested aspects and admitting gaps.

`verify_answer_claims()` builds the normalized passage lookup once. It rejects unknown
sources, missing quotes, changed numeric tokens and injected citation labels.
Claim task IDs must belong to the cited passage and match any scoped paper.
Tasks without accepted support remain missing. This checks declared associations,
not semantic entailment; it cannot prove every paraphrase is true.

`build_chat_response()` is the single graph exit formatter. It returns clarification,
abstention, a complete answer or a partial answer listing unsupported tasks.
No evidence skips Gemma. No accepted claims causes abstention. Sources are built
through one shared `build_source_citations()` helper.

### Step 7 Model calls

In `llm.py`, read:

- `get_embedding_model()`: shared Ollama embedding client.
- `embed_document_chunks()`: document prefix and vector dimension/count checks.
- `embed_search_question()`: search-query prefix using the same embedding model.
- `get_chat_model()`: bounded Gemma context, output and timeout.
- `generate_from_evidence()`: evidence-only instructions for answers and quizzes.
- `request_structured_response()`: shared structured model invocation; planning supplies
  different instructions from answer generation.

Untrusted questions and paper content are serialized as data, not interpolated
into system instructions. Pydantic checks output structure, not factual truth.
`@lru_cache` reuses clients and the reranker model; it does not cache answers.

### Step 8 Returning to the browser

`answer_research_question()` returns the final result plus exact evidence and trace.
The browser command router removes full context; offline evaluation retains it.
`main.handle_chat_request()` saves the user/assistant messages and sources. `app.js` calls
`appendChatMessage()` and `renderSourceLinks()` to display text and HTTPS links, then refreshes
the recent-chat list. A deleted conversation is not silently recreated mid-answer.

## 4 Read indexing separately

### Learning references alongside the PDFs

Read `study_materials.py` for the small catalog/guide path. Its functions do this:

```text
sources.local.json -> load_study_sources()
-> load_paper_catalog(): CSV identities/phases + the guide's depth table
-> load_study_passages(): cited phase/priority groups and guide sections
-> retrieval.load_scoped_chunks() -> BM25 candidates -> MiniLM -> answer verification
```

The source files are not inserted into the PDF vector collections. Reference
passages use keyword search and the same relevance model as PDF candidates.
`list_indexed_papers()` joins catalog metadata to PDFs actually present in Chroma,
so the planner can resolve paper numbers and canonical names without pretending
that every catalog paper is already indexed. Planner metadata stays compact.

Learning questions use an unscoped search so the guide/catalog can be retrieved;
specific-PDF research questions retain exact-paper filtering. Answers attribute
study recommendations to the guide and scientific claims to research evidence.
The guide's complete text remains available for prerequisites and exceptions to
the default depth table. It is reference data, not application instructions.

Both files are identified by their manifest titles: `index.csv` and
`how_to_read_llm_papers.md`. Drive sync includes them in addition to PDFs and can
export the named guide from a Google Doc as text. Structured depth joining expects
the Markdown Depth/Papers/Why table; other guide text is still searchable without
that table. Missing optional sources leave PDF retrieval available; malformed,
duplicate or missing referenced files raise an error rather than silently using
an outdated cached copy. Tests isolate the manifest path as well as databases.

### PDF indexing

The identifiers used by this flow each have a consumer:

- The Drive folder ID selects what to sync. The Drive file ID becomes the manifest
  source ID (`gdrive-...`), saved as `document_id` in Chroma.
- A section ID identifies stored section text. Each child has its own Chroma ID and
  a `parent_id` linking to that section. `section_index` is a temporary processing
  position used to build that link, not another permanent section ID.
- Fingerprints detect changed papers/settings; index-run IDs identify SQLite run summaries.
- `session_id` selects conversation history; `plan_id` identifies the exact proposal
  being approved. `pending_plan_id` returns that same plan ID so the browser can
  disable outdated approval buttons.
- Task positions connect each question part to its evidence. `S1` source labels
  connect claims to selected evidence; `chunk_id` preserves the original chunk ID.
  Task positions and source labels are local to a batch/answer.
- Catalog paper numbers join filenames and phases to guide recommendations. Study
  passage IDs identify cited guide/catalog text. Evaluation case/requirement IDs
  match saved results to private labels and human reviews. Quiz, attempt and daily
  challenge keys identify saved learning records.

New records do not include unused Drive `parent_ids` or child `child_index` fields.
Older manifests/indexes may still contain these harmless fields; readers ignore
them. No index rebuild is needed for this metadata cleanup.

1. `drive_sync.sync_research_folder()` downloads PDFs and the two named learning-reference
   files from one folder and its subfolders. The manifest records local paths, source IDs,
   titles and URLs. PDF ingestion ignores the reference rows; study_materials.py reads them.
2. `ingest.ingest_research_papers()` validates the manifest and initializes application storage.
3. `index_research_paper()` fingerprints content, metadata and pipeline settings. A complete
   unchanged paper skips embedding.
4. `document_processing.extract_pdf_sections()` calls
   `extract_text_pages()` in the same file. `NeuralPageReader.read_page()` sends
   each page to a reusable worker. The model identifies text, heading and caption
   regions in reading order. `extract_region_text()` reads their native PDF text;
   table, image and formula regions are skipped. Table grids are not reconstructed.
   Extraction errors or unusable results use `extract_fast_page()` for that page.
   Fallback pages are marked uncertain and never inherit section hierarchy.
   `group_layout_pages_into_sections()` preserves section nesting and page spans.
   Without headings, prose stays grouped by page. Captions become separate records
   containing only the caption paragraph, its label and one exact PDF page.
   Page numbers must not become headings; code fences protect code comments.
   OCR stays off. Missing text and unreadable characters are reported.
5. `split_sections_into_chunks()` splits oversized sections with LangChain. Defaults are
   1200 characters including title/heading prefix and 120 configured overlap.
6. `llm.embed_document_chunks()` embeds children in batches of 16. Ingestion finishes
   embedding before writing the replacement paper.
7. `paper_store.open_chunk_store()` stores child vectors/text/citations in
   `research_papers`. `open_parent_section_store()` stores full sections in
   `research_sections`; their vectors are child centroids. Search uses children,
   then fetches parents by ID. Child and parent metadata also carry section_path,
   heading_level. Sections form a flat list; parent_id points from a child chunk
   to its containing text section. Heading breadcrumbs describe nesting without
   linking sections into a tree. Retrieval keeps the breadcrumb in the child
   prefix and citation location. Text offsets are not stored.
8. Ingestion replaces changed records and removes papers missing from the valid
   manifest. Empty manifests are rejected. Extraction failure leaves the older
   paper intact. OCR is not configured.
9. `save_index_run_summary()` records status, counts, timestamps and hashes in SQLite.
   These summaries are not full historical copies of the vector index.

Do not change embedding dimensions or chunk rules without a deliberate rebuild.
Chroma checks an index signature to avoid searching incompatible vectors.

## 5 Other files to understand after the main path

`db.open_database()` commits successful work, rolls back errors and always closes
SQLite. `schema.sql` defines settings, index_runs, chat_sessions, quizzes,
quiz_attempts and daily_challenges. `create_record_id()` creates browser-safe text UUIDs.

`quiz_validation.py` checks quiz structure, distinct options, source IDs and
submitted answers. `services.create_and_save_quiz()` saves the answer key server-side.
`challenge.py` saves one daily question and retains its first answer. The browser
creates quizzes through chat; submission routes remain available to the UI.
There is no separate manual research-filter API or standalone quiz creation route.

`evaluate.py` accepts any CSV with unique id and question columns. Only questions
enter the agent. Reference labels remain private to scoring. Results are saved
after each case and failed cases can be retried with resume. Code, model settings
and indexed IDs/fingerprints are recorded; changed configurations cannot resume
an old report. Use a new filename after this workflow upgrade.

`review_evaluations.py` exports answers to a review CSV, validates imported review
fields, and updates metrics. Unreviewed correctness is unknown, not zero or perfect.
Human review still measures whether quotes actually support the claims.

## 6 Run and inspect the application

From the project folder, use the project Python or activate its environment:

```powershell
.\.venv-research\Scripts\python.exe -m app.cli init-db
.\.venv-research\Scripts\python.exe -m app.drive_sync
.\.venv-research\Scripts\python.exe -m app.ingest sources.local.json
.\.venv-research\Scripts\python.exe -m uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`. Inspect retrieval without answer generation:

```powershell
.\.venv-research\Scripts\python.exe -m app.retrieval "What is attention?"
.\.venv-research\Scripts\python.exe -m pytest -q
```

## 7 Build from scratch in this order

1. Settings and storage: config.py, db.py, schema.sql and paper_store.py.
2. Embedding and structured model clients in llm.py.
3. Extract, chunk and index one local PDF before adding Drive sync.
4. Retrieve and inspect passages before asking Gemma to answer.
5. Add structured claims and verification in grounding.py.
6. Add bounded question planning and the conditional graph in services.py.
7. Connect FastAPI and the browser, then save conversations.
8. Add quizzes and daily challenge using the existing retrieval functions.
9. Evaluate real questions and human-review support before tuning thresholds.

## 8 Troubleshooting and limits

- Wrong evidence: inspect task paper scopes, candidate sources and reranker results.
- No answer: inspect missing evidence and rejected claims before loosening checks.
- Stale knowledge: rerun Drive sync and ingestion; uploads alone do not update it.
- Unreadable PDF: scans need OCR; unusual layouts, figures and tables need review.
- Model failure: confirm Ollama and configured models, then inspect server logs.
- A changed pipeline cannot resume an old evaluation: start a new report.

There is no external AI judge, unrestricted agent loop, asynchronous backend
retrieval or answer cache. The focus is a readable, bounded research workflow.

## Reading the smaller helpers

PDF interpretation starts with neural region detection. The worker keeps one
model across papers and skips table-grid reconstruction. A small converter reads
native text from accepted regions. Section grouping then adds heading breadcrumbs,
and LangChain splits oversized sections into children. Fast text extraction is
reserved for failed or slow pages, with uncertain structure recorded explicitly.

Keep helpers that represent important operations. group_layout_pages_into_sections()
collects section text and page spans. retrieve_relevant_passages() runs the full
retrieval pipeline. verify_answer_claims() checks sources, quotes, numbers and task
scope before accepting a claim. Filename cleanup and paper hashing stay beside
the code using them instead of becoming extra one-use wrappers.

Small functions and decorators are not automatically unnecessary. `open_database()`
owns commit, rollback and connection cleanup. Model caching avoids loading clients
and the reranker repeatedly. FastAPI decorators register routes; Pydantic validators
reject invalid input. These remain because removing them would duplicate work or
remove behavior. The browser helpers likewise group API calls, safe text rendering
and UI actions that are used throughout the page.

## Plan review in the browser

Read ChatRequest, main.handle_chat_request(), services.plan_search_tasks(), choose_route_after_planning(), and build_chat_response().
Multipart proposals route to formatting before retrieval. The UI displays tasks
and Approve/Revise actions. SQLite pending_plans stores the current token and payload.
main.handle_chat_request() atomically claims that exact token. Approval sends the stored plan to
the graph, which skips replanning. Revisions create a fresh token and require
approval even when reduced to one task. Stale or duplicate approval returns 409.
Model failure unlocks the proposal for retry. Refresh restores current controls;
chat deletion removes its pending plan.

All planned tasks are preserved. Execution uses batches of up to five searches
under the existing context budget. Remaining tasks are explicitly listed and saved
for the next approval. Recent assistant text helps resolve follow-ups. Offline
evaluation uses the automatic_offline mode, not a simulation of user approval.


## Function names describe actions

Use a verb and the thing being acted on. A name should explain what happens before
the reader opens the function. Standard Python CLI entry points still use main().

| Name | Meaning |
|---|---|
| extract_text_pages() | Read neural pages, recover failed extraction and report coverage. |
| extract_pdf_sections() | Group prose and create caption records with exact pages. |
| group_layout_pages_into_sections() | Collect blocks under detected headings and preserve page spans. |
| split_sections_into_chunks() | Create searchable children from the extracted sections. |
| open_chunk_store() | Open the Chroma collection containing child chunks. |
| retrieve_relevant_passages() | Run hybrid retrieval, reranking and parent expansion. |
| answer_research_question() | Run the research graph and return its result and evidence. |
| verify_answer_claims() | Check a draft's quotes, numbers, sources and task associations. |

The browser follows submitChatMessage() -> runUiAction() -> requestApi(). The
backend follows handle_chat_request() -> handle_chat_message() ->
answer_research_question(). Function names changed; API URLs, JSON keys and saved
record formats did not. Previously saved chats and plans retain their format.

## Current PDF extractor: neural regions and text

The matching PyMuPDF/PyMuPDF4LLM/layout packages remain pinned to 1.28.2.
`neural_pdf_extraction.create_region_model()` loads a neural region classifier
with reading-order detection. It disables the separate table-grid component.
Version 1.28.2 has no public switch for this, so the internal adapter is isolated,
version-checked and tested. Upgrade it deliberately when upgrading the library.
The classifier still uses visual page features internally; image export, table
cell reconstruction and OCR are not performed.

The execution order is:

```text
extract_pdf_sections() -> extract_text_pages() -> NeuralPageReader.read_page()
-> run_region_worker(): model.predict() -> extract_region_text()
-> group_layout_pages_into_sections() -> split_sections_into_chunks()
```

Read `document_processing.py` first. Its entry point `extract_pdf_sections()` is
near the top, followed by page extraction and fallback helpers, section grouping,
text cleanup and chunk creation. It contains the complete processing flow.
`neural_pdf_extraction.py` contains only the model worker, region-to-text conversion
and process lifecycle/cleanup; read it when you need those technical details.

For a slow page, the branch stays inside `document_processing.py`:

```text
extract_text_pages() -> extract_fast_page() -> identify_markdown_blocks()
-> mark structure uncertain -> group as a page-based section
```

`save_extraction_report()` writes coverage metadata after the pages are read.
Caption type comes from the neural model or the caption-label pattern; caption
records with exact pages are created by `extract_pdf_sections()`.

The converter reads native text only from text, title, heading, caption, list,
footnote and code regions, in predicted reading order. Table, picture, formula,
running-header and footer regions are excluded. Neural heading font sizes suggest
relative depth; explicit section numbering such as 2.1 takes precedence. A heading
marked `model_detected` is a model result, not a measured confidence probability
or a guarantee that its structure is correct.

One worker is reused across pages and papers. Slow work continues until it
completes, fails or is cancelled. `PDF_LAYOUT_WARN_SECONDS=30` prints periodic elapsed
waiting-time warnings without switching methods. The warning interval is a
progress-display setting, not a quality threshold or a library requirement.

Errors, worker crashes or blank neural results despite available native PDF text
trigger `extract_fast_page()`. A scanned page without native text cannot be
recovered without OCR. Missing headings alone do not trigger extraction fallback.
Fast extraction disables neural
layout, images, graphics, background-color detection and table reconstruction.
Its heading guesses are off: these pages become page-based parents marked
`structure_confidence=uncertain`, with a `review_reason`. Heading breadcrumbs stop
at each fallback page so later text cannot inherit an uncertain hierarchy.

There are no hard extraction timeout settings, deadline calculations or fallback
rules based on elapsed time. The worker is checked for crashes while waiting.
Ctrl+C cancels the command; the owning ingestion process closes its worker.
Short join waits during explicit worker shutdown allow a graceful exit before
termination; they never interrupt a page merely because extraction is slow.
No asynchronous backend is introduced.

Each section keeps its own text and heading breadcrumb, avoiding
subsection duplication. Our cleanup preserves ordinary hyphens and numbers;
the upstream Markdown extractor can still rewrap or join hyphenated words.
Captured caption paragraphs get their own section record with reference_label,
evidence_type=caption and an exact page. Chroma stores those fields with the
usual paper URL. A caption's page does not inherit a whole section's page span.
Sources expose locations such as page 7: Table 2. A caption says what its words
say; it does not prove table-cell values. Gemma is instructed to report that gap
when an answer requires unread cells or graph values. Verification still checks
source IDs, quotations, numbers and task associations.

Fallback pages do not semantically classify every raw PDF text fragment.
Unstructured table cells, graph labels and inline formulas can remain in fallback
text. Neural region labels can also be wrong. Neither path interprets a figure.
Only explicit prose/caption statements should support factual answers; quoting
isolated numbers does not establish what they mean. Caption labels without a
clear marker are retained when the neural model identifies a caption, but cannot
be given a specific Table/Figure label. Fallback caption detection needs an
explicit label. Wrapped caption boundaries can be imperfect.
Exact caption pages are PDF page numbers, not printed journal numbering.

`data/extraction_reports/<path-hash>.json` records page times, extraction methods,
replacement-character counts, fallback reasons and neural/fallback page totals.
It stores metadata rather than passage text and is replaced on the next extraction
of that path. Review it before trusting section-based answers from fallback pages.
These flags are also stored on Chroma parents/children and sent to the grounded
answer prompt, which limits uncertain evidence to explicit quoted facts.

### Restarting and resuming

Stop an old running ingestion before restarting; a running Python process keeps
its already-loaded extractor. Start the new code without --rebuild:

```powershell
.\.venv-research\Scripts\python.exe -m app.ingest sources.local.json
```

The current v7 storage/vector layout remains compatible. A separate extraction
profile is included in each paper fingerprint and saved in Chroma metadata.
Old-profile papers refresh once, preserving existing records until replacement
extraction and embedding succeed. Subsequent resumes skip complete unchanged
papers, including completed papers with fallback pages. Adjusting either time
limit changes the extraction profile and refreshes papers once; normal resumes
do not automatically retry the same fallbacks. No production index was reset by
this code update. Incompatible older
index signatures and changed vector settings still require an explicit rebuild.

Progress logs show each page time, extraction totals, embedding batch number,
embedding totals, storage time and total paper time. No asynchronous backend or
embedding cache was added. An extraction-only preview remains available through
examples.trace_document_processing; its page progress is written to stderr.

## Example: a nested section becomes searchable evidence

PDF heading "2 Methods" -> section 0 (its own overview text)
-> heading "2.1 Training" -> section 1 (its own text, breadcrumb Methods > Training)
-> LangChain children of section 1 -> EmbeddingGemma -> Chroma.

For a training question, hybrid search and MiniLM score the child text with the
prefix "Section: 2 Methods > 2.1 Training". Parent expansion fetches section 1's
text, while the result retains that full path and its page span. No section-to-section
link is needed, and section 0 does not contain a copy of the training text.
Citations still use section page spans rather than exact child pages.

The current text/caption update uses per-paper extraction fingerprints. Resume
without --rebuild for an existing v7 index; old-profile papers refresh once.
Existing chats are unchanged.

### Flat sections and heading breadcrumbs

`group_layout_pages_into_sections()` starts a new section at every detected
heading, including 2.1 and 2.1.1. A short stack holds only heading titles and
levels to build paths such as Methods > Training > Optimizer. It does not hold
section IDs or connect section records into a tree. Each section stores its own
text once. Numbered headings with an unrelated or missing containing heading do
not receive a guessed breadcrumb, and fallback pages clear the remembered path.

`parent_id` still connects a child chunk to its containing text section for
context expansion. The unused `parent_section_index` and `ancestor_section_id`
fields were removed from code and new records. Old stored ancestor metadata is
ignored; this cleanup does not change embedding dimensions or section text.
