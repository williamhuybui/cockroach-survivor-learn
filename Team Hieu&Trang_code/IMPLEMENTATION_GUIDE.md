# Junior developer guide to the research agent

Evaluation instrumentation: read `evaluation_metrics.py` after `evaluate.py`.
It scores private reviewed evidence requirements at the candidate and final-context
stages and summarizes latency. `EVALUATION_GUIDE.md` explains the label format,
stage audits, cold/warm measurement and human review. Retrieval settings are unchanged.

## 1 Code tree

Start in `Reagan_code`. Files below are source code unless marked generated.

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
  document_processing.py     Classifies PDFs, finds sections and splits children
  ingest.py                  Embeds and updates papers, records index runs
  paper_store.py             Opens child and parent Chroma collections
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
chat-form submission -> sendMessage() -> run() -> api('/api/chat', body)
```

The browser trims the question, displays a user bubble and sends `message` plus
`session_id` as JSON. `startNewChat()` creates a UUID and retains it in
sessionStorage. `run()` manages loading and error messages. Browser async keeps
the interface responsive; backend model searches are sequential.

### Step 2 The request route

Read `ChatRequest` and `chat()` in `main.py`. Pydantic validates a message of
1 to 2000 characters and an optional session ID. FastAPI routes the browser
request to `chat()`. It reads the conversation from SQLite, closes the connection,
calls `respond()`, then saves updated messages and returns JSON.

This class is a data contract, not an AI agent. Keep request validation and the
database cleanup wrapper: both replace error-prone repeated code.

### Step 3 Commands and the graph

Read `respond()` in `services.py`:

- `/papers` calls `list_papers()` without searching or generating.
- `/quiz "topic" medium 3` parses a quiz request and calls `make_quiz()`.
- Other messages call `answer_with_context()`.

Read `ChatState` and graph construction next:

```text
START -> plan -> retrieve -> generate -> verify -> format_response -> END
           |        |
           |        +-- no evidence -----------> format_response
           +-- clarification needed ----------> format_response
```

Nodes do work. Conditional edges choose the next node. There are no automatic
graph retry loops. `chat_graph.invoke()` runs the whole workflow; a later
`model.invoke()` inside generation calls Gemma. They are nested operations.

### Step 4 Planning the question

Read `plan_question()`, `SearchTask` and `QuestionPlan` in `question_planning.py`.
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

In `services.retrieve()`, every task goes through the same `retrieval.search()`:

```text
search -> hybrid_candidates
       -> matching_records
       -> retrieve_candidates -> llm.embed_query -> Chroma.query
       -> keyword_candidates
       -> deduplicate -> rerank_candidates -> expand_parents
```

`matching_records()` loads child text and metadata, optionally scoped to a title.
`retrieve_candidates()` embeds the task question and requests a bounded dense
candidate pool. Exact-paper filtering happens inside Chroma before top-k selection.
`keyword_candidates()` calculates BM25 word counts, rarity and length-normalized
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
`draft_answer()`, `verify_claims()` and `format_answer()`.

`generate_answer()` bounds recent conversation text and calls `draft_answer()`.
Gemma receives the original question, planned tasks and labeled evidence. It must
return separate claims with exact supporting quotes, source IDs and task IDs.
The instructions require comparing the same requested aspects and admitting gaps.

`verify_claims()` builds the normalized passage lookup once. It rejects unknown
sources, missing quotes, changed numeric tokens and injected citation labels.
Claim task IDs must belong to the cited passage and match any scoped paper.
Tasks without accepted support remain missing. This checks declared associations,
not semantic entailment; it cannot prove every paraphrase is true.

`format_response()` is the single graph exit formatter. It returns clarification,
abstention, a complete answer or a partial answer listing unsupported tasks.
No evidence skips Gemma. No accepted claims causes abstention. Sources are built
through one shared `source_cards()` helper.

### Step 7 Model calls

In `llm.py`, read:

- `embedding_model()`: shared Ollama embedding client.
- `embed_texts()`: document prefix and vector dimension/count checks.
- `embed_query()`: search-query prefix using the same embedding model.
- `chat_model()`: bounded Gemma context, output and timeout.
- `generate()`: evidence-only instructions for answers and quizzes.
- `structured_request()`: shared structured model invocation; planning supplies
  different instructions from answer generation.

Untrusted questions and paper content are serialized as data, not interpolated
into system instructions. Pydantic checks output structure, not factual truth.
`@lru_cache` reuses clients and the reranker model; it does not cache answers.

### Step 8 Returning to the browser

`answer_with_context()` returns the final result plus exact evidence and trace.
The browser command router removes full context; offline evaluation retains it.
`main.chat()` saves the user/assistant messages and sources. `app.js` calls
`addMessage()` and `sources()` to display text and HTTPS links, then refreshes
the recent-chat list. A deleted conversation is not silently recreated mid-answer.

## 4 Read indexing separately

1. `drive_sync.sync()` authenticates and downloads PDFs from one folder and its
   subfolders. The manifest records local paths, source IDs, titles and URLs.
2. `ingest.run()` validates the manifest and initializes application storage.
3. `index_paper()` fingerprints content, metadata and pipeline settings. A complete
   unchanged paper skips embedding.
4. `document_processing.extract_sections()` uses PyMuPDF text and heading styles.
   `classify_document()` chooses section-aware processing or page fallback.
5. `chunk_sections()` splits oversized sections with LangChain. Defaults are
   1200 characters including title/heading prefix and 120 configured overlap.
6. `llm.embed_texts()` embeds children in batches of 16. Ingestion finishes
   embedding before writing the replacement paper.
7. `paper_store.collection()` stores child vectors/text/citations in
   `research_papers`. `parent_collection()` stores full sections in
   `research_sections`; their vectors are child centroids. Search uses children,
   then fetches parents by ID. Text offsets are not used in this version.
8. Ingestion replaces changed records and removes papers missing from the valid
   manifest. Empty manifests are rejected. Extraction failure leaves the older
   paper intact. OCR is not configured.
9. `save_index_run()` records status, counts, timestamps and hashes in SQLite.
   These summaries are not full historical copies of the vector index.

Do not change embedding dimensions or chunk rules without a deliberate rebuild.
Chroma checks an index signature to avoid searching incompatible vectors.

## 5 Other files to understand after the main path

`db.open_database()` commits successful work, rolls back errors and always closes
SQLite. `schema.sql` defines settings, index_runs, chat_sessions, quizzes,
quiz_attempts and daily_challenges. `new_id()` creates browser-safe text UUIDs.

`quiz_validation.py` checks quiz structure, distinct options, source IDs and
submitted answers. `services.make_quiz()` saves the answer key server-side.
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

## Plan review in the browser

Read ChatRequest, main.chat(), services.plan(), after_plan(), and format_response().
Multipart proposals route to formatting before retrieval. The UI displays tasks
and Approve/Revise actions. SQLite pending_plans stores the current token and payload.
main.chat() atomically claims that exact token. Approval sends the stored plan to
the graph, which skips replanning. Revisions create a fresh token and require
approval even when reduced to one task. Stale or duplicate approval returns 409.
Model failure unlocks the proposal for retry. Refresh restores current controls;
chat deletion removes its pending plan.

All planned tasks are preserved. Execution uses batches of up to five searches
under the existing context budget. Remaining tasks are explicitly listed and saved
for the next approval. Recent assistant text helps resolve follow-ups. Offline
evaluation uses the automatic_offline mode, not a simulation of user approval.
