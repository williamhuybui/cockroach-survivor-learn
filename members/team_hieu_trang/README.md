# Team Research Agent

For reviewed evidence coverage, stage audits and P50/P95 latency measurement,
see [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md). Relevance labels stay offline;
questions without them still run with unscored evidence metrics.

Ask questions about PDF papers in one Google Drive folder. Answers cite the
paper and page/section. The existing chat UI, Markdown export, custom quizzes
and daily challenge remain available in a single-user workspace.

## Run

Use Python 3.11 or 3.12, Ollama, and the existing Google OAuth files.
No Docker or PostgreSQL server is needed.

```powershell
python -m venv .venv-research
.\.venv-research\Scripts\python.exe -m pip install -r requirements.txt
ollama pull gemma3:4b
ollama pull embeddinggemma
.\.venv-research\Scripts\python.exe -m app.cli init-db
```

Add `RESEARCH_FOLDER_ID=your-folder-id` to your existing `.env`. The ID is the
part after `/folders/` in the research folder's Drive URL. For a fresh install,
copy `.env.example` to `.env` first. Keep `client_secrets.json` and the saved
`credentials.json` in this project folder. The old `GDRIVE_FOLDER_NAMES` setting
is deliberately ignored so an old two-folder setup cannot broaden this sync.

```powershell
.\.venv-research\Scripts\python.exe -m app.drive_sync
.\.venv-research\Scripts\python.exe -m app.ingest sources.local.json
.\.venv-research\Scripts\python.exe -m uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000. Activation is optional because each command uses the
virtual environment's Python directly. The reranker downloads its model on
first use; after that it loads from the local Hugging Face cache. Ollama must
remain running. Read [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) for the
code-reading order and troubleshooting.

## Data flow

1. `drive_sync.py` downloads PDFs from the selected folder and its subfolders.
2. `document_processing.py` checks PDF headings and font styling, keeps paper
   sections together, and falls back to page boundaries when structure is unclear.
3. LangChain splits only oversized sections into at most 1,200 characters with
   120-character overlap. Each child chunk links to its complete parent section.
   These are character limits, not guaranteed token counts.
4. EmbeddingGemma embeds batches of 16 child chunks. Chroma stores child vectors
   for search and complete parent sections for answer context, plus Drive links,
   page locations and document fingerprints.
5. Ordinary questions use one task. Complex requests produce a reviewable plan
   without a fixed three-task cap. The browser shows Approve and Revise. Only the
   exact saved plan executes; unresolved references lead to clarification.
6. Every task uses the same hybrid search: EmbeddingGemma vectors, Chroma cosine
   search and Python BM25. Exact-paper filtering happens before dense top-k.
   A local MiniLM cross-encoder reranks candidates before parent expansion.
7. Task results share one 14,000-character content budget. Shared passages retain
   task associations and receive citation labels once. Conditional edges skip
   retrieval for clarification and skip generation if evidence is empty.
8. Separate graph nodes generate structured claims, verify quotes, sources,
   numbers and task associations, then format an answer, partial answer or
   abstention. Missing tasks are identified explicitly.

The reranker threshold defaults to raw score `0`, a starting setting that needs
evaluation on your papers. Scores are not probabilities. This version uses dense
retrieval, BM25 keyword retrieval, a neural reranker and parent-section context.
It deliberately does not use RRF: the neural reranker makes the final comparison.
Reranking cannot recover a paper missing from the candidate pool.

## Storage and migration

- `data/chroma/`: the research index, including passage text and source metadata.
- `data/app.sqlite3`: chat messages, quizzes, daily challenge state and immutable
  index-run summaries.
- `synced_sources/drive/`: downloaded PDFs for repeatable local ingestion.
- `sources.local.json`: the Drive download manifest.

Chroma and SQLite run locally without separate servers. SQLite is only for
ordinary app state; the RAG index is entirely in Chroma. Existing PostgreSQL
data is left untouched. Old chat/quiz history is not automatically copied;
the new local workspace starts empty. Re-ingest the PDFs to build Chroma.

Run sync and ingestion again when papers change. Unchanged papers skip
embedding; changed ones replace their previous chunks. PDFs removed from a
successful manifest are removed from the index. An empty folder is rejected
so a mistaken folder selection does not clear the index. An unreadable changed
PDF leaves its older indexed version intact and prints a warning.

After changing the embedding model, dimensions, or chunking rules:

```powershell
.\.venv-research\Scripts\python.exe -m app.ingest sources.local.json --rebuild
```

`--rebuild` clears this app's Chroma collection first. Stop the web server while
ingesting or rebuilding; this small app does not coordinate concurrent index
writers. Back up `data/` before an intentional rebuild if you need rollback.

## Test and inspect

```powershell
.\.venv-research\Scripts\python.exe -m pytest -q
.\.venv-research\Scripts\python.exe -m app.retrieval "How does RAG-Token work?"
.\.venv-research\Scripts\python.exe -m app.evaluate path\to\cases.csv --output data\evaluations\run.json
# Continue an interrupted local benchmark without repeating saved answers.
.\.venv-research\Scripts\python.exe -m app.evaluate path\to\cases.csv --output data\evaluations\run.json --resume
# Limit a run to a small restartable batch on memory-constrained machines.
.\.venv-research\Scripts\python.exe -m app.evaluate path\to\cases.csv --output data\evaluations\run.json --resume --max-cases 5
# Give a reviewer a spreadsheet with answers, citations and empty review fields.
.\.venv-research\Scripts\python.exe -m app.review_evaluations export data\evaluations\run.json data\evaluations\review.csv
# Import the completed review CSV and recalculate reviewed accuracy/support metrics.
.\.venv-research\Scripts\python.exe -m app.review_evaluations import data\evaluations\run.json data\evaluations\review.csv
```

The retrieval command prints evidence without calling Gemma. The evaluator accepts
any CSV with `id` and `question` columns and preserves optional labels such as
`type`, `difficulty`, `source_docs`, `paper_scope`, and `expected_behavior`.
It never sends a reference answer or expected source title to the agent. It records source coverage, abstention
behavior, errors and latency. A person must review correctness and claim support
before the report can state accuracy or hallucination rates.

Review CSV exports require a new filename so an existing human review is never
overwritten. Fill `correctness_review` with `correct`, `partial`, or
`incorrect`; fill `citation_support_review` with `supported`, `partial`, or
`unsupported`. Enter both supported and total claim counts together. Blank
fields preserve saved review values. Import rejects duplicate IDs and invalid
counts before writing the report.

Users ask questions in the chat UI. Its `/api/chat` route accepts the message
and conversation ID. The workflow chooses retrieval internally; users do not
construct filter objects. Comparison retrieval can scope itself to discovered
paper titles. Offline evaluation calls the same Python answer function directly.

## Limits

PDF extraction is text-based. Scans need OCR; figures, equations, tables and
unusual two-column layouts need manual review. Section classification is a
heuristic, not a guarantee that a document is a research paper. Exact quotation
and number checks reduce unsupported claims but cannot prove every paraphrase or
resolve conflicting passages. The context is bounded by parent sections and
shortened conversation history; character limits approximate model token usage.
Complexity detection is heuristic. Model-based paper resolution and subquestion
completeness need reviewed evaluation. Task coverage checks do not prove semantic
entailment. Image understanding is deferred.

The app is a single workspace without sign-in. Run it on localhost or a trusted
network. Drive permissions are not reproduced per user.

## Conditional workflow and evaluation compatibility

The question graph is `plan -> retrieve -> generate -> verify -> format_response`.
Clarification and empty evidence route directly to formatting. There are no graph
retry loops. The browser creates quizzes through chat; only the unused standalone
quiz creation endpoint was removed, and quiz submission still works.

Evaluation uses the same question workflow. Reports record code, model settings
and an index fingerprint. Resume rejects changed configurations or indexes;
start a new output file for reports created before this workflow upgrade.

## Review a multipart plan before execution

Multipart plans pause before searching. Approve runs the saved tasks without
replanning. Revise creates a fresh token, invalidating earlier approval buttons.
SQLite pending_plans preserves proposals across refresh. Deleting a chat removes
its plan; failed execution unlocks a proposal for retry. Initialization adds the
table without changing existing messages.

There is no fixed task-count cap on a plan. Execution batches currently handle
up to five searches, derived from the existing 14000-character budget and
2400-character full windows. Larger plans display and save the remaining tasks
for another approval. Ordinary questions remain immediate. Bounded recent user
and assistant messages support follow-up planning, not unrestricted dependent
reasoning. Offline evaluation bypasses review and labels traces automatic_offline.
