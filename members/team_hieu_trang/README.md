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

Set `MIN_RERANK_SCORE=1.5` in `.env`. Git does not track `.env`, so an existing
copy keeps the old `0` until you change it; `.env.example` already has 1.5.
Passages the reranker scores below this value are dropped, so an off-topic
question reaches Gemma with no evidence and abstains instead of answering from
loosely related text. Every answerable dev and unseen question scores above 1.6:
a lower value lets more off-topic questions through, a higher one can drop real
answers. Restart the server after changing it.

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

The two learning-reference files are included alongside the PDFs. `index.csv`
supplies paper numbers, canonical titles, phases, years and links.
`how_to_read_llm_papers.md` supplies reading-depth recommendations, prerequisites,
focus notes and study advice. The agent cites these files for learning guidance;
the recommendations do not change PDF extraction or require visual interpretation.

`study_materials.py` reads these small files from the manifest when needed. It
builds phase/priority groups and heading-based guide passages for BM25 search and
the shared MiniLM reranker. PDF vectors remain in Chroma. Updating the reference
files requires no paper re-embedding; reference changes are tracked by evaluation
hashes so an old evaluation cannot resume against different guidance.

Both existing downloaded files have been registered in `sources.local.json`.
The next Drive sync also downloads them, while continuing to exclude unrelated
non-PDF files. Restart the web server to load changed Python code; an active
PDF rebuild can continue. For a custom manifest, set `SOURCE_MANIFEST_PATH` in
`.env` to the same file used by sync and ingestion.

1. `drive_sync.py` downloads PDFs plus `index.csv` and `how_to_read_llm_papers.md`
   from the selected folder and its subfolders. Other file types remain excluded.
2. `document_processing.py` reads neural text, heading and caption regions through
   one reusable worker in `neural_pdf_extraction.py`. Table-grid reconstruction
   and image export are skipped. Slow neural work prints periodic warnings and
   continues; page extraction and model startup have no time cutoffs.
   Extraction errors or unusable results use page-based text fallback marked
   structurally uncertain. Completed
   neural pages keep nested headings; captions keep exact PDF pages. OCR is off.
   Metadata reports in `data/extraction_reports` show coverage and fallback reasons.
3. LangChain splits only oversized sections into at most 1,200 characters with
   120-character overlap. Each child chunk links to its complete parent section.
   Sections are a flat list with heading breadcrumbs such as Methods > Training;
   no section-to-section tree is stored. Detected subsections keep their own text.
   These are character limits, not guaranteed token counts.
4. EmbeddingGemma embeds batches of 16 child chunks. Chroma stores child vectors
   for search and complete parent sections for answer context, plus Drive links,
   page locations and document fingerprints.
5. Simple questions skip the planner model call and run as one task. Questions
   that list several items ("A, B and C", "each", "both") or name a paper go to
   the planner, which writes self-contained sub-questions. A task that names
   exactly one indexed paper is searched in that paper only. The browser shows
   Approve and Revise for multi-task plans; unattended callers (the evaluation
   gateway) never get a clarification and run every task.
6. Every task uses the same hybrid search: EmbeddingGemma vectors, Chroma cosine
   search and Python BM25. Exact-paper filtering happens before dense top-k.
   A local MiniLM cross-encoder reranks candidates before parent expansion.
   A task with no passage above the threshold keeps its two best passages when
   sibling tasks found evidence.
7. Up to eight 1,200-character windows per task share one 10,000-character
   budget (about 2.5k prompt tokens; CPU prefill costs about 16 s per 1k tokens).
   Shared passages keep task associations and receive citation labels once.
8. One Gemma call writes at most four claims with short exact quotes. The
   verifier then checks each claim without another model call: the quote must
   exist (case and PDF hyphenation ignored) and relate to the claim, numbers must
   appear in the quote, a year/version/size named by the task (2019, 8x7B) must
   appear in the cited passage, and a difference ("by more than 2.0") must not be
   stated as a value. A claim citing the wrong sentence is repaired from a
   supporting sentence of the same paper. When an abstract and a body passage give
   different numbers for the same result, the abstract value is kept.
9. Answers contain only verified claims. If none survive, the answer is
   "I don't know - this is not in the documents." Unverified tasks are listed
   in `missing_tasks`, not in the answer text.

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


### Inspect PDF sections and chunks

Use `tests/inspect_chunks.py` to inspect how research PDFs are processed before embedding and indexing.

The script uses the existing `extract_sections()` and `chunk_sections()` functions from `app/document_processing.py`.

It helps evaluate:
- Section detection and heading accuracy.
- Section boundaries and page locations.
- Chunk size distribution.
- Chunk content and readability.
- Section and child chunk metadata.

#### Usage

Run the commands from the project root with the Python environment activated.

**macOS / Linux**

```bash
# Inspect the default paper: Attention Is All You Need
PYTHONPATH=. python tests/inspect_chunks.py

# Inspect another paper by partial filename
PYTHONPATH=. python tests/inspect_chunks.py "Chinchilla"

# Inspect BERT
PYTHONPATH=. python tests/inspect_chunks.py "BERT"

# Display the first 10 chunks
PYTHONPATH=. python tests/inspect_chunks.py --samples 10

# Inspect a specific PDF using its path
PYTHONPATH=. python tests/inspect_chunks.py "/path/to/paper.pdf"
```

**Windows PowerShell**

```powershell
# Inspect the default paper
.\.venv-research\Scripts\python.exe -m tests.inspect_chunks

# Or run the script directly from the project root
$env:PYTHONPATH = "."
.\.venv-research\Scripts\python.exe tests\inspect_chunks.py

# Inspect another paper
.\.venv-research\Scripts\python.exe tests\inspect_chunks.py "Chinchilla"

# Display 10 chunks
.\.venv-research\Scripts\python.exe tests\inspect_chunks.py --samples 10
```

#### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `paper` | PDF filename, partial filename, or file path | `Attention` |
| `--samples` | Number of sample chunks to display | `5` |

When no paper is specified, the script searches for **Attention Is All You Need**.

For partial filenames, the script searches recursively under `synced_sources/drive/`. If multiple PDFs match, provide a more specific name or an explicit path.

### Inspection output

The script displays three sections:

1. **Paper summary and extracted sections**
   - Document classification.
   - Total number of sections and chunks.
   - Detected section headings.
   - Page locations and section lengths.

2. **Chunk statistics**
   - Minimum, maximum, average, and median chunk lengths.
   - Number of chunks shorter than 200 characters.
   - Number of chunks exceeding `MAX_CHUNK_CHARACTERS`.

3. **Sample chunks**
   - Actual chunk text.
   - Section heading and page location.
   - `section_index` and `child_index`.
   - Extraction strategy and character length.

#### What to check

Review the extracted sections before evaluating chunk quality.

- Are major paper headings detected correctly?
- Does each section contain the appropriate content?
- Are chunk sizes within the configured limit?
- Are chunks readable and meaningful independently?
- Do chunk metadata and locations match the source PDF?

The script is read-only. It does not generate embeddings, update ChromaDB, or modify source PDFs.

**Note:** This is a manual inspection tool, not an automated quality evaluation. Chunk size compliance does not guarantee good semantic boundaries or retrieval performance.

### Chunk Inspection Scoring

The `tests/inspect_chunks.py` script evaluates PDF section extraction and chunking quality using a **rule-based score out of 100**, without requiring an LLM.

The goal is to identify structural issues before embedding chunks into ChromaDB.

#### Scoring Methodology

| Metric | Weight | Description |
|---|---:|---|
| Size compliance | 30 | Percentage of chunks within the 1,200-character limit |
| Metadata completeness | 25 | Presence of `location`, `strategy`, `section_index`, and `child_index` |
| Content substance | 25 | Chunk length relative to a 200-character threshold; shorter chunks receive partial credit |
| Content cleanliness | 20 | Percentage of chunks without predefined PDF-noise patterns |
| **Total** | **100** | **Sum of all four metrics** |

**Calculation:**

- **Size:** `30 × (valid chunks / total chunks)`
- **Metadata:** `25 × (present fields / expected fields)`
- **Substance:** `25 × average(min(chunk length / 200, 1))`
- **Cleanliness:** `20 × (1 - noisy chunks / total chunks)`

The script also flags short sections (under 100 characters), undersized or oversized chunks, missing metadata, and possible PDF artifacts.

#### Usage

Run from the project root:

```bash
# Default: Attention Is All You Need
PYTHONPATH=. python tests/inspect_chunks.py

# Inspect another paper
PYTHONPATH=. python tests/inspect_chunks.py "BERT"

# Display 10 sample chunks
PYTHONPATH=. python tests/inspect_chunks.py --samples 10
```

The default filename pattern is `Attention_Is_All_You_Need`, avoiding conflicts with `FlashAttention`.

#### Interpretation

A higher score indicates better compliance with the configured structural checks. However, **the score does not measure semantic coherence, retrieval accuracy, or answer correctness**.

For example, a chunk containing unrelated paragraphs could still receive full points if it meets all four checks.

Use the score alongside manual section/chunk inspection and separate retrieval evaluation. The weights and thresholds are initial heuristics and may be refined after testing multiple research papers.

## Measure accuracy and latency

`scripts/benchmark.py` runs questions through the same call as `api_view.team1_agent`
and records answers, per-stage timings, LLM calls and tokens, and memory.
`scripts/judge.py` grades the run with the official grader's `judge()`
(gpt-4o-mini, +1 correct, +0.5 partial, 0 idk, -0.5 hallucinated; unanswerable
questions score +1 only for an explicit abstention). It needs `OPENAI_API_KEY`
in the environment or this folder's `.env`. Results go to `data/evaluations/`,
which is not committed.

Pick the level that matches what changed:

| Level | Questions | Time | Use for |
|---|---|---|---|
| Unit tests | `tests/test_agent.py`, `test_grounding.py`, `test_question_planning.py`, `test_hybrid.py` | < 1 min | Verifier, formatting and planner logic |
| Smoke | `smoke_set.csv` (D01, D04, D05, D09, D13: numbers, multipart, two papers, unanswerable, five papers) | ~10 min | Planning, retrieval or prompt changes |
| Full | `../../dev_set.csv` (15) and `unseen_set.csv` (10, papers outside the dev set) | ~50 min | Before accepting changes or a test-set run |

```powershell
.\.venv-research\Scripts\python.exe -m pytest -q tests/test_agent.py tests/test_grounding.py tests/test_question_planning.py tests/test_hybrid.py
.\.venv-research\Scripts\python.exe scripts\benchmark.py smoke_set.csv --output data\evaluations\smoke.json
.\.venv-research\Scripts\python.exe scripts\judge.py data\evaluations\smoke.json --dataset smoke_set.csv
# Full check
.\.venv-research\Scripts\python.exe scripts\benchmark.py ..\..\dev_set.csv --output data\evaluations\dev.json
.\.venv-research\Scripts\python.exe scripts\judge.py data\evaluations\dev.json
.\.venv-research\Scripts\python.exe scripts\benchmark.py unseen_set.csv --output data\evaluations\unseen.json
.\.venv-research\Scripts\python.exe scripts\judge.py data\evaluations\unseen.json --dataset unseen_set.csv
```

On macOS use `.venv-research/bin/python` and forward slashes. Run one benchmark
at a time: Ollama serves one request at a time (`OLLAMA_NUM_PARALLEL` defaults
to 1), so parallel runs slow each other and push answers past the grader's 120 s
timeout. `benchmark.py` has no HTTP timeout; check `seconds` against 120 yourself.
Never run these scripts on the hidden test set or read its per-question results;
use only the score the dashboard reports.

### Try the agent through the evaluation gateway

1. Make sure Ollama is running (`curl localhost:11434/api/version`) and no other benchmark is.
2. From the repo root, start the gateway and leave the terminal open:
   `members/team_hieu_trang/.venv-research/bin/python -m uvicorn api_view:app --port 8000`
3. Open http://127.0.0.1:8000/docs.
4. `GET /{team}/health` with team `team1` -> *Try it out* -> *Execute* returns `{"ok": true}`.
5. `POST /{team}/rag-agent` with team `team1` and body `{"question": "..."}` returns `answer` and
   `sources`. Expect 1.5-3 minutes per question; the first one also loads the models.
6. `POST /{team}/evaluate` scores a set: team `team1`, `set=dev`, `limit=5`, `workers=3`. Check
   that `errors` is 0. With `set=test` and no `limit`, the run is an official practice run: it updates
   the leaderboard and is allowed once every 7 days. A run with `limit` is not.
7. Do not ask questions elsewhere (chat UI, another tab) while a run is going; Ollama serves one
   request at a time. Stop the gateway with Ctrl+C.

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

There is no fixed task-count cap on a plan. Reviewed execution batches handle
up to five searches; larger plans display and save the remaining tasks for
another approval. Unattended runs execute every task in one batch. Ordinary questions remain immediate. Bounded recent user
and assistant messages support follow-up planning, not unrestricted dependent
reasoning. Offline evaluation bypasses review and labels traces automatic_offline.

### Restart indexing after an extraction update

Stop the previous indexing process with Ctrl+C, then start a new process:

```powershell
.\.venv-research\Scripts\python.exe -m app.ingest sources.local.json
```

For the current v7 index, no full reset is needed. The new extraction profile is
part of each paper's fingerprint, so papers built with the old extractor refresh
once. Later resumes skip papers completed with this profile. Changed embedding
models/dimensions or incompatible older index versions still require a deliberate
rebuild. Never run two ingestion processes against the same Chroma directory.

Progress shows each PDF page, extraction duration, embedding batches, storage
writes and total paper time. Caption pointers support their explicit text and
page location; they do not establish values inside an unread table or graph.
`PDF_LAYOUT_WARN_SECONDS=30` controls periodic progress warnings only. Slow
neural work keeps running until it completes, fails or the operator cancels.
There are no extraction timeout settings or elapsed-time fallback rules.
Ctrl+C cancels the command and its worker is cleaned up. Fallback, embedding
and storage are separate stages with their own progress messages.
The new extraction policy refreshes old paper fingerprints once. Unchanged resumes
skip completed papers, including papers with reported fallback pages.
See IMPLEMENTATION_GUIDE.md for the code-reading order and limitations.
