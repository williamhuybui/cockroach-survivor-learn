# Team Research Agent

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
5. The planner model writes self-contained search tasks. A question whose
   clauses each start with a question word is split without a model call when
   each clause names its own paper, or the question names one paper ("How many
   parameters does X have and how many tokens was it trained on?"); a clause
   about an unindexed model (GPT-4) stays unresolved. With several papers, a
   clause with a pronoun is split only when it is about the one paper it names:
   "its" after that name ("how many examples does ReAct use in its prompts").
   "with it", "they", a comparison ("compare", "than", "both") or "how many are
   used" leaves the question to the planner model. A task that names exactly
   one indexed paper is searched in that paper only. A paper without a short name
   is also named by the first words of its title plus "paper" ("the Scaling Laws
   paper"); a bare "scaling laws" names nothing. A question about each of
   several named papers ("each", "Which X do A and B use?", a repeated preposition
   as in "in the A paper and in the B paper"), or a planned task
   that names three or more papers, becomes one focused task per paper; the other
   papers' mentions leave with their prepositions, articles and list joints ("in
   A and in B"), and a removal that would break the question keeps the user's own
   wording instead. A task
   naming two papers stays joint: the second is often only a baseline or model
   mentioned in passing ("LoRA ... for GPT-3 175B"). The browser shows Approve and
   Revise for multi-task plans; unattended callers (the evaluation gateway) never
   get a clarification and run every task.
6. Every task uses the same hybrid search: EmbeddingGemma vectors, Chroma cosine
   search and Python BM25. Exact-paper filtering happens before dense top-k.
   A local MiniLM cross-encoder reranks candidates before parent expansion.
   A task with no passage above the threshold keeps its two best passages when
   sibling tasks found evidence.
7. Each task keeps up to five section windows of at most 2,400 characters, and
   all tasks share one 14,000-character budget (about 3.5k prompt tokens; CPU
   prefill costs about 16 s per 1k tokens). Shared passages keep task
   associations and receive citation labels once.
8. A task that takes something for granted ("Why does X use Y?", "Which Y does X
   train?", "How large is the Y that X trains?") also gets a premise search with
   no model call. `premise_search.py` reads the task's paper section by section for
   a sentence of the paper's own model (naming it, or its authors in the abstract,
   introduction or conclusion) where "without", "dispensing with", "foregoes",
   "avoids", "free of", "no" or "does not use" is followed within three words by Y.
   References, acknowledgements and experiment or ablation sections are skipped,
   as are "without compromising Y", a negated cue ("do not eliminate Y"), "with
   and without Y" and a Y that the sentence continues into another noun the question
   never used ("without activation checkpointing" rules out no activation
   function). Such a sentence is added first to the task's evidence and
   answers the task as a quoted correction ("This question assumes something the
   paper rules out. The paper states: ... [S1]"); when every task is corrected,
   Gemma is not called. Without such a sentence the question continues unchanged.
9. One Gemma call answers each task in its own record with source labels; no
   quotes are required. The verifier checks every record without another model
   call: its labels must belong to that task and paper (an extra label from
   another task is dropped only when the answer's values are found in the task's
   own passages), a benchmark year or model size named by the task (2019, 8x7B)
   must appear in the cited text, a requested currency must match, every number
   must appear in the cited passages (2.5B equals 2,500M; a passage's "eight"
   supports 8; a per-unit value cannot be given as a total that the passage also
   states), the values must be stated for a requested property (batch size,
   vocabulary, context length, layers, attention heads, temperature, latency,
   random seed, number of raters, BLEU, GSM8K; "NVIDIA Quadro RTX8000" counts as
   hardware), a one-part "which optimizer / hardware / tokenizer / positional
   encoding / normalization / activation" question must be answered with one by
   name ("Orion uses multi-head attention" names no optimizer; naming one does not
   prove the citation supports it), a request for GPU or TPU hours must be answered in that unit,
   a value requested for one training stage (pre-training, fine-tuning,
   post-training, context extension, reward modeling, reinforcement learning) must
   not be tied by the cited text to another stage or, for a paper that gives that
   stage its own figure in the same unit, to the whole run (`stage_scope.py`; it
   binds each number to its own clause, a scope clause such as "For fine-tuning,",
   the one stage its sentence names or the section heading, accepts any number
   that nothing ties to another stage, and does not read "the entire training
   dataset" as the whole run). A rejected whole-run total is replaced by the
   sentence that states the asked stage's own figure when the passages break the
   total into stage figures that add up to it, the figure is the stage's only one
   in the unit the question names (not a rate such as "per trillion tokens"), and
   that sentence passes all of these checks itself. A value written as a word ("a single
   epoch") counts when the cited text states it, and "according to the X paper"
   needs a citation from paper X: for every task when the question opens with it,
   otherwise only for the tasks about paper X ("Which algorithm does A use, and
   which does the X paper introduce?" lets A's task cite A).
10. Answers contain only verified records; unanswered tasks are listed under
   "Missing information". With no evidence the answer is "I don't know - this is
   not in the documents."; when no record survives verification it is "I don't
   know - the retrieved documents do not provide a supported answer ...".

Without `MIN_RERANK_SCORE` no cutoff is applied; `.env.example` sets 1.5 (see
Run). Scores are not probabilities. This version uses dense
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

## Test

```powershell
.\.venv-research\Scripts\python.exe -m pytest -q
```

Users ask questions in the chat UI. Its `/api/chat` route accepts the message
and conversation ID. The workflow chooses retrieval internally; users do not
construct filter objects. Comparison retrieval can scope itself to discovered
paper titles. The evaluation gateway calls the same Python answer function directly.

## Try the agent through the evaluation gateway

1. Make sure Ollama is running (`curl localhost:11434/api/version`) and no other evaluation run is.
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

On a long run, start Ollama with `LLAMA_ARG_CACHE_RAM=0` (PowerShell:
`$env:LLAMA_ARG_CACHE_RAM = 0; ollama serve`). Its default 8 GiB prompt cache in host RAM
otherwise grows with every distinct question.

## Limits

PDF extraction is text-based. Scans need OCR; figures, equations, tables and
unusual two-column layouts need manual review. Section classification is a
heuristic, not a guarantee that a document is a research paper. Exact quotation
and number checks reduce unsupported claims but cannot prove every paraphrase or
resolve conflicting passages. The context is bounded by parent sections and
shortened conversation history; character limits approximate model token usage.
Complexity detection is heuristic. Model-based paper resolution and subquestion
completeness need reviewed evaluation. Task coverage checks do not prove semantic
entailment. The premise search matches wording: it misses a false premise the paper
rules out only in other words (a critic model excluded just by "no reinforcement
learning") and a paper that both uses and avoids similar things can mislead it.
On 16 held-out false-premise questions written after the build it quoted a sentence
for 7 and the right one for 6; it has no cue for "remove", "rather than" or "frozen",
and one quote was unrelated because it matched "attention", a word of the paper's
own name. A quoted correction is graded correct when the question counts as
answerable, but a grader that files the question as unanswerable (no source
documents) scores a plain "I don't know" +1 and a correction -0.5.
The stage check compares wording only. It cannot bind a number in a flattened table
row that names every stage ("Pre-Training Context Extension Post-Training Total
2664K 119K 5K 2788K"), it ignores a stage named in a neighbouring sentence (often
only an adjective such as "the pre-trained weights"), and it reads a scientific
value such as 1e-4 only in that spelling. It rejects a wrong-stage number only
when the cited text itself shows the other stage; when retrieval never finds the
right stage's passage the answer becomes "I don't know" at best, not the right
value. The repair after a rejection is narrower still: it replaces only a
whole-run total, never another stage's figure; it needs the question to name the
unit ("GPU hours", not "compute") and stage figures in the retrieved text that add
up to the total within 1%; and it never reads table cells, so per-stage dollar
costs stay unanswered. Of the 39 indexed papers only DeepSeek-V3 reports such a
breakdown, so it is the only paper where the repair can fire. Answers without a number (a name, a sentence) are not checked against the
question at all: an off-topic sentence, or a name Gemma knows but the cited text does
not state, is accepted. Three lexical gates were measured on saved drafts and
rejected because each also rejected correct answers or caught nothing: a rare
question word absent from the cited text (about 6 of 20 correct dev questions lost),
a capitalised name in the answer absent from the cited text (caught none of six
new off-topic answers), and the reranker score of the best cited passage or
sentence (correct and wrong answers overlap). Multi-paper questions lose points to
three causes that need different fixes: a paper named informally ("the Scaling
Laws paper") is not resolved, a clause with a pronoun ("... as its tool") keeps the
two-paper question as one task, and table-style or attribute-word facts (layer
counts, GPUs, a normalization name) are not retrieved.
Image understanding is deferred.

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
