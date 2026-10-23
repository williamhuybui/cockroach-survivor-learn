# Validation — research RAG update, 2026-10-20

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
- Local Gemma generation took roughly 6–25 seconds per case after model load.
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
