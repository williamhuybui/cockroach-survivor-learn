# Evaluate evidence quality and response time

Our first dashboard answers five questions:

1. Did candidate retrieval find the reviewed evidence requirements?
2. Did the final context retain evidence for every required part?
3. Was the answer correct and supported, according to human review?
4. Did the system abstain appropriately?
5. How long did the request take?

## Private relevance labels

Keep the question CSV unchanged: unique `id` and `question` columns are required.
Use an optional, separate JSON file for reviewed evidence requirements:

```json
[
  {
    "question_id": "q1",
    "requirement_id": "method_from_paper_a",
    "paper": "Exact indexed title of Paper A.pdf",
    "quote": "An exact passage from Paper A supporting the requested method."
  },
  {
    "question_id": "q1",
    "requirement_id": "method_from_paper_b",
    "paper": "Exact indexed title of Paper B.pdf",
    "quote": "An exact passage from Paper B supporting the requested method."
  }
]
```

Replace the example IDs, titles and quotations with human-reviewed evidence from
the indexed papers. These examples are not actual source quotations. Use short,
self-contained evidence spans so an arbitrary chunk boundary does not defeat the
match. Matching ignores Unicode/whitespace differences, not words or punctuation.
It requires an exact indexed title. A matching title without matching text fails.

Different requirements represent separate facts or requested parts that must all
be covered. Alternative supporting spans share the same `requirement_id`: finding
any one alternative satisfies that requirement. Labels have stable human-defined
IDs and text, not response-local S1 labels or rebuild-dependent chunk IDs.

Only questions enter the agent. Review labels are loaded and scored by the offline
evaluator after inference; they never select papers or alter generation. No labels
means evidence metrics remain null, not zero. A question with labels and an empty
final context has zero coverage. Missing candidate audits remain unscored.

## Run evaluation

```powershell
.\.venv-research\Scripts\python.exe -m app.evaluate questions.csv --output data/evaluations/new-run.json --relevance relevance.json
```

`--relevance` is optional. Existing paper-title recall remains available as a coarse
diagnostic but is not a passage relevance metric. The new report fields are:

- `candidate_evidence_recall`: fraction of reviewed requirements found anywhere in
  the combined candidate pool across tasks. This is not ranked Recall@K.
- `final_evidence_coverage`: fraction found in the actual budgeted context.
- `all_requirements_covered`: true only when every reviewed requirement is present.
- `missing_requirements`: requirement IDs still absent from that context.
- Existing human correctness/support and abstention metrics remain available.

Each requirement counts once. An annotation set is only as complete as the human
review: a low score can reveal missing labels as well as missed evidence. Correct
quote presence is an observable proxy, not proof that the passage fully answers
the question. Human review still checks interpretation and source conflicts.

## Inspect where evidence was lost

In each result, `retrieval_trace.tasks` stores the task and three passage lists:

```text
candidate_passages -> reranked_passages -> expanded_passages
                                        -> result.context
```

These preserve text, original IDs, metadata and available scores. Candidate union
order is not a global relevance ranking. Reranked passages retain score order;
final context is selected round-robin across tasks. Each final passage retains
`chunk_id`, its short citation `id`, and task associations.

Inspect these lists to identify split facts, irrelevant headings, duplicate text,
bad reranking, or useful evidence removed by the shared budget. Browser responses
do not contain these full passage audits. Large offline reports are intentional;
do not publish private paper text unintentionally.

## Latency

Reports summarize P50 and P95 seconds, plus sample counts, for total request time,
planning, retrieval, generation and verification. Retrieval includes reranking,
parent expansion and evidence selection; detailed retrieval traces still contain
individual search/reranking/expansion timings. Skipped stages have no observations,
not artificial zero durations. Failed cases contribute total latency and error
counts, though their incomplete stage timings are unavailable.

Groups include actual direct/multipart/comparison routes and declared runtime
conditions. `--runtime-state cold|warm|unclassified` is a measurement label, not
an action that loads or unloads models. Default is unclassified; never infer cold
or warm merely from request position. For controlled cold runs reset the runtime
outside evaluation and run one case. For warm runs warm models before the run.
Do not label a whole mixed batch cold. Errors without a returned route are grouped
as unknown. Small-sample P95 is descriptive and unstable; compare matching hardware,
runtime conditions, route mix and sample counts. Percentiles use linear interpolation.

## Human review and repeatability

```powershell
.\.venv-research\Scripts\python.exe -m app.review_evaluations export data/evaluations/new-run.json review.csv
.\.venv-research\Scripts\python.exe -m app.review_evaluations import data/evaluations/new-run.json review.csv
```

Review correctness, supported/total claims and citation support against the papers.
Unreviewed correctness remains null. Benchmark labels are evaluation-only; do not
special-case questions in the application. Review cannot prove all future answers
are safe. Use the saved results to identify a specific failure before tuning.

Resume verifies dataset, relevance-file hash, runtime classification, workflow/code,
model settings and index fingerprints. Start a new report after code or labels
change. Chunking, overlap, reranking thresholds and context settings were not tuned
as part of this instrumentation update. MRR, NDCG and fixed-K precision/recall are
deferred until a ranking-comparison question justifies them.

## Approval mode in evaluation

Browser multipart requests now pause for plan approval. The offline evaluator
still executes automatically through the same graph and records approval_mode as
automatic_offline. It does not simulate approval or revisions. Large plans can
return a continuation proposal after one batch; treat these as incomplete rather
than a full answer to the original request. Existing reports require a new run
after this workflow change.
