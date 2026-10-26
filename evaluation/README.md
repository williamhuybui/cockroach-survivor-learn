`python scripts/download_google_drive.py`

`uvicorn api_view:app --reload` (from the repo root; `api_view.py` now lives there)

## Evaluating a team

`api_view.py` (repo root) is the gateway. Each team implements one function in its own marked section of that file: `team1_agent(question)`, `team2_agent(question)` (and `team3_agent`, Huy's own agent, currently a simple placeholder), each returning `{"answer": "...", "sources": ["name", ...]}`. The gateway exposes them as `POST /team1/rag-agent`, `/team2/rag-agent`, `/team3/rag-agent`, all in one process on one port. A team that has not plugged in yet answers 501.

```
uvicorn api_view:app --port 8000                    # gateway, from the repo root
# put OPENAI_API_KEY=... in the repo-root .env (gitignored)
python evaluation/evaluate.py team1 --mode practice                    # hidden test set, once per week
python evaluation/evaluate.py team2 --test-set dev_set.csv --limit 5   # public dev set, board not touched
```

An OpenAI model (default gpt-4o-mini, `--judge-model` to change) grades each answer as correct / partial / idk / hallucinated.
Review the `hallucinated` rows in `eval_results/*.csv`, fix verdicts by hand, then re-run with `--rescore`.

## One-call evaluation (no question needed)

`POST /<team>/evaluate` walks the whole question set itself: it asks the team's agent every question, judges each answer and returns the score.

```
curl -X POST "localhost:8000/team3/evaluate?set=dev&limit=5"   # public dev set, quick check
curl -X POST "localhost:8000/team1/evaluate"                   # hidden test set; a full run updates the leaderboard
```

Query options: `set=test|dev` (default test), `limit=N`, `workers=N`, `judge_model=...`.
The response has the score and counts only; per-question answers and ground truths stay in `evaluation/eval_results/`.
It blocks until the run finishes (about 5 seconds per question), so use `/docs` or a long curl timeout.

## team3 (Huy's agent, code in members/team_huy)

`team3_agent` in `api_view.py` is where Huy develops his own agent. For now it is a simple placeholder: keyword retrieval over the documents, answered by gemma3. It also serves as a worked example and to test the pipeline end to end. It reads `tmp/pdf_table.csv`.

```
uvicorn api_view:app --port 8000
python evaluation/evaluate.py team3 --test-set dev_set.csv --limit 5
```

team3 is not on the leaderboard, so a full run leaves `docs/scores.csv` unchanged.

## Plugging in another team

Give the other team's coding agent [PLUG_IN_PROMPT.md](PLUG_IN_PROMPT.md). It tells them which function to implement in `api_view.py`, the rules for `answer` and `sources`, and how to verify.
