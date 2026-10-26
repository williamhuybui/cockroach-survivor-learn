# Prompt: plug your agent into the evaluation gateway

Copy everything below the line into your coding agent. The only thing to change is the first line: write 1 or 2.

---

I am on team **<1 or 2>**. Your task: fix our team's placeholder in `api_view.py` (repo root) so the course evaluation gateway can call our RAG agent. Find our row in this table:

| Team | Members | Placeholder to replace in `api_view.py` | Our folder | Our existing agent |
|------|---------|------------------------------------------|------------|--------------------|
| 1 | Hieu & Trang | `team1_agent = not_implemented("team1")` | `members/team_hieu_trang/` | FastAPI app in `app/`: `app/main.py` (`POST /api/chat`), `app/services.py` (`respond()`, `answer_with_context()`), retrieval in `app/retrieval.py` |
| 2 | Ha & Tri | `team2_agent = not_implemented("team2")` | `members/team_tri_ha/` | `rag/` package: `rag/agent.py` (`Agent` class), `rag/config.py` (settings); `phase3_ask.py` shows how to ask a question, `eval_dev.py` how we score on the dev set |

Import notes:
- Team 1's code is a package named `app`. Team 2's code does `from rag import config`, so `members/team_tri_ha/` must be on `sys.path` when it runs. Do any `sys.path` change inside the function, not globally, and make sure it cannot clash with other teams' `app` packages.
- Ignore `members/ha/` (an unrelated demo), and every folder that is not ours.

## How to fix the placeholder

1. Open `api_view.py` and find your section, marked `# === TEAM 1: ... ===` or `# === TEAM 2: ... ===`. It currently contains a placeholder:

   ```python
   team1_agent = not_implemented("team1")
   ```

   Replace that line with a real function in the same place:

   ```python
   def team1_agent(question: str) -> dict:
       ...
       return {"answer": "<string>", "sources": ["<document name>", ...]}
   ```

   (use `team2_agent` for team 2). The name must stay exactly `team1_agent` or `team2_agent`; the gateway already routes `/team1/rag-agent` and `/team2/rag-agent` to it.
2. Put your helpers, constants and imports inside your own section (prefix helper names with your team, e.g. `t1_`, so they cannot clash). Do not edit any other team's section, the gateway section at the bottom, `evaluation/evaluate.py`, or other members' folders.
3. Larger code (a vector store, an ingestion script, a database) can live in your own folder under `members/`. Import it from your section. Add new dependencies to your folder's `requirements.txt` and list them in your final message.
4. Use gemma3 through the existing `ask_llm(prompt)` helper (Ollama at localhost:11434). Read the documents with `load_documents()`, which returns rows with `file_name`, `file_link` and `content` from `tmp/`, or from an index you build yourself. If `tmp/pdf_table.csv` is missing, run `python scripts/download_google_drive.py` then `python scripts/extract_pdf.py`. Any API keys go in the repo-root `.env`, which is gitignored.
5. `team3_agent` in the same file, with its code in `members/team_huy/agent.py`, is a worked example: the section in `api_view.py` stays tiny and forwards to code in the member's folder.

## Rules for the function

- `answer` is a plain string: the final answer only, no chunk dumps, no chain-of-thought.
- `sources` is a list of plain strings: the file name or title of each document the answer used (e.g. `03_Attention_Is_All_You_Need.pdf`). Empty list if none.
- Every question is independent. Do not rely on a session or on chat history.
- If the documents do not contain the answer, say so explicitly, e.g. "I don't know - this is not in the documents." Wrong answers cost -0.5, an honest "I don't know" costs 0, and for questions that cannot be answered from the documents it scores +1.
- Handle your own failures: return an `answer` that explains what went wrong instead of letting an exception escape for normal questions.
- Make the function thread-safe: the evaluation asks several questions at once (4 by default), so build indexes and load models once, behind a lock, not per call.
- Keep it under about 60 seconds per question; the evaluation asks every question in turn.

## Do not

- Do not open, read, copy or train on `evaluation/test_set.csv` or `evaluation/eval_results/`. They hold the hidden answers. Use `dev_set.csv` (public).
- Do not hard-code answers to dev or test questions.

## How to check your work

From the repo root run `uvicorn api_view:app --port 8000`, then (TEAM_ID is `team1` or `team2`, whichever is ours):

1. `curl localhost:8000/TEAM_ID/health` returns `{"ok": true}` (a 501 means the placeholder is still there).
2. `curl -X POST localhost:8000/TEAM_ID/rag-agent -H 'Content-Type: application/json' -d '{"question": "What BLEU score does the Transformer get on WMT 2014 English-to-German?"}'` returns `{"answer": "...", "sources": ["..."]}`.
3. A question that is not in the documents (e.g. "Who won the 2018 World Cup?") gets an explicit "I don't know".
4. Quick score on the public dev set (needs `OPENAI_API_KEY` in `.env`): `curl -X POST "localhost:8000/TEAM_ID/evaluate?set=dev&limit=5"`. `errors` in the response must be 0.

When it works, reply with: what you changed in `api_view.py`, any files you added, new dependencies, and the output of check 2.
