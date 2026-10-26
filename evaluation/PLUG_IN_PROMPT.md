# Prompt: plug your agent into the evaluation gateway

Copy everything below the line into your coding agent. Set TEAM_ID first (team1 = Hieu & Trang, team2 = Ha & Tri).

---

Your job: implement our team's RAG agent as one Python function inside `api_view.py` at the repo root, so the course evaluation gateway can call it. `TEAM_ID = <team1 or team2>`.

## What to do

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
- Keep it under about 60 seconds per question; the evaluation asks every question in turn.

## Do not

- Do not open, read, copy or train on `evaluation/test_set.csv` or `evaluation/eval_results/`. They hold the hidden answers. Use `dev_set.csv` (public).
- Do not hard-code answers to dev or test questions.

## How to check your work

From the repo root run `uvicorn api_view:app --port 8000`, then:

1. `curl localhost:8000/TEAM_ID/health` returns `{"ok": true}` (a 501 means the placeholder is still there).
2. `curl -X POST localhost:8000/TEAM_ID/rag-agent -H 'Content-Type: application/json' -d '{"question": "What BLEU score does the Transformer get on WMT 2014 English-to-German?"}'` returns `{"answer": "...", "sources": ["..."]}`.
3. A question that is not in the documents (e.g. "Who won the 2018 World Cup?") gets an explicit "I don't know".
4. Quick score on the public dev set (needs `OPENAI_API_KEY` in `.env`): `curl -X POST "localhost:8000/TEAM_ID/evaluate?set=dev&limit=5"`. `errors` in the response must be 0.

When it works, reply with: what you changed in `api_view.py`, any files you added, new dependencies, and the output of check 2.
