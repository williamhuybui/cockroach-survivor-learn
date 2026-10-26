# team_huy (team3)

`agent.py` holds Huy's agent. `api_view.py` (repo root) serves it as `POST /team3/rag-agent` by calling `answer(question, ask_llm, load_documents)`.
It is a placeholder for now (keyword retrieval + gemma3); replace it with the real agent here.

Run and test from the repo root:

```
uvicorn api_view:app --port 8000
curl -X POST localhost:8000/team3/rag-agent -H 'Content-Type: application/json' -d '{"question": "..."}'
python evaluation/evaluate.py team3 --test-set dev_set.csv --limit 5
```
