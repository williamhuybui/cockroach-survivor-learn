# team_huy (team3)

`agent.py` holds Huy's agent: a LangGraph ReAct agent (plan, search, extract, answer) over the research papers, with
weighted hybrid retrieval (EmbeddingGemma + BM25 + exact keyword). `api_view.py` (repo root) serves it as
`POST /team3/rag-agent` by calling `answer(question, ask_llm, load_documents)`.

## Setup (once)

The agent searches chunks built from the PDFs, so build them first. Ollama must be running with `gemma3` and `embeddinggemma`.

```
ollama pull gemma3 && ollama pull embeddinggemma
python3 members/team_huy/chunking.py        # writes members/team_huy/chunking_output/ (about 1 minute; the output is not committed)
```

## Run and test from the repo root

```
python3 members/team_huy/agent.py "Which GPUs trained the Transformer and DeepSeek-V3?"   # prints the plan, searches, answer
uvicorn api_view:app --port 8000
curl -X POST localhost:8000/team3/rag-agent -H 'Content-Type: application/json' -d '{"question": "..."}'
python evaluation/evaluate.py team3 --test-set dev_set.csv --limit 5
```

## Files

| File | What it does |
|---|---|
| `agent.py` | The ReAct agent and the weighted retriever |
| `chunking.py` | PDFs to documents / sections / chunks CSVs, with EmbeddingGemma embeddings |
| `tfidf_bm25.py` | TF-IDF and BM25 keyword retrieval over the chunks |
| `keyword_search.py` | Exact keyword search ("Ctrl+F" over the chunks) |
| `test_retrieval_methods.py` | Compares embedding, TF-IDF, BM25, keyword and hybrid retrieval on `dev_set.csv` |
| `eda.ipynb` | Exploration of the papers: size, pages, tokens, sections |
