# Team Knowledge

A small chat app for your team's Google Drive documents.

Ask questions with citations, export answers as Markdown, and practise with quizzes.
The top-right quiz icon opens one shared Daily Challenge each day.

## Run the updated app

From this folder in PowerShell:

    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    ollama pull gemma3:4b
    .\.venv\Scripts\python.exe -m app.cli init-db
    .\.venv\Scripts\python.exe -m uvicorn app.main:app --reload

Open http://127.0.0.1:8000. The shared chat workspace opens immediately.
Keep PostgreSQL and Ollama running. Virtual-environment activation is optional
because the commands use its Python executable directly.

The update uses Gemma 3 (4B) for answers and quizzes. Embeddings still use
embeddinggemma, so existing documents do not need to be embedded again.
The database command adds missing tables and the keyword index without deleting data.

For a new installation, follow [the setup guide](IMPLEMENTATION_GUIDE.md).

## Refresh your knowledge

Drive sync is limited to the **Meeting notes** and **RAG-data** folders configured by
`GDRIVE_FOLDER_NAMES`. It searches for those exact folders in My Drive, then reads their
supported files and subfolders. It does not ingest other My Drive folders.

    .\.venv\Scripts\python.exe -m app.drive_sync
    .\.venv\Scripts\python.exe -m app.ingest sources.local.json

Run both commands when Drive documents change. Chat uses the latest successful
ingestion; it does not fetch Drive on every question.

The ingestion pipeline chooses a strategy by file type: PDF pages, Word headings
and tables, PowerPoint slides, Markdown headings, Python functions/classes, SQL
statements, CSV row batches, or a plain-text fallback. Natural sections stay whole;
only oversized sections are split further.

After changing chunking rules or the embedding model, rebuild once:

    .\.venv\Scripts\python.exe -m app.ingest sources.local.json --rebuild

## Use the interface

- Ask questions and follow the source links under each answer.
- Hover over a recent chat and select × to delete it.
- Save an answer as Markdown for Obsidian or another notes app.
- Open the quiz icon for today's question.
- Type `/quiz "Alembic" easy 5` for a custom quiz.
- Difficulty: easy, medium, hard. Question count: 1–10.

While Gemma is working, the composer shows elapsed seconds and a spinner, followed by a
visible completion message. The active chat ID is saved before generation begins, so a
browser refresh returns to the same conversation instead of creating another recent chat.

Daily questions are generated on first open and reused across the team.
The workspace's first answer is saved. Explanations appear after submission.

## Read the code

| File | Responsibility |
| --- | --- |
| app/main.py | Shared HTTP routes and chat storage |
| app/services.py | LangGraph retrieval → answer; custom quizzes |
| app/challenge.py | Shared daily question and result |
| app/retrieval.py | pgvector + PostgreSQL keyword search, merged with RRF |
| app/llm.py | Local chat and embedding clients |
| app/drive_sync.py | Google Drive downloads and exports |
| app/document_processing.py | File-aware extraction and adaptive chunking |
| app/ingest.py | Parse, chunk, and embed changed documents |
| app/db.py, app/schemas.py | Data models and input/output validation |
| app/config.py | Environment configuration |
| app/static/ | Plain HTML, CSS, and JavaScript |

The organization follows the team's [OpenLine project](https://github.com/williamhuybui/cockroach-db):
purpose-based modules, explicit functions, centralized configuration, and comments explaining
why steps are needed. No frontend build system is required.

## Tests

    .\.venv\Scripts\python.exe -m pytest -q

Tests use fixtures and mocked model responses. They do not ingest sample documents into the
knowledge database. See [validation results](VALIDATION.md).

## Current limits

The app is one shared workspace without individual accounts. Anyone who can reach the
FastAPI server can use its chat history and indexed collection, so keep it on a trusted
local network unless you add access control at deployment. Drive permissions are not copied
per file. Weekly summaries need session_date metadata in sources.local.json.
Scanned PDFs need OCR before ingestion.

Hybrid retrieval uses native PostgreSQL full-text ranking plus pgvector, not BM25.
Neural reranking is deferred until evaluation demonstrates a benefit. Python selects
explicit quiz commands; Gemma does not need native tool calling. Answers display when
generation finishes; token streaming is not implemented.
