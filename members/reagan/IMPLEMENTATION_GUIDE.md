# Local setup with PowerShell

FastAPI serves the website. LangGraph coordinates retrieval and answers.
PostgreSQL stores text, vectors, chats, and quizzes. Ollama runs the models.

## 1. Prepare the tools

Use Python 3.11 or 3.12, PostgreSQL, and Ollama. Reuse your existing installations.
Install pgvector for your PostgreSQL version using the
[official Windows instructions](https://github.com/pgvector/pgvector#windows).

If compiling pgvector, run cl and nmake in the **same x64 Visual Studio developer terminal**.
Set PGROOT to the actual PostgreSQL installation. PowerShell uses $env:TEMP; %TEMP% is cmd syntax.

## 2. Install packages

    Set-Location 'C:\Users\hieup\team_knowledge_agent_starter'
    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Skip creating the virtual environment when it already exists. Activation is optional.

## 3. Configure the app

For a new installation only, copy .env.example to .env. Preserve any existing .env.

| Variable | Value or purpose |
| --- | --- |
| DATABASE_URL | postgresql+psycopg://user:password@localhost:5432/team_knowledge |
| CHAT_MODEL | gemma3:4b |
| EMBED_MODEL | embeddinggemma |
| EMBED_DIM | 768 |
| GDRIVE_FOLDER_NAMES | Exact My Drive folder names, separated by commas |
| TEAM_TIMEZONE | America/Chicago or your team's IANA timezone |

Keep .env, client_secrets.json, and credentials.json private.
LangSmith tracing is optional and off by default. Enabling it may transmit prompts and
retrieved passages to your LangSmith project.

## 4. Prepare the database

For a new installation, use pgAdmin or psql as the PostgreSQL administrator:

    CREATE USER team WITH PASSWORD 'choose-a-strong-password';
    CREATE DATABASE team_knowledge OWNER team;

Connect to team_knowledge as an administrator and run:

    CREATE EXTENSION IF NOT EXISTS vector;

Then, in the project folder:

    .\.venv\Scripts\python.exe -m app.cli init-db

The command adds missing tables, the internal shared-workspace record, and a GIN keyword
index while preserving existing records. No application account needs to be created.

## 5. Prepare models

    ollama pull gemma3:4b
    ollama pull embeddinggemma
    ollama list

Keep Ollama running at http://localhost:11434.
Gemma 3 answers and generates quizzes. EmbeddingGemma produces the search vectors.
Changing the chat model alone does not require re-indexing.

## 6. Connect Google Drive

Use the team's existing PyDrive2 OAuth pattern:

1. Enable Google Drive API in your Google Cloud project.
2. Create a Desktop OAuth client and download its JSON.
3. Save it in this folder as client_secrets.json.
4. If consent is in testing, add your Google account as a test user.
5. Set `GDRIVE_FOLDER_NAMES=Meeting notes,RAG-data` in .env.

The sync resolves both exact names from the root of My Drive and recursively reads their
supported files and subfolders. If duplicate folders have the same name, use
`GDRIVE_FOLDER_IDS=first-folder-id,second-folder-id` instead.

Run:

    .\.venv\Scripts\python.exe -m app.drive_sync
    .\.venv\Scripts\python.exe -m app.ingest sources.local.json

The first command opens consent when needed, downloads documents, and builds the manifest.
The second extracts and embeds changed documents. Later syncs reuse credentials.

Supported files: native Google Docs and Slides, PDF, DOCX, PPTX, Markdown, text, Python, SQL, CSV.
Image-only files need OCR/transcription. No sample data is included in ingestion.

Chunking follows the document instead of using one fixed rule. PDFs stay page-aware,
Word content is grouped under headings, slides stay together, Markdown follows headings,
Python follows functions/classes, SQL follows statements, and CSV files repeat their
header in each row batch. Only oversized sections receive character overlap.

For weekly summaries, add a session_date field in YYYY-MM-DD format to the relevant
sources.local.json records and ingest again. Optional author and topic fields are preserved
by future Drive syncs.

## 7. Start the app

    .\.venv\Scripts\python.exe -m uvicorn app.main:app --reload

Open http://127.0.0.1:8000. The chat opens directly; try citations, Markdown export,
and the quiz icon.

The default server is local. This version has no application login, so anyone who can reach
the server can use the shared history and knowledge. Shared hosting needs an access layer,
HTTPS, and deliberate network configuration.

## Understand the flow

1. FastAPI validates the request.
2. Explicit /quiz commands use the quiz service.
3. Chat enters LangGraph: retrieve, then answer.
4. Retrieval merges keyword and vector rankings with reciprocal rank fusion (RRF).
5. Gemma produces structured JSON; Python validates it and its citations.
6. The browser safely renders text and provides export.

Daily Challenge selects an indexed document by date, generates a question on first open,
and stores it in PostgreSQL. A transaction lock prevents concurrent duplicate generation.
Correct answers, explanations, and citations are revealed after submission.

## Troubleshooting

- Run Drive sync **and** ingestion after documents change.
- Missing model: pull gemma3:4b or embeddinggemma.
- Missing table: rerun the init-db command above.
- Google access blocked: check the OAuth test-user list and signed-in account.
- Revoked Google token: rerun Drive sync and complete the fresh browser sign-in.
- Invalid quiz output: narrow the topic or request fewer questions.
- Chunking, embedding-model, or dimension changes require a deliberate full rebuild:
  `.\.venv\Scripts\python.exe -m app.ingest sources.local.json --rebuild`.
- Empty Drive folders do not automatically clear existing knowledge.
- The sidebar shows the last successful ingestion time, not a live Drive check.
- Weekly summary requests require manually recorded session dates; they are not inferred.
