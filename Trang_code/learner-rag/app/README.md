
# Research Paper RAG

A simple RAG application that answers questions about research papers using FastAPI, LangChain, LangGraph, ChromaDB, and Ollama.

## Features

- Extract text from research papers in Google Drive.
- Split documents into chunks.
- Generate embeddings and store them in ChromaDB.
- Retrieve relevant information to answer questions.
- Use Ollama to generate answers locally.
- Provide a FastAPI endpoint for asking questions.

## Tech Stack

- Python
- FastAPI
- LangChain
- LangGraph
- ChromaDB
- Ollama
- PyMuPDF
- PyDrive2

## Project Structure

```text
learner-rag/
├── app/
│   ├── main.py
│   ├── drive.py
│   ├── ingest.py
│   ├── rag.py
│   └── secrets/
├── data/
│   └── chroma_db/
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file:

```dotenv
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_CHAT_MODEL=gemma3
```

### 4. Download Ollama models

```bash
ollama pull nomic-embed-text
ollama pull gemma3
```

Make sure Ollama is running and Google Drive credentials are configured.

## Usage

### 1. Ingest documents

```bash
python -m app.ingest
```

### 2. Run the RAG pipeline

```bash
python -m app.rag
```

### 3. Start the API

```bash
uvicorn app.main:app --reload
```

Open the API documentation at:

http://127.0.0.1:8000/docs

### 4. Ask a question

Send a `POST` request to `/ask`:

```json
{
  "question": "What training losses does vec2vec use to learn its translation?"
}
```

## Security

- Do not commit `.env` or Google Drive credentials.
- Add `.venv/` and `data/chroma_db/` to `.gitignore`.
