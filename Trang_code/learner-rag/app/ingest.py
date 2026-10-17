
import hashlib
import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

os.chdir(PROJECT_ROOT)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.drive import (
    BASE_DIR,
    download_file,
    extract_text,
    get_drive_service,
    list_drive_files,
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

load_dotenv(BASE_DIR / ".env")

CHROMA_DIR = BASE_DIR / "data" / "chroma_db"
COLLECTION_NAME = "research_papers"

EMBEDDING_MODEL = os.getenv("OLLAMA_EMBED_MODEL")

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100


# --------------------------------------------------
# Initialize embeddings and vector store
# --------------------------------------------------

def get_vector_store():
    """Initialize Ollama embeddings and persistent ChromaDB."""

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    return vector_store


# --------------------------------------------------
# Convert PDF pages to LangChain documents
# --------------------------------------------------

def pages_to_documents(file, pages):
    """Create one LangChain Document per extracted PDF page."""

    documents = []

    for page in pages:
        documents.append(
            Document(
                page_content=page["text"],
                metadata={
                    "source": file["name"],
                    "file_id": file["id"],
                    "page": page["page"],
                    "url": file.get("webViewLink", ""),
                },
            )
        )

    return documents


# --------------------------------------------------
# Split documents into chunks
# --------------------------------------------------

#### We split a PDF into chunks so the RAG system 
#### can find the most relevant pieces of information 
### instead of searching through an entire paper at once.

def split_documents(documents):
    """Split pages into smaller chunks while preserving metadata."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    return splitter.split_documents(documents)


# --------------------------------------------------
# Generate deterministic IDs
# --------------------------------------------------

def generate_chunk_id(document, chunk_index):
    """Generate a stable ID for a chunk."""

    identity = (
        f"{document.metadata['file_id']}:"
        f"{document.metadata['page']}:"
        f"{chunk_index}:"
        f"{document.page_content}"
    )

    return hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()


# --------------------------------------------------
# Ingest PDFs into ChromaDB
# --------------------------------------------------

def ingest_pdfs():
    """Download PDFs, extract text, chunk, embed, and store."""

    drive = get_drive_service()
    vector_store = get_vector_store()

    folder_id = os.getenv("DRIVE_FOLDER_ID") or None
    files = list_drive_files(drive, folder_id=folder_id)

    if not files:
        print("No matching PDF papers found in Google Drive.")
        return

    print(f"Found {len(files)} PDF(s).")

    for file in files:
        local_path = None

        try:
            print(f"\nProcessing: {file['name']}")

            # 1. Download the PDF to a temporary file
            local_path = download_file(drive, file)

            # 2. Extract text page by page using PyMuPDF
            pages = extract_text(file, local_path)

            if not pages:
                print("No extractable text found; skipping.")
                continue

            # 3. Convert pages to LangChain Documents
            documents = pages_to_documents(file, pages)

            # 4. Split documents into chunks
            chunks = split_documents(documents)

            if not chunks:
                print("No text chunks generated; skipping.")
                continue

            # 5. Generate stable IDs for the chunks
            ids = [
                generate_chunk_id(chunk, index)
                for index, chunk in enumerate(chunks)
            ]

            # 6. Delete existing chunks for this PDF
            # before inserting the updated chunks
            vector_store.delete(
                where={"file_id": file["id"]}
            )

            # 7. Embed chunks and store them in ChromaDB
            vector_store.add_documents(
                documents=chunks,
                ids=ids,
            )

            total_chars = sum(
                len(page["text"]) for page in pages
            )

            print(f"Pages extracted: {len(pages)}")
            print(f"Characters extracted: {total_chars}")
            print(f"Chunks stored: {len(chunks)}")

        except Exception as exc:
            print(f"Failed to process {file['name']}: {exc}")

        finally:
            # 8. Remove the temporary PDF
            if local_path is not None:
                local_path.unlink(missing_ok=True)

    print(f"\nIngestion finished. ChromaDB location: {CHROMA_DIR}")


# --------------------------------------------------
# Entry point
# --------------------------------------------------

if __name__ == "__main__":
    ingest_pdfs()