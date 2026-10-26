from langchain_chroma import Chroma
from langchain_core.embeddings import DeterministicFakeEmbedding

from rag import config

COLLECTION = "papers"


def get_embeddings(fake: bool = False):
    if fake:
        return DeterministicFakeEmbedding(size=256)
    from langchain_ollama import OllamaEmbeddings

    emb = OllamaEmbeddings(model=config.EMBED_MODEL, base_url=config.OLLAMA_URL)
    try:
        emb.embed_query("ping")
    except Exception as e:
        raise SystemExit(
            f"Cannot use Ollama embeddings ({type(e).__name__}).\n"
            f"  1) Is Ollama running?   ollama serve\n"
            f"  2) Is the model pulled? ollama pull {config.EMBED_MODEL}\n"
            f"  URL used: {config.OLLAMA_URL}   (change in .env)"
        )
    return emb


def get_llm(fake: bool = False):
    if fake:
        from langchain_core.language_models.fake_chat_models import FakeListChatModel

        return FakeListChatModel(responses=["(fake LLM: run without --fake to get a real answer)"])
    from langchain_ollama import ChatOllama

    return ChatOllama(model=config.LLM_MODEL, base_url=config.OLLAMA_URL, temperature=0)


def get_vectorstore(embeddings, persist_dir=None):
    """persist_dir=None -> in-memory index (used by compare_chunking.py)."""
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(persist_dir) if persist_dir else None,
        collection_metadata={"hnsw:space": "cosine"},
    )
