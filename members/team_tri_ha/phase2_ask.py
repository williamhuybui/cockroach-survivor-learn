import argparse

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from rag import config
from rag.models import get_embeddings, get_llm, get_vectorstore

PROMPT = ChatPromptTemplate.from_template(
    """You answer questions about research papers using ONLY the context below.
If the context does not contain the answer, say "I could not find this in the documents."
After each claim, cite the source like [file p.page].

Context:
{context}

Question: {question}

Answer:"""
)


def format_context(docs) -> str:
    return "\n\n".join(f"[{d.metadata['source']} p.{d.metadata['page']}]\n{d.page_content}" for d in docs)


def show_chunks(results):
    print("\n--- Retrieved chunks (dist: smaller = more similar) ---")
    for rank, (doc, dist) in enumerate(results, start=1):
        m = doc.metadata
        print(f"\n#{rank}  dist={dist:.3f}  {m['source']}  p.{m['page']}  (chunk {m['chunk_index']})")
        print(doc.page_content[:400].replace("\n", " "))


def ask(question, store, llm, k, retrieve_only):
    results = store.similarity_search_with_score(question, k=k)
    show_chunks(results)
    if retrieve_only:
        return
    chain = PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": format_context([d for d, _ in results]), "question": question})
    print("\n--- Answer ---")
    print(answer)
    print("\nSources:", ", ".join(sorted({f"{d.metadata['source']} p.{d.metadata['page']}" for d, _ in results})))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--k", type=int, default=config.TOP_K)
    ap.add_argument("--retrieve-only", action="store_true")
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()

    persist_dir = config.chroma_dir(args.fake)
    if not persist_dir.exists():
        raise SystemExit("No index found. Run phase2_index.py first.")

    store = get_vectorstore(get_embeddings(args.fake), persist_dir)
    llm = None if args.retrieve_only else get_llm(args.fake)

    if args.question:
        ask(args.question, store, llm, args.k, args.retrieve_only)
    else:
        while True:
            q = input("\nQuestion (empty to quit): ").strip()
            if not q:
                break
            ask(q, store, llm, args.k, args.retrieve_only)


if __name__ == "__main__":
    main()
