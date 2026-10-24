import argparse

from rag import config
from rag.chunkers import chunk_documents
from rag.loaders import read_docs
from rag.models import get_embeddings, get_llm, get_vectorstore


def stats(chunks):
    lens = [len(c.page_content) for c in chunks]
    clean = sum(1 for c in chunks if c.page_content.rstrip().endswith((".", "?", "!", ":", "\n")))
    return {
        "chunks": len(chunks),
        "avg": round(sum(lens) / len(lens)) if lens else 0,
        "min": min(lens, default=0),
        "max": max(lens, default=0),
        "tiny%": round(100 * sum(1 for n in lens if n < 100) / max(len(lens), 1)),
        "clean_end%": round(100 * clean / max(len(chunks), 1)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[config.CHUNK_SIZE])
    ap.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP)
    ap.add_argument("--semantic", action="store_true")
    ap.add_argument("--query", help="also compare retrieval quality for this question")
    ap.add_argument("--answer", action="store_true", help="with --query: also let Gemma answer from each strategy's chunks")
    ap.add_argument("--k", type=int, default=config.TOP_K, help="chunks given to Gemma with --answer (same default as phase2_ask.py)")
    ap.add_argument("--files", nargs="+", help="only use documents whose file name contains one of these words (much faster)")
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()

    if args.answer and not args.query:
        raise SystemExit("--answer needs --query \"your question\"")
    docs = read_docs(config.DOCS_FILE)
    if args.files:
        words = [w.lower() for w in args.files]
        docs = [d for d in docs if any(w in str(d.metadata.get("source", "")).lower() for w in words)]
        print(f"Using {len(docs)} pages from files matching {args.files}")
    embeddings = get_embeddings(args.fake) if (args.query or args.semantic) else None
    chain = None
    if args.answer:
        from langchain_core.output_parsers import StrOutputParser

        from phase2_ask import PROMPT, format_context  # the exact prompt phase2_ask.py uses

        chain = PROMPT | get_llm(args.fake) | StrOutputParser()

    strategies = ["fixed", "recursive", "markdown"] + (["semantic"] if args.semantic else [])
    runs = []
    for strategy in strategies:
        # semantic chunking decides its own size, so run it once
        for size in ([args.sizes[0]] if strategy == "semantic" else args.sizes):
            runs.append((strategy, size))

    print(f"{'strategy':10} {'size':>5} {'chunks':>7} {'avg':>6} {'min':>5} {'max':>6} {'tiny%':>6} {'clean_end%':>11}")
    print("-" * 64)
    results = {}
    for strategy, size in runs:
        chunks = chunk_documents(docs, strategy, size, args.overlap, embeddings)
        results[(strategy, size)] = chunks
        s = stats(chunks)
        size_label = "auto" if strategy == "semantic" else size
        print(f"{strategy:10} {size_label:>5} {s['chunks']:>7} {s['avg']:>6} {s['min']:>5} {s['max']:>6} "
              f"{s['tiny%']:>6} {s['clean_end%']:>11}")

    # one example chunk per strategy, so people can SEE the difference
    print("\n=== Example chunk (the 3rd chunk overall) ===")
    for (strategy, size), chunks in results.items():
        if len(chunks) >= 3:
            print(f"\n--- {strategy} (size {size}) ---")
            print(chunks[2].page_content[:500])

    if args.query:
        print(f"\n=== Retrieval for: {args.query!r} (top 3 per strategy) ===")
        for (strategy, size), chunks in results.items():
            store = get_vectorstore(embeddings) 
            for i in range(0, len(chunks), 256): 
                batch = chunks[i : i + 256]
                store.add_documents(batch, ids=[c.id for c in batch])
                print(f"  embedded {min(i + 256, len(chunks))}/{len(chunks)}", end="\r")
            print()
            print(f"\n--- {strategy} (size {size}) ---")
            found = store.similarity_search_with_score(args.query, k=max(3, args.k if args.answer else 3))
            for doc, dist in found[:3]:
                m = doc.metadata
                text = " ".join(doc.page_content.split())  # real spaces instead of literal \n
                junk = sum(1 for ch in text if ord(ch) < 32 or ch == "�")
                flag = "   <-- GARBLED TEXT (bad PDF font encoding)" if junk > 0.05 * max(len(text), 1) else ""
                print(f"\n  [dist {dist:.3f}]  {m['source']}  p.{m['page']}{flag}")
                print(f"     {text[:300]}")
            if chain:
                answer = chain.invoke({"context": format_context([d for d, _ in found[: args.k]]), "question": args.query})
                print(f"\n  >>> Gemma's answer from the top {args.k} chunks ({strategy}, size {size}):")
                print("  " + answer.strip().replace("\n", "\n  "))
            store.delete_collection()


if __name__ == "__main__":
    main()