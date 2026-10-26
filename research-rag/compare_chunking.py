import argparse

from rag import config
from rag.chunkers import chunk_documents
from rag.loaders import read_docs
from rag.models import get_embeddings, get_vectorstore


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
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()

    docs = read_docs(config.DOCS_FILE)
    embeddings = get_embeddings(args.fake) if (args.query or args.semantic) else None

    strategies = ["fixed", "recursive", "markdown"] + (["semantic"] if args.semantic else [])
    runs = []
    for strategy in strategies:
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

    print("\n=== Example chunk (the 3rd chunk overall) ===")
    for (strategy, size), chunks in results.items():
        if len(chunks) >= 3:
            print(f"\n--- {strategy} (size {size}) ---")
            print(chunks[2].page_content[:500])

    if args.query:
        print(f"\n=== Retrieval for: {args.query!r} (top 3 per strategy) ===")
        for (strategy, size), chunks in results.items():
            store = get_vectorstore(embeddings)  
            store.add_documents(chunks, ids=[c.id for c in chunks])
            print(f"\n--- {strategy} (size {size}) ---")
            for doc, dist in store.similarity_search_with_score(args.query, k=3):
                m = doc.metadata
                print(f"[dist {dist:.3f}] {m['source']} p.{m['page']}: {doc.page_content[:160]!r}")
            store.delete_collection()


if __name__ == "__main__":
    main()
