import argparse
import shutil
import time

from rag import config
from rag.chunkers import STRATEGIES, chunk_documents
from rag.loaders import read_docs
from rag.models import get_embeddings, get_vectorstore

BATCH = 32  


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default=config.CHUNK_STRATEGY, choices=STRATEGIES)
    ap.add_argument("--size", type=int, default=config.CHUNK_SIZE)
    ap.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP)
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()

    docs = read_docs(config.DOCS_FILE)
    embeddings = get_embeddings(args.fake)

    chunks = chunk_documents(docs, args.strategy, args.size, args.overlap, embeddings)
    print(f"Chunking : {args.strategy}  size={args.size}  overlap={args.overlap}")
    print(f"           {len(docs)} documents -> {len(chunks)} chunks")

    persist_dir = config.chroma_dir(args.fake)
    if persist_dir.exists():
        shutil.rmtree(persist_dir)
    store = get_vectorstore(embeddings, persist_dir)

    start = time.time()
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i : i + BATCH]
        store.add_documents(batch, ids=[c.id for c in batch])
        print(f"  embedded {min(i + BATCH, len(chunks))}/{len(chunks)}", end="\r")
    print()

    dim = len(embeddings.embed_query("test"))
    print(f"Embedding: {'FAKE (random)' if args.fake else config.EMBED_MODEL}  ({dim} dimensions)")
    print("Index    : Chroma, HNSW approximate nearest-neighbour, cosine distance")
    print(f"Done in {time.time() - start:.1f}s  ->  {persist_dir}")


if __name__ == "__main__":
    main()
