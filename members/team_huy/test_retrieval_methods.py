"""Compare the retrieval methods on dev_set.csv.

Methods:  embedding (EmbeddingGemma) | tfidf | bm25 | keyword (exact match on the question's
          distinctive terms) | hybrid (BM25 + embedding, reciprocal rank fusion)

For every answerable dev question, with the top-K chunks of a method:
  all_docs  every expected paper appears among the chunks' papers
  fact_hit  share of the numbers in the ground truth (e.g. 28.4, 41.8, 15.6T) found in the chunk text
Unanswerable questions have no expected paper; only their top score is reported.

    python3 members/team_huy/test_retrieval_methods.py [--k 10]

Needs chunking_output/ (chunking.py) and Ollama running with embeddinggemma. Writes
retrieval_output/method_comparison.csv (one row per question and method).
"""
import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from keyword_search import question_keywords, search as keyword_search  # noqa: E402
from tfidf_bm25 import Bm25Index, TfidfIndex, load_chunks  # noqa: E402

OUT = HERE / "chunking_output"
DEV_SET = HERE.parents[1] / "dev_set.csv"
OLLAMA_URL = "http://localhost:11434/api/embed"


class EmbeddingIndex:
    def __init__(self, chunks):
        emb = pd.read_csv(OUT / "chunks.csv", usecols=["chunk_embedding"]).chunk_embedding
        self.matrix = np.array(emb.map(json.loads).tolist())
        self.matrix /= np.linalg.norm(self.matrix, axis=1, keepdims=True)

    def scores(self, query):
        body = json.dumps({"model": "embeddinggemma", "input": [f"task: search result | query: {query}"]}).encode()
        req = urllib.request.Request(OLLAMA_URL, body, {"Content-Type": "application/json"})
        v = np.array(json.load(urllib.request.urlopen(req, timeout=120))["embeddings"][0])
        return self.matrix @ (v / np.linalg.norm(v))


def top_chunk_ids(method, question, chunks, indexes, k):
    """Chunk ids and top score for one method."""
    if method == "keyword":
        hits = keyword_search(question_keywords(question), mode="any").head(k)
        return list(hits.chunk_id), (float(hits.n_found.iloc[0]) if len(hits) else 0.0)
    if method == "hybrid":  # reciprocal rank fusion: score = sum 1 / (60 + rank) over both rankings
        fused = np.zeros(len(chunks))
        for name in ("bm25", "embedding"):
            order = np.argsort(-indexes[name].scores(question))
            fused[order] += 1 / (60 + np.arange(1, len(order) + 1))
        order = np.argsort(-fused)[:k]
        return list(chunks.chunk_id.iloc[order]), float(fused[order[0]])
    s = indexes[method].scores(question)
    order = np.argsort(-s)[:k]
    return list(chunks.chunk_id.iloc[order]), float(s[order[0]])


def fact_hit(ground_truth, text):
    """Share of the numbers in the ground truth that occur in the text (None if it has none)."""
    nums = set(re.findall(r"\d+(?:\.\d+)?[A-Za-z]?", ground_truth))
    nums = {n for n in nums if len(n) > 1}  # skip lone digits
    if not nums:
        return None
    low = text.lower().replace(" ", "")
    return sum(n.lower() in low for n in nums) / len(nums)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10, help="chunks retrieved per question")
    k = ap.parse_args().k

    chunks = load_chunks()
    docs = pd.read_csv(OUT / "documents.csv")
    stem = dict(zip(docs.doc_id, docs.file.str.replace(".pdf", "", regex=False)))
    text_of = dict(zip(chunks.chunk_id, chunks.chunk_text))
    doc_of = dict(zip(chunks.chunk_id, chunks.doc_id))
    indexes = {"embedding": EmbeddingIndex(chunks), "tfidf": TfidfIndex(chunks), "bm25": Bm25Index(chunks)}
    methods = ["embedding", "tfidf", "bm25", "keyword", "hybrid"]

    rows = []
    for _, q in pd.read_csv(DEV_SET).iterrows():
        expected = [x.strip() for x in str(q.source_docs).split(";")] if q.type != "unanswerable" else []
        for m in methods:
            ids, top = top_chunk_ids(m, q.question, chunks, indexes, k)
            papers = {stem[doc_of[c]] for c in ids}
            rows.append({
                "id": q.id, "type": q.type, "difficulty": q.difficulty, "method": m, "top_score": round(top, 4),
                "all_docs": all(e in papers for e in expected) if expected else None,
                "docs_found": f"{sum(e in papers for e in expected)}/{len(expected)}" if expected else "",
                "fact_hit": fact_hit(q.ground_truth, " ".join(text_of[c] for c in ids)) if expected else None,
                "chunk_ids": json.dumps(ids),
            })
    df = pd.DataFrame(rows)
    out = HERE / "retrieval_output"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "method_comparison.csv", index=False)

    ans = df[df["type"] != "unanswerable"].copy()
    ans["all_docs"] = ans.all_docs.astype(float)
    summary = ans.pivot_table(index="method", columns="type", values="all_docs", aggfunc="sum").reindex(methods)
    n = ans[ans.method == "bm25"].groupby("type").size()
    summary = summary.astype(int).astype(str) + "/" + n.astype(str)  # e.g. 5/6
    summary["fact_hit (avg)"] = ans.groupby("method").fact_hit.mean().reindex(methods).round(2)
    print(f"Top-{k} chunks per question. all_docs = every expected paper retrieved:\n")
    print(summary.to_string())
    print("\nQuestions where a method misses an expected paper:")
    for m in methods:
        miss = ans[(ans.method == m) & (ans.all_docs == 0)]
        print(f"  {m:10} {', '.join(f'{r.id} ({r.docs_found})' for r in miss.itertuples()) or 'none'}")
    print(f"\nWrote {out / 'method_comparison.csv'}")


if __name__ == "__main__":
    main()
