"""Keyword (lexical) retrieval over the chunks: TF-IDF and BM25.

Both rank chunks by the words they share with the question; a word that is rare across
chunks but frequent in one chunk scores high. Index text is `embed_text` from chunks.csv,
so the paper title and section title count as words too.

    python3 members/team_huy/tfidf_bm25.py          # evaluates both on dev_set.csv
    from tfidf_bm25 import TfidfIndex, Bm25Index    # index.search("question", k=10)

Needs chunking_output/chunks.csv (run chunking.py first). Writes to retrieval_output/:
    dev_retrieval.csv  every retrieved chunk: question id, method, rank, score, chunk/section/doc ids, paper, text
    dev_eval.csv       one row per dev question: were the expected papers retrieved, and the top score
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer

HERE = Path(__file__).resolve().parent
CHUNKS_CSV = HERE / "chunking_output" / "chunks.csv"
DOCS_CSV = HERE / "chunking_output" / "documents.csv"
DEV_SET = HERE.parents[1] / "dev_set.csv"
OUT_DIR = HERE / "retrieval_output"

BM25_K1 = 1.5  # how fast repeated words stop adding score
BM25_B = 0.75  # how much long chunks are penalised

# "DeepSeek-V3" -> deepseek, v3 ; "405B" stays 405b ; "15.6T" -> 15.6, t. Same rule for chunks and questions.
TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def tokenize(text):
    return [t for t in TOKEN.findall(text.lower()) if t not in ENGLISH_STOP_WORDS]


class TfidfIndex:
    """TF-IDF vectors (log-scaled term frequency) compared by cosine similarity."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(chunks["embed_text"])  # rows are L2-normalised

    def scores(self, query):
        return (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()

    def search(self, query, k=10):
        return top_k(self.chunks, self.scores(query), k)


class Bm25Index:
    """Okapi BM25: idf(word) * tf*(k1+1) / (tf + k1*(1 - b + b*len/avg_len)), summed over question words."""

    def __init__(self, chunks, k1=BM25_K1, b=BM25_B):
        self.chunks = chunks
        self.vectorizer = CountVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
        tf = self.vectorizer.fit_transform(chunks["embed_text"]).tocsr().astype(float)
        n = tf.shape[0]
        df = np.asarray((tf > 0).sum(axis=0)).ravel()
        idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
        length = np.asarray(tf.sum(axis=1)).ravel()
        norm = k1 * (1 - b + b * length / length.mean())
        rows = np.repeat(np.arange(n), np.diff(tf.indptr))  # row index of every stored value
        tf.data = tf.data * (k1 + 1) / (tf.data + norm[rows]) * idf[tf.indices]
        self.matrix = tf

    def scores(self, query):
        q = self.vectorizer.transform([query])
        q.data[:] = 1  # a word repeated in the question counts once
        return (self.matrix @ q.T).toarray().ravel()

    def search(self, query, k=10):
        return top_k(self.chunks, self.scores(query), k)


def top_k(chunks, scores, k):
    order = np.argsort(-scores)[:k]
    out = chunks.iloc[order][["chunk_id", "doc_id", "section_id", "chunk_text"]].copy()
    out.insert(0, "score", scores[order])
    return out.reset_index(drop=True)


def load_chunks():
    return pd.read_csv(CHUNKS_CSV, usecols=["chunk_id", "section_id", "doc_id", "chunk_text", "embed_text"])


def evaluate(indexes, k=10):
    """Dev set: are all expected papers among the papers of the top-k chunks?

    Returns (eval table, table of every retrieved chunk)."""
    docs = pd.read_csv(DOCS_CSV)
    stem = dict(zip(docs.doc_id, docs.file.str.replace(".pdf", "", regex=False)))
    dev = pd.read_csv(DEV_SET)
    rows, retrieved = [], []
    for _, r in dev.iterrows():
        expected = [x.strip() for x in str(r.source_docs).split(";")] if r.type != "unanswerable" else []
        row = {"id": r.id, "type": r.type, "difficulty": r.difficulty}
        for name, index in indexes.items():
            hits = index.search(r.question, k)
            retrieved.append(hits.assign(question_id=r.id, method=name, rank=range(1, len(hits) + 1),
                                         paper=hits.doc_id.map(stem)))
            found = {stem[d] for d in hits.doc_id}
            row[name] = None if not expected else all(e in found for e in expected)
            row[f"{name}_top_score"] = round(hits.score.iloc[0], 3)
        rows.append(row)
    cols = ["question_id", "method", "rank", "score", "chunk_id", "section_id", "doc_id", "paper", "chunk_text"]
    return pd.DataFrame(rows), pd.concat(retrieved)[cols]


def main():
    chunks = load_chunks()
    indexes = {"tfidf": TfidfIndex(chunks), "bm25": Bm25Index(chunks)}
    result, retrieved = evaluate(indexes, k=10)
    OUT_DIR.mkdir(exist_ok=True)
    result.to_csv(OUT_DIR / "dev_eval.csv", index=False)
    retrieved.to_csv(OUT_DIR / "dev_retrieval.csv", index=False)
    print(result.to_string(index=False))
    answerable = result[result["type"] != "unanswerable"]
    print("\nall expected papers within top-10 chunks:")
    for name in indexes:
        for t, g in answerable.groupby("type"):
            print(f"  {name:6} {t:7} {int(g[name].sum())}/{len(g)}")
    print(f"\nWrote dev_eval.csv and dev_retrieval.csv to {OUT_DIR}")


if __name__ == "__main__":
    main()
