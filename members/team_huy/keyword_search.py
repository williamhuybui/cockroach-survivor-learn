"""Exact keyword search over the chunks: which papers, sections and chunks contain these words?

Mimics how a person checks a paper: Ctrl+F for "405B" or "DeepSeek", then read the text around it.
Matching is case-insensitive on whole terms ("V3" does not match "V32"); a keyword with spaces or
hyphens ("DeepSeek-V3", "Llama 3") must appear exactly like that.

    python3 members/team_huy/keyword_search.py 405B DeepSeek V3          # chunks with ANY keyword
    python3 members/team_huy/keyword_search.py 405B "15.6T" --all         # chunks with ALL keywords
    python3 members/team_huy/keyword_search.py BLEU --context 150 --per-doc 2 --csv matches.csv

    from keyword_search import search   # search(["405B", "V3"], mode="any") -> DataFrame, one row per chunk

Searches chunking_output/chunks.csv, so references, appendix and table of contents are not searched.
"""
import argparse
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "chunking_output"


def keyword_pattern(keyword):
    # whole-term match: not glued to letters/digits on either side
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(keyword) + r"(?![A-Za-z0-9])", re.I)


STOP = {"what", "which", "who", "how", "why", "when", "where", "does", "did", "the", "and", "for", "are", "was",
        "were", "both", "each", "their", "this", "that", "with", "from", "used", "use", "many", "much"}


def question_keywords(question):
    """Distinctive terms: contain a digit or inner capital, or are capitalised mid-sentence (405B, DeepSeek-V3, BLEU)."""
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9.\-]*[A-Za-z0-9]|[A-Za-z0-9]", question)
    keep = [w for i, w in enumerate(words) if len(w) > 1 and w.lower() not in STOP and (
        any(c.isdigit() for c in w) or any(c.isupper() for c in w[1:]) or (i > 0 and w[0].isupper()))]
    return list(dict.fromkeys(keep)) or [w for w in words if len(w) > 3 and w.lower() not in STOP]


def search(keywords, mode="any"):
    """Chunks containing the keywords (mode "any" or "all"), best first.

    Columns: doc_id, title, section_title, chunk_id, found (keywords present), n_found, n_hits, text.
    """
    chunks = pd.read_csv(OUT / "chunks.csv", usecols=["chunk_id", "section_id", "doc_id", "chunk_text"])
    titles = pd.read_csv(OUT / "documents.csv").set_index("doc_id").title
    sections = pd.read_csv(OUT / "sections.csv").set_index("section_id").section_title
    patterns = {kw: keyword_pattern(kw) for kw in keywords}

    rows = []
    for c in chunks.itertuples():
        counts = {kw: len(p.findall(c.chunk_text)) for kw, p in patterns.items()}
        found = [kw for kw, n in counts.items() if n]
        if not found or (mode == "all" and len(found) < len(keywords)):
            continue
        rows.append({"doc_id": c.doc_id, "title": titles[c.doc_id], "section_title": sections[c.section_id],
                     "chunk_id": c.chunk_id, "found": found, "n_found": len(found),
                     "n_hits": sum(counts.values()), "text": c.chunk_text})
    cols = ["doc_id", "title", "section_title", "chunk_id", "found", "n_found", "n_hits", "text"]
    df = pd.DataFrame(rows, columns=cols)
    return df.sort_values(["n_found", "n_hits"], ascending=False, kind="stable").reset_index(drop=True)


def snippet(text, keywords, context):
    """Text around the first keyword hit, with the hits wrapped in [[ ]]."""
    spans = [m.span() for kw in keywords if (m := keyword_pattern(kw).search(text))]
    start = max(min(s for s, _ in spans) - context, 0)
    end = min(max(e for _, e in spans) + context, len(text))
    out = text[start:end]
    for kw in keywords:
        out = keyword_pattern(kw).sub(lambda m: f"[[{m.group(0)}]]", out)
    return ("..." if start else "") + out + ("..." if end < len(text) else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("keywords", nargs="+")
    ap.add_argument("--all", action="store_true", help="chunk must contain every keyword (default: any)")
    ap.add_argument("--context", type=int, default=120, help="characters shown around the hits")
    ap.add_argument("--per-doc", type=int, default=3, help="chunks shown per paper")
    ap.add_argument("--docs", type=int, default=10, help="papers shown")
    ap.add_argument("--csv", help="also save every matching chunk to this CSV (inside retrieval_output/)")
    args = ap.parse_args()

    df = search(args.keywords, "all" if args.all else "any")
    if df.empty:
        print("No chunk matches.")
        return

    # papers ranked by how many different keywords they contain, then by total hits
    per_doc = (df.groupby(["doc_id", "title"]).agg(chunks=("chunk_id", "size"), hits=("n_hits", "sum"),
               keywords=("found", lambda f: len({k for ks in f for k in ks})))
               .reset_index().sort_values(["keywords", "hits"], ascending=False))
    print(f"{len(df)} matching chunks in {len(per_doc)} papers\n")
    print(per_doc.head(args.docs).to_string(index=False), "\n")

    for d in per_doc.head(args.docs).itertuples():
        print(f"== {d.title} ({d.doc_id})")
        for c in df[df.doc_id == d.doc_id].head(args.per_doc).itertuples():
            print(f"  [{c.chunk_id}] {c.section_title} | found: {', '.join(c.found)}")
            print(f"     {snippet(c.text, c.found, args.context)}")

    if args.csv:
        path = HERE / "retrieval_output" / args.csv
        path.parent.mkdir(exist_ok=True)
        df.assign(found=df.found.map(", ".join)).to_csv(path, index=False)
        print(f"\nWrote {len(df)} chunks to {path}")


if __name__ == "__main__":
    main()
