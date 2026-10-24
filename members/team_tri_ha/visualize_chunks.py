"""See how each chunking strategy cuts the SAME text. Writes one HTML report you open in a browser.

Run:
    python visualize_chunks.py                                  # longest page in docs.jsonl
    python visualize_chunks.py --file LoRA --page 3             # a specific page of a specific file
    python visualize_chunks.py --file LoRA --sizes 500 1000     # also compare chunk sizes
    python visualize_chunks.py --corpus                         # + length histogram over ALL documents
    python visualize_chunks.py --semantic                       # + semantic chunking (slow, needs Ollama)

Then open data/chunk_report.html (WSL: explorer.exe data/chunk_report.html).

How to read the page view:
    each colour block = one chunk (the small #n marks where chunk n starts)
    red stripes       = text that appears in TWO chunks (the overlap)
    grey dotted       = text that is in NO chunk (cut away, usually only whitespace)
"""
import argparse
import html
from pathlib import Path

from rag import config
from rag.chunkers import chunk_documents
from rag.loaders import read_docs

HUES = [210, 30, 120, 290, 50, 180]  # colours of the chunks, repeated in turn


def stats(chunks):
    """Same numbers as compare_chunking.py (copied so this script needs no Ollama/Chroma imports)."""
    lens = [len(c.page_content) for c in chunks]
    clean = sum(1 for c in chunks if c.page_content.rstrip().endswith((".", "?", "!", ":", "\n")))
    n = max(len(lens), 1)
    return {
        "chunks": len(chunks),
        "avg": round(sum(lens) / len(lens)) if lens else 0,
        "min": min(lens, default=0),
        "max": max(lens, default=0),
        "tiny%": round(100 * sum(1 for x in lens if x < 100) / n),
        "clean_end%": round(100 * clean / n),
    }


def locate(text, chunks):
    """Find where each chunk sits in the original text. Returns a (start, end) or None per chunk."""
    spans, search_from = [], 0
    for c in chunks:
        piece = c.page_content
        i = text.find(piece, search_from)
        if i < 0:  # some strategies (markdown) re-join lines: fall back to the first 40 characters
            i = text.find(piece[:40], search_from)
        if i < 0:
            spans.append(None)
            continue
        spans.append((i, min(i + len(piece), len(text))))
        search_from = i + 1  # next chunk may start inside this one (overlap), never before it
    return spans


def paint(text, spans):
    """The page text as HTML: one <span> per stretch that has the same set of chunks covering it."""
    cuts = {0, len(text)}
    for s in spans:
        if s:
            cuts.update(s)
    cuts = sorted(cuts)
    out = []
    starts = {s[0]: k for k, s in enumerate(spans, start=1) if s}  # chunk number that starts at a position
    for a, b in zip(cuts, cuts[1:]):
        covering = [k for k, s in enumerate(spans, start=1) if s and s[0] <= a < s[1]]
        label = f'<sup class="tag">#{starts[a]}</sup>' if a in starts else ""
        body = html.escape(text[a:b])
        if not covering:
            out.append(f'{label}<span class="gap" title="not in any chunk">{body}</span>')
        elif len(covering) == 1:
            k = covering[0]
            out.append(f'{label}<span class="c{(k - 1) % len(HUES)}" title="chunk #{k}">{body}</span>')
        else:
            names = ", ".join(f"#{k}" for k in covering)
            out.append(f'{label}<span class="ov" title="overlap of chunks {names}">{body}</span>')
    return "".join(out)


def bars(lengths, size):
    """One bar per chunk, height = chunk length. The dashed line is the size limit you asked for."""
    top = max(max(lengths, default=1), size)
    cols = "".join(f'<i style="height:{100 * n / top:.1f}%" title="{n} chars"></i>' for n in lengths)
    return f'<div class="bars"><div class="limit" style="bottom:{100 * size / top:.1f}%"></div>{cols}</div>'


def histogram(lengths, size, bins=12):
    top = max(max(lengths, default=1), size)
    width = top / bins
    counts = [0] * bins
    for n in lengths:
        counts[min(int(n / width), bins - 1)] += 1
    peak = max(counts) or 1
    cols = "".join(
        f'<div class="hcol"><i style="height:{100 * c / peak:.1f}%" title="{c} chunks"></i>'
        f'<span>{int(k * width)}</span></div>'
        for k, c in enumerate(counts)
    )
    return f'<div class="hist">{cols}</div>'


def pick_document(docs, file_part, page):
    pool = [d for d in docs if not file_part or file_part.lower() in str(d.metadata.get("source", "")).lower()]
    if page is not None:
        pool = [d for d in pool if str(d.metadata.get("page")) == str(page)]
    if not pool:
        raise SystemExit("No page matches --file/--page. Check the names in the table printed by phase1_ingest.py.")
    return pool[0] if file_part or page is not None else max(pool, key=lambda d: len(d.page_content))


CSS = """
body{font:15px/1.5 system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#222;background:#fff}
@media(prefers-color-scheme:dark){body{color:#e6e6e6;background:#161616}}
h1{font-size:22px} h2{font-size:17px;margin:28px 0 6px} .meta{opacity:.7;font-size:13px}
table{border-collapse:collapse;font-size:13px;margin:8px 0} td,th{padding:4px 10px;border-bottom:1px solid #8884;text-align:right}
td:first-child,th:first-child,td:nth-child(2),th:nth-child(2){text-align:left}
.text{white-space:pre-wrap;font:13px/1.7 ui-monospace,Menlo,Consolas,monospace;border:1px solid #8884;border-radius:8px;padding:12px;margin:8px 0}
.tag{font:700 10px sans-serif;color:#e11;margin:0 2px;vertical-align:super}
.ov{background:repeating-linear-gradient(45deg,rgba(220,38,38,.38) 0 6px,rgba(220,38,38,.14) 6px 12px)}
.gap{background:hsla(0,0%,50%,.18);border-bottom:1px dotted #888}
.legend span{display:inline-block;padding:1px 8px;margin-right:6px;border-radius:4px;font-size:12px}
.bars{position:relative;display:flex;align-items:flex-end;gap:2px;height:70px;margin:6px 0 2px;border-bottom:1px solid #8886}
.bars i{display:block;flex:1;min-width:2px;background:hsla(210,70%,55%,.7);border-radius:2px 2px 0 0}
.hcol i{display:block;flex:none;background:hsla(210,70%,55%,.7);border-radius:2px 2px 0 0}
.limit{position:absolute;left:0;right:0;border-top:1px dashed #e11;pointer-events:none}
.hist{display:flex;align-items:flex-end;gap:3px;height:90px;margin:6px 0 18px}
.hcol{flex:1;height:100%;display:flex;flex-direction:column;justify-content:flex-end;align-items:stretch;font-size:10px;text-align:center}
.hcol span{opacity:.65;margin-top:2px}
"""
for _i, _h in enumerate(HUES):
    CSS += f".c{_i}{{background:hsla({_h},70%,60%,.30)}}\n"


def table(rows):
    cols = ["strategy", "size", "chunks", "avg", "min", "max", "tiny%", "clean_end%"]
    head = "".join(f"<th>{c}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td>{r[c]}</td>" for c in cols) + "</tr>" for r in rows)
    return f"<table><tr>{head}</tr>{body}</table>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="part of the file name, e.g. LoRA")
    ap.add_argument("--page", help="page / sheet / slide to show")
    ap.add_argument("--sizes", type=int, nargs="+", default=[config.CHUNK_SIZE])
    ap.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP)
    ap.add_argument("--semantic", action="store_true", help="add semantic chunking (slow, needs Ollama)")
    ap.add_argument("--corpus", action="store_true", help="also show chunk-length histograms over all documents")
    ap.add_argument("--docs", default=str(config.DOCS_FILE), help="docs.jsonl from phase 1")
    ap.add_argument("--out", default=str(config.ROOT / "data" / "chunk_report.html"))
    ap.add_argument("--fake", action="store_true", help="fake embeddings for --semantic (code check only)")
    args = ap.parse_args()

    docs = read_docs(Path(args.docs))
    doc = pick_document(docs, args.file, args.page)
    text = doc.page_content
    src, page = doc.metadata.get("source"), doc.metadata.get("page")
    print(f"Showing: {src}  page {page}  ({len(text)} characters)")

    embeddings = None
    if args.semantic:
        from rag.models import get_embeddings  # imported here: only this option needs Ollama

        embeddings = get_embeddings(args.fake)

    strategies = ["fixed", "recursive", "markdown"] + (["semantic"] if args.semantic else [])
    runs = [(s, z) for s in strategies for z in ([args.sizes[0]] if s == "semantic" else args.sizes)]

    summary, sections = [], []
    for strategy, size in runs:
        chunks = chunk_documents([doc], strategy, size, args.overlap, embeddings)
        spans = locate(text, chunks)
        st = stats(chunks)
        summary.append({"strategy": strategy, "size": "auto" if strategy == "semantic" else size, **st})
        lost = sum(1 for s in spans if s is None)
        note = f" ({lost} chunk(s) could not be placed on the text)" if lost else ""
        title = f"{strategy}, size {'auto' if strategy == 'semantic' else size}, overlap {args.overlap}"
        sections.append(
            f"<h2>{html.escape(title)}</h2>"
            f'<div class="meta">{st["chunks"]} chunks, avg {st["avg"]} chars, min {st["min"]}, max {st["max"]}{note}</div>'
            f'<div class="text">{paint(text, spans)}</div>'
            f'<div class="meta">length of each chunk (dashed line = size limit)</div>'
            f"{bars([len(c.page_content) for c in chunks], size)}"
        )

    corpus_html = ""
    if args.corpus:
        parts, corpus_rows = [], []
        for strategy, size in runs:
            if strategy == "semantic":
                continue  # would embed every sentence of every document
            chunks = chunk_documents(docs, strategy, size, args.overlap)
            corpus_rows.append({"strategy": strategy, "size": size, **stats(chunks)})
            parts.append(
                f"<h2>{strategy}, size {size}: all {len(docs)} documents</h2>"
                f'<div class="meta">x axis = chunk length in characters, bar height = how many chunks</div>'
                f"{histogram([len(c.page_content) for c in chunks], size)}"
            )
        corpus_html = "<h1>Whole corpus</h1>" + table(corpus_rows) + "".join(parts)

    legend = ('<div class="legend"><span class="c0">chunk</span><span class="c1">next chunk</span>'
              '<span class="ov">overlap (in 2 chunks)</span><span class="gap">in no chunk</span></div>')
    page_html = (
        f"<!doctype html><meta charset='utf-8'><title>Chunk report</title><style>{CSS}</style>"
        f"<h1>How chunking cuts one page</h1>"
        f'<div class="meta">{html.escape(str(src))}, page {html.escape(str(page))}, {len(text)} characters</div>'
        f"{legend}{table(summary)}{''.join(sections)}{corpus_html}"
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page_html, encoding="utf-8")
    print(f"Wrote {out}\nOpen it in a browser (WSL: explorer.exe \"{out}\")")


if __name__ == "__main__":
    main()