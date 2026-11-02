"""Section chunking of the research-paper PDFs; writes three CSVs to members/team_huy/chunking_output/.

Three linked tables (ids link them): documents -> sections -> chunks.
Every section keeps its own title; a section longer than MAX_CHUNK_TOKENS is split into several chunks.

documents.csv  doc_id, title, document_summary, pages, section_ids, file, year
sections.csv   section_id, doc_id, section_title, section_order, section_summary, chunk_ids
chunks.csv     chunk_id, section_id, doc_id, chunk_order, chunk_text, chunk_summary, chunk_embedding,
               est_tokens, embed_text
chunk_embedding is a JSON list of floats from EmbeddingGemma (embed_text is the exact string embedded).
Summary columns are left empty here. For queries, EmbeddingGemma expects "task: search result | query: <question>".

Run from anywhere:  python3 members/team_huy/chunking.py
"""
import json
import re
import urllib.request
from pathlib import Path

import pandas as pd
import pymupdf

ROOT = Path(__file__).resolve().parents[2]
PDF_DIR = ROOT / "tmp" / "RAG-data" / "Research Paper"
CHARS_PER_TOKEN = 4  # rough estimate for English text (no tokenizer installed)
OLLAMA_URL = "http://localhost:11434/api/embed"  # Ollama must be running: ollama serve
EMBED_MODEL = "embeddinggemma"  # ollama pull embeddinggemma
EMBED_BATCH = 32
OUT_DIR = Path(__file__).resolve().parent / "chunking_output"
ROLES = ["abstract", "introduction", "conclusion", "references"]

NUMBERED = re.compile(r"^(?:\d{1,2}|[A-Z])\.?\s+[A-Z][^.!?]{2,70}$")  # "3 Model Architecture", "A Details"
UNNUMBERED = re.compile(
    r"^(?:abstract|introduction|acknowledg(?:e)?ments?(?:\s+and\s+\w+)?|references|bibliography|appendix(?:\s+[A-Z](?::.*)?)?"
    r"|supplementary.*|conclusions?"
    r"|(?:discussion|limitations?)(?:\s+(?:and|&)\s+[a-z ]{3,30})?|future (?:work|directions)"
    r"|ethic(?:s|al)(?:\s+(?:and\s+\w+\s+)?(?:statement|considerations?))?|broader impacts?|impact statement"
    r"|reproducibility statement|author contributions?|contributions?|(?:core )?contributors(?:\s+and\s+\w+)?|funding)$",
    re.I,
)
BARE_NUM = re.compile(r"^(?:\d{1,2}|[A-Z])\.?$")  # "1", "3.", "A": number alone on its own line
NOT_HEADING = re.compile(r"^(?:table|figure|fig\.|algorithm|listing|example)\b", re.I)


def role_of(heading):
    """Map a heading to abstract / introduction / conclusion / references / ... / body."""
    h = re.sub(r"^(?:\d{1,2}|[A-Z])\.?\s+", "", heading).strip().lower()
    if h.startswith("abstract"):
        return "abstract"
    if "introduction" in h:
        return "introduction"
    if h.startswith(("reference", "bibliography")):
        return "references"
    if h.startswith("acknowledg"):
        return "acknowledgements"
    if "conclusion" in h or h.startswith("concluding"):
        return "conclusion"
    if h.startswith(("appendix", "supplementary")):
        return "appendix"
    if h.startswith(("discussion", "limitation", "future work", "future direction")):
        return "discussion"
    if h.startswith(("ethic", "broader impact", "impact statement", "reproducibility", "author contribution",
                     "contribution", "core contributors", "contributors", "funding")):
        return "back_matter"  # only here so that it ends the section before it
    return "body"


def keep_in_sequence(heads):
    """Drop numbered 'headings' that are really chart labels, affiliations or list items.

    Top-level numbers must increase (1, 2, 3 ...); appendix letters (A, B, ...) are only
    accepted after References/Appendix and must also increase. Unnumbered headings always stay.
    """
    kept, last_num, last_letter, in_back = [], 0, 0, False
    # numbering starts at the numbered Introduction; earlier "1 University of ..." lines are affiliations
    first_intro = next((i for i, h in enumerate(heads)
                        if re.match(r"^\d{1,2}\.?\s+introduction", h["heading"], re.I)), 0)
    for idx, h in enumerate(heads):
        if idx < first_intro and re.match(r"^\d", h["heading"]):
            continue
        if h["heading"].endswith((",", ":")) or re.search(
                r"^\S+\s+(?:thousand|million|billion|trillion)\b", h["heading"], re.I):
            continue
        m = re.match(r"^(\d{1,2}|[A-Z])\.?\s", h["heading"])
        role = role_of(h["heading"])
        if role in ("conclusion", "discussion"):
            pass  # "5. Discussion & Conclusion" is never a chart label, even if "5" was already used
        elif role in ("references", "appendix"):
            in_back = True  # "9 References" etc. are accepted whatever their number
        elif m and m.group(1).isdigit():
            n = int(m.group(1))
            if not last_num < n <= last_num + 2:
                continue
            last_num = n
        elif m:
            letter = ord(m.group(1)) - 64
            if not in_back or not last_letter < letter <= last_letter + 2:
                continue
            last_letter = letter
        if kept and kept[-1]["heading"] == h["heading"]:
            continue
        kept.append(h)
    return kept


def find_headings(pages):
    """Headings as dicts: heading, offset into the full text.

    PyMuPDF often emits the section number and the title as two lines ("1", "Introduction"),
    so a bare number line is glued to the next non-empty line before matching.
    """
    heads, tocs, offset = [], [], 0
    for text in pages:
        lines = text.split("\n")
        consumed = -1  # index of a title line already merged into the previous number line
        for i, line in enumerate(lines):
            s = line.strip()
            start = offset
            offset += len(line) + 1
            if i == consumed:
                continue
            if BARE_NUM.match(s):
                j = next((k for k in range(i + 1, min(i + 3, len(lines))) if lines[k].strip()), None)
                nxt = lines[j].strip() if j is not None else ""
                if not nxt or NOT_HEADING.match(nxt) or len(nxt) > 70 or len(nxt.split()) > 8:
                    continue
                s = f"{s.rstrip('.')} {nxt}"
                consumed = j
            if 3 <= len(s) <= 80 and not NOT_HEADING.match(s) and (UNNUMBERED.match(s) or NUMBERED.match(s)):
                after = next((l.strip() for l in lines[max(i, consumed) + 1:max(i, consumed) + 3] if l.strip()), "")
                if re.fullmatch(r"\d{1,3}", after):
                    # table-of-contents entry (title followed by its page number): it still ends the
                    # section before it, but is kept out of the numbering logic and never extracted
                    if role_of(s) != "body":  # chart labels like "9 Device 1" must not cut sections
                        tocs.append({"heading": s, "offset": start, "toc": True})
                else:
                    heads.append({"heading": s, "offset": start})
    return sorted(keep_in_sequence(heads) + tocs, key=lambda h: h["offset"])


def extract_sections(pages, roles=ROLES):
    """Return {role: text} for the wanted roles ('' if the paper has no such heading).

    A role can span several detected sections, so their texts are joined. When a paper has no
    "Abstract" heading (e.g. Chinchilla, Llama 3), the front matter is used as the abstract.
    When it has no "Conclusion" heading, its closing "Discussion" / "Limitations" / "Future Work"
    section is used as the conclusion.
    """
    text = "".join(line + "\n" for p in pages for line in p.split("\n"))  # offsets refer to this text
    heads = find_headings(pages)
    # only the last "References" heading counts (earlier ones are a table of contents or mentions)
    ref_idx = [i for i, h in enumerate(heads) if role_of(h["heading"]) == "references" and not h.get("toc")]
    heads = [h for i, h in enumerate(heads) if i not in ref_idx[:-1]]

    bounds = [{"heading": "(front matter)", "offset": 0}] + heads
    parts = {}  # role -> list of section texts
    for i, h in enumerate(bounds):
        end = bounds[i + 1]["offset"] if i + 1 < len(bounds) else len(text)
        role = "front_matter" if i == 0 else "toc" if h.get("toc") else role_of(h["heading"])
        if role not in ("references", "toc") and any(
                role_of(b["heading"]) == "references" and not b.get("toc") and b["offset"] < h["offset"] for b in heads):
            role = "appendix"
        parts.setdefault(role, []).append(text[h["offset"]:end].strip())

    out = {r: "\n".join(parts.get(r, [])) for r in roles}
    if "conclusion" in out and not out["conclusion"]:
        out["conclusion"] = "\n".join(parts.get("discussion", [])[-1:])  # last one: earlier hits are the table of contents
    if "abstract" in out and not out["abstract"]:
        out["abstract"] = "\n".join(parts["front_matter"])
    return out


import math

# Chunks are made per section; every section keeps its own title (no generic "other" bucket).
# A section longer than MAX_CHUNK_TOKENS is split into several chunks of similar size.
# References, appendix, table of contents, acknowledgements/ethics/contributors are not chunked.
MAX_CHUNK_TOKENS = 400
OVERLAP_TOKENS = 50  # last sentences of a chunk repeated at the start of the next one
KEEP_ROLES = {"abstract", "introduction", "conclusion", "body", "discussion"}  # roles that get chunked


def drop_heading(text, heading):
    """Remove the heading line(s) from the start of a section's text."""
    key = lambda s: re.sub(r"\W", "", s.lower())
    lines, acc = text.split("\n"), ""
    for i, line in enumerate(lines):
        acc += key(line)
        if acc == key(heading):
            return "\n".join(lines[i + 1:]).strip()
        if len(acc) > len(key(heading)):
            break
    return text


def split_sections(pages):
    """All sections of a paper as dicts: heading, role, text, keep (False for references, appendix, ...)."""
    text = "".join(line + "\n" for p in pages for line in p.split("\n"))
    heads = find_headings(pages)
    ref_idx = [i for i, h in enumerate(heads) if role_of(h["heading"]) == "references" and not h.get("toc")]
    heads = [h for i, h in enumerate(heads) if i not in ref_idx[:-1]]
    ref_offset = next((h["offset"] for h in heads if role_of(h["heading"]) == "references" and not h.get("toc")), None)

    bounds = [{"heading": "(front matter)", "offset": 0}] + heads
    secs = []
    for i, h in enumerate(bounds):
        end = bounds[i + 1]["offset"] if i + 1 < len(bounds) else len(text)
        role = "front_matter" if i == 0 else "toc" if h.get("toc") else role_of(h["heading"])
        if role not in ("references", "toc") and ref_offset is not None and h["offset"] > ref_offset:
            role = "appendix"
        body = text[h["offset"]:end].strip()
        secs.append({"heading": h["heading"], "role": role,
                     "text": body if i == 0 else drop_heading(body, h["heading"])})

    # same fallbacks as extract_sections: front matter as abstract, closing discussion as conclusion
    if not any(s["role"] == "abstract" for s in secs):
        secs[0].update(role="abstract", heading="Abstract")
    if not any(s["role"] == "conclusion" for s in secs):
        last = [s for s in secs if s["role"] == "discussion"][-1:]
        for s in last:
            s["role"] = "conclusion"
    for s in secs:
        s["keep"] = s["role"] in KEEP_ROLES
    return secs


def split_into_chunks(text):
    """Split one section into balanced chunks of at most ~MAX_CHUNK_TOKENS, cut at sentence ends."""
    text = re.sub(r"-\n(?=[a-z])", "", text)        # words hyphenated across lines
    text = re.sub(r"\s+", " ", text).strip()
    limit = MAX_CHUNK_TOKENS * CHARS_PER_TOKEN
    sents = []
    for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"“])", text):
        while len(s) > limit:                         # no sentence end (tables, equations): cut at a space
            cut = s.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            sents.append(s[:cut])
            s = s[cut:].lstrip()
        if s:
            sents.append(s)

    target = len(text) / max(1, math.ceil(len(text) / limit))  # balanced size, so no tiny last chunk
    chunks, cur, fresh = [], [], 0
    def flush():
        nonlocal cur, fresh
        chunks.append(" ".join(cur))
        carry, n = [], 0                              # overlap: carry the last sentences forward
        for t in reversed(cur):
            if n + len(t) > OVERLAP_TOKENS * CHARS_PER_TOKEN:
                break
            carry.insert(0, t)
            n += len(t)
        cur, fresh = carry, 0

    for s in sents:
        if fresh and fresh + len(s) > limit:          # this sentence would make the chunk too big
            flush()
        cur.append(s)
        fresh += len(s)
        if fresh >= target:
            flush()
    if fresh:                                         # leftover with new text (not only overlap)
        chunks.append(" ".join(cur))
    return chunks


def embed(texts):
    """Embed texts with EmbeddingGemma through Ollama; returns one vector (list of floats) per text."""
    vectors = []
    for i in range(0, len(texts), EMBED_BATCH):
        body = json.dumps({"model": EMBED_MODEL, "input": texts[i:i + EMBED_BATCH]}).encode()
        req = urllib.request.Request(OLLAMA_URL, body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            vectors += json.load(resp)["embeddings"]
        print(f"  embedded {len(vectors)}/{len(texts)}", end="\r")
    print()
    return vectors


def main():
    index = pd.read_csv(PDF_DIR / "index.csv")  # Title / Year / Phase per paper
    docs, sections, chunks = [], [], []
    for pdf in sorted(PDF_DIR.glob("*.pdf")):
        with pymupdf.open(pdf) as doc:
            pages = [page.get_text() for page in doc]
        meta = index[index["File"] == pdf.name].iloc[0]
        doc_id = f"doc{pdf.stem[:2]}"

        section_ids = []
        for sec in split_sections(pages):
            if not sec["keep"]:
                continue
            texts = split_into_chunks(sec["text"])
            if not texts:  # empty section, e.g. a duplicate "Introduction" heading right after "1 Introduction"
                continue
            section_id = f"{doc_id}-s{len(section_ids) + 1:02d}"
            section_ids.append(section_id)
            chunk_ids = []
            for k, text in enumerate(texts, 1):
                chunk_ids.append(f"{section_id}-c{k:02d}")
                chunks.append({
                    "chunk_id": chunk_ids[-1],
                    "section_id": section_id,
                    "doc_id": doc_id,
                    "chunk_order": k,
                    "chunk_text": text,
                    "chunk_summary": "",     # to be filled by an LLM
                    "chunk_embedding": "",   # filled below with EmbeddingGemma
                    "est_tokens": len(text) // CHARS_PER_TOKEN,
                    # EmbeddingGemma's document format: "title: ... | text: ..."; the title carries paper + section
                    "embed_text": f"title: {meta['Title']} › {sec['heading']} | text: {text}",
                })
            sections.append({
                "section_id": section_id,
                "doc_id": doc_id,
                "section_title": sec["heading"],
                "section_order": len(section_ids),
                "section_summary": "",       # to be filled by an LLM
                "chunk_ids": json.dumps(chunk_ids),
            })
        docs.append({
            "doc_id": doc_id,
            "title": meta["Title"],
            "document_summary": "",          # to be filled by an LLM
            "pages": len(pages),
            "section_ids": json.dumps(section_ids),
            "file": pdf.name,
            "year": meta["Year"],
        })

    OUT_DIR.mkdir(exist_ok=True)
    df_docs, df_secs, df_chunks = pd.DataFrame(docs), pd.DataFrame(sections), pd.DataFrame(chunks)
    print(f"Embedding {len(df_chunks):,} chunks with {EMBED_MODEL} ...")
    vectors = embed(df_chunks.embed_text.tolist())
    df_chunks["chunk_embedding"] = [json.dumps([round(x, 6) for x in v]) for v in vectors]
    df_docs.to_csv(OUT_DIR / "documents.csv", index=False)
    df_secs.to_csv(OUT_DIR / "sections.csv", index=False)
    df_chunks.to_csv(OUT_DIR / "chunks.csv", index=False)
    print(f"{len(df_docs)} documents, {len(df_secs)} sections, {len(df_chunks):,} chunks "
          f"(median {df_chunks.est_tokens.median():.0f} tokens, max {df_chunks.est_tokens.max()})")
    print(f"Wrote documents.csv, sections.csv, chunks.csv to {OUT_DIR}")


if __name__ == "__main__":
    main()
