import argparse
import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from rag import config
from rag.loaders import read_docs

MANUAL_FILE = config.ROOT / "aliases.txt"  
FIRST_PAGES = 2  
STOP = {"of", "and", "the", "for", "in", "on", "to", "a", "an", "with"}
LONG_FORM = re.compile(r"((?:[A-Za-z][\w'-]*[ \t]+){1,7})\(\s*([A-Za-z][A-Za-z0-9-]{1,9})\s*\)")

PER_PAPER_K = 3 

SYNTH_PROMPT = ChatPromptTemplate.from_template(
    """You answer a question that involves several research papers, using ONLY the evidence below.
The evidence is grouped by paper. Cover EVERY paper listed. If a paper's evidence does not
contain what is needed, say so for that paper instead of guessing.
After each claim, cite the source like [file p.page].

{evidence}

Question: {question}

Answer:"""
)


def norm(text: str) -> str:
    """lower case, every punctuation mark -> space: 'Chain-of-Thought' == 'chain of thought', 'ReAct?' == 'react'."""
    return re.sub(r"[\W_]+", " ", text.lower()).strip()


def is_abbreviation(abbr: str, words: list) -> bool:
    """'cot' fits 'chain of thought', 'lora' fits 'low rank adaptation': same first letter, letters in order."""
    a, letters = norm(abbr).replace(" ", ""), "".join(words)
    if len(a) < 2 or not words or a[0] != words[0][0]:
        return False
    it = iter(letters)
    return all(ch in it for ch in a)  


def long_forms(text: str) -> list:
    """[(abbreviation, 'long form')] for every  'long form (ABBR)'  in the text."""
    found = []
    for m in LONG_FORM.finditer(text):
        words = norm(m.group(1)).split()
        for n in range(2, min(len(words), 6) + 1):  
            if is_abbreviation(m.group(2), words[-n:]):
                found.append((norm(m.group(2)), " ".join(words[-n:])))
                break
    return found


def title_variants(src: str) -> set:
    """Names from the file name: the whole title and its first 3+ words ('chain of thought prompting' -> 'chain of thought')."""
    title = norm(re.sub(r"^\d+[_-]?", "", re.sub(r"\.[A-Za-z0-9]+$", "", src)))
    words = title.split()
    return {title} | {" ".join(words[:n]) for n in range(3, len(words))}


def manual_aliases() -> dict:
    out = {}
    if MANUAL_FILE.exists():
        for line in MANUAL_FILE.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                alias, piece = line.split("=", 1)
                out[norm(alias)] = norm(piece)
    return out


def build_catalog(docs) -> dict:
    sources = sorted({str(d.metadata["source"]) for d in docs})
    owners = {}  
    for src in sources:  
        for name in title_variants(src):
            owners.setdefault(name, set()).add(src)
    pairs = []
    for d in docs:
        page = d.metadata.get("page")
        if isinstance(page, int) and page <= FIRST_PAGES:
            pairs += [(abbr, long, str(d.metadata["source"])) for abbr, long in long_forms(d.page_content)]
    title_owned = dict(owners)  
    for abbr, long, _ in pairs:  
        if long in title_owned and len(title_owned[long]) == 1:
            owners.setdefault(abbr, set()).update(title_owned[long])
    for abbr, long, src in pairs:  
        for name in (abbr, long):
            if name not in title_owned and not (name == abbr and long in title_owned and len(title_owned[long]) == 1):
                owners.setdefault(name, set()).add(src)
    for alias, piece in manual_aliases().items():  
        found = {s for s in sources if piece in norm(s)}
        if found:
            owners[alias] = found
    catalog = {src: [] for src in sources}
    for name, files in owners.items():
        if len(name) >= 3 and len(files) == 1:
            catalog[next(iter(files))].append(name)
    return {src: sorted(names) for src, names in catalog.items()}


def find_papers(question: str, catalog: dict) -> list:
    """Papers named in the question, in the order they are mentioned."""
    q = " " + norm(question) + " "
    found = []
    for src, names in catalog.items():
        pos = [q.find(f" {n} ") for n in names if f" {n} " in q]
        if pos:
            found.append((min(pos), src, next(n for n in names if f" {n} " in q)))
    return [(src, name) for _, src, name in sorted(found)]


def plan(question: str, catalog: dict) -> list:
    """The plan = a list of {question, source}. Empty list = not a multi-paper question."""
    papers = find_papers(question, catalog)
    if len(papers) < 2:
        return []
    return [
        {"source": src, "question": f"How does {name} work? Main idea, method and key results. (Context: {question})"}
        for src, name in papers
    ]


def show_plan(steps, question):
    print(f"\nQuestion: {question}")
    if not steps:
        print("Plan: fewer than 2 papers named -> fall back to plain Phase 2 search (phase2_ask.py).")
        return
    print(f"Plan: {len(steps)} sub-questions, one search per paper")
    for i, s in enumerate(steps, start=1):
        print(f"  {i}. [{s['source']}]  {s['question']}")


def retrieve(steps, store, k):
    """One filtered search per paper. Returns [(step, [(doc, dist), ...]), ...]."""
    return [(s, store.similarity_search_with_score(s["question"], k=k, filter={"source": s["source"]})) for s in steps]


def format_evidence(evidence) -> str:
    blocks = []
    for step, hits in evidence:
        body = "\n\n".join(f"[{d.metadata['source']} p.{d.metadata['page']}]\n{d.page_content}" for d, _ in hits)
        blocks.append(f"=== Paper: {step['source']} ===\n{body or '(no text found for this paper)'}")
    return "\n\n".join(blocks)


def show_evidence(evidence):
    for step, hits in evidence:
        print(f"\n--- {step['source']} ({len(hits)} chunks) ---")
        for rank, (d, dist) in enumerate(hits, start=1):
            text = " ".join(d.page_content.split())
            print(f"  #{rank} dist={dist:.3f} p.{d.metadata['page']}: {text[:250]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--show-aliases", action="store_true", help="print the names found for every paper and stop")
    ap.add_argument("--plan-only", action="store_true", help="show the plan and stop (needs only docs.jsonl)")
    ap.add_argument("--retrieve-only", action="store_true", help="plan + retrieve, no LLM")
    ap.add_argument("--k", type=int, default=PER_PAPER_K, help="chunks per paper")
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()

    catalog = build_catalog(read_docs(config.DOCS_FILE))
    if args.show_aliases:
        for src, names in catalog.items():
            print(f"{src:50} {', '.join(names)}")
        return
    if not args.question:
        raise SystemExit("Give a question, or use --show-aliases.")
    steps = plan(args.question, catalog)
    show_plan(steps, args.question)
    if args.plan_only:
        return

    from rag.models import get_embeddings, get_llm, get_vectorstore  # only needed from here on

    persist_dir = config.chroma_dir(args.fake)
    if not persist_dir.exists():
        raise SystemExit("No index found. Run phase2_index.py first.")
    store = get_vectorstore(get_embeddings(args.fake), persist_dir)

    if not steps:  # fallback = Phase 2
        from phase2_ask import ask

        ask(args.question, store, None if args.retrieve_only else get_llm(args.fake), config.TOP_K, args.retrieve_only)
        return

    evidence = retrieve(steps, store, args.k)
    show_evidence(evidence)
    if args.retrieve_only:
        return
    chain = SYNTH_PROMPT | get_llm(args.fake) | StrOutputParser()
    print("\n--- Answer ---")
    print(chain.invoke({"evidence": format_evidence(evidence), "question": args.question}))
    cited = sorted({f"{d.metadata['source']} p.{d.metadata['page']}" for _, hits in evidence for d, _ in hits})
    print("\nSources:", ", ".join(cited))


if __name__ == "__main__":
    main()