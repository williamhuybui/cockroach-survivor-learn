import argparse
import json
import math
import re
from collections import Counter

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from rag import config
from rag.loaders import read_docs

MANUAL_FILE = config.ROOT / "aliases.txt"  
CATALOG_TYPES = {"pdf"}  
FIRST_PAGES = 2  
STOP = {"of", "and", "the", "for", "in", "on", "to", "a", "an", "with", "is", "are", "do", "does", "how", "why",
        "what", "by", "from", "at", "as", "its", "it", "or"}

GENERIC = {"llm", "llms", "large language model", "large language models", "language model", "language models",
           "artificial intelligence", "ai", "neural network", "neural networks", "machine learning", "deep learning",
           "nlp", "natural language processing", "reinforcement learning", "transformer"}
LONG_FORM = re.compile(r"((?:[A-Za-z][\w'-]*[ \t]+){1,7})\(\s*([A-Za-z][A-Za-z0-9-]{1,9})\s*\)")

PER_PAPER_K = 3 
MIN_SHARE, MIN_COUNT = 0.5, 5 

PLANNER_PROMPT = ChatPromptTemplate.from_template(
    """You route questions to research papers. These are the papers (file name, then the start of the paper):

{catalog}

Question: {question}

List the file names of EVERY paper the question is about. A paper can be named directly, or described
(for example "the Transformer" is the paper that introduced the Transformer). Use ONLY file names from the list above.
Reply with a JSON list of file names and nothing else, for example ["01_A.pdf", "02_B.pdf"]."""
)

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
    """Names from the file name:
      - the whole title and its first 2+ words ('llama 3 herd of models' -> 'llama 3'); a start that ends in
        'of', 'is', 'in'... is skipped ('chain of' is not a name, 'chain of thought' is)
      - initials of the first 3-5 words ('chain of thought' -> 'cot', 'retrieval augmented generation' -> 'rag')
      - a distinctive first word such as LLaMA, ReAct or DeepSeek (capital inside the word, ALL CAPS, or a digit)"""
    raw = re.sub(r"^\d+[_-]?", "", re.sub(r"\.[A-Za-z0-9]+$", "", src))
    words = norm(raw).split()
    if words and words[0] == "the" and len(words) > 1:
        words = words[1:]
    names = {" ".join(words)}
    names |= {" ".join(words[:n]) for n in range(2, len(words)) if words[n - 1] not in STOP}
    first = re.split(r"[\W_]+", raw.strip("_- "))[0]
    if len(words) >= 3 and not first.isupper():  
        names.add("".join(w[0] for w in words[:3]))
    if len(first) >= 3 and (first.isupper() or re.search(r"[a-z][A-Z]|\d", first)):
        names.add(first.lower())
    return names


def related_to_title(abbr: str, long: str, src: str) -> bool:
    """Is 'long form (ABBR)' about THIS paper, or a generic term it merely uses ('large language model')?
    Yes if the abbreviation is a word of the title, or most words of the long form are in the title."""
    title = set(norm(re.sub(r"^\d+[_-]?", "", re.sub(r"\.[A-Za-z0-9]+$", "", src))).split())
    words = [w for w in long.split() if w not in STOP]
    return abbr in title or (bool(words) and sum(w in title for w in words) / len(words) >= 0.6)


def content_names(docs, sources) -> list:
    """[(word, file)] for title words that appear (almost) only inside their own paper.
    Reads the text of every paper, so it is based on content, not only on the file name."""
    candidates = {}
    for src in sources:
        for w in title_words(src):
            candidates.setdefault(w, set()).add(src)
    per_paper = {w: Counter() for w in candidates}
    for d in docs:
        src = str(d.metadata["source"])
        for w in norm(d.page_content).split():
            if w in per_paper:
                per_paper[w][src] += 1
    out = []
    for w, srcs in candidates.items():
        total = sum(per_paper[w].values())
        if total < MIN_COUNT:
            continue
        best, n = per_paper[w].most_common(1)[0]
        if best in srcs and n / total >= MIN_SHARE:
            out.append((w, best))
    return out


def title_words(src: str) -> list:
    """Single words of the file-name title that could be a name (long enough, not a plain function word)."""
    raw = re.sub(r"^\d+[_-]?", "", re.sub(r"\.[A-Za-z0-9]+$", "", src))
    return [w for w in norm(raw).split() if len(w) >= 5 and w not in STOP and w not in GENERIC]


def manual_aliases() -> dict:
    out = {}
    if MANUAL_FILE.exists():
        for line in MANUAL_FILE.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                alias, piece = line.split("=", 1)
                out[norm(alias)] = norm(piece)
    return out


def build_catalog(docs) -> dict:
    """{file name: [names this paper can be called]}, built from the papers themselves.

    Who owns a name?
      1. the file name decides: 'chain of thought' is a start of '11_Chain-of-Thought_Prompting.pdf'
      2. an abbreviation follows its long form: '(CoT)' defined as 'chain of thought' belongs to that same paper,
         even if other papers (like ReAct) also mention it
      3. any other name found on the first pages counts only if it is about the paper itself (not 'large language
         model') and exactly ONE paper uses it
      4. a word of the title counts as a name if the CONTENT of the corpus says it belongs to this paper
         ('chinchilla' is used almost only inside the Chinchilla paper; 'attention' is used everywhere, so it is not a name)
    """
    sources = sorted({str(d.metadata["source"]) for d in docs})
    owners = {}  # name -> set of files
    for src in sources:  # rule 1
        for name in title_variants(src):
            owners.setdefault(name, set()).add(src)
    pairs = []
    for d in docs:
        page = d.metadata.get("page")
        if isinstance(page, int) and page <= FIRST_PAGES:
            pairs += [(abbr, long, str(d.metadata["source"])) for abbr, long in long_forms(d.page_content)]
    title_owned = dict(owners)  # snapshot of rule 1
    for abbr, long, _ in pairs:  # rule 2
        if long in title_owned and len(title_owned[long]) == 1:
            owners.setdefault(abbr, set()).update(title_owned[long])
    for abbr, long, src in pairs:  # rule 3: only definitions that are about the paper itself
        if not related_to_title(abbr, long, src):
            continue
        for name in (abbr, long):
            if name not in title_owned and not (name == abbr and long in title_owned and len(title_owned[long]) == 1):
                owners.setdefault(name, set()).add(src)
    for word, src in content_names(docs, sources):  # rule 4: the paper's own text says this word belongs to it
        if word not in title_owned:
            owners.setdefault(word, set()).add(src)
    manual = manual_aliases()
    for alias, piece in manual.items():  # manual lines always win
        found = {s for s in sources if piece in norm(s)}
        if found:
            owners[alias] = found
    catalog = {src: [] for src in sources}
    for name, files in owners.items():
        if len(name) >= 3 and len(files) == 1 and (name not in GENERIC or name in manual):
            catalog[next(iter(files))].append(name)
    return {src: sorted(names) for src, names in catalog.items()}


def find_papers(question: str, catalog: dict) -> list:
    """Papers named in the question, in the order they are mentioned.
    Longest names are matched first and each stretch of the question is used once, so
    'Llama 3' goes to the Llama 3 paper and only the leftover 'LLaMA' goes to the LLaMA paper."""
    q = " " + norm(question) + " "
    names = sorted(((n, src) for src, ns in catalog.items() for n in ns), key=lambda t: -len(t[0]))
    taken, found = [], {}
    for name, src in names:
        for m in re.finditer(rf"(?<= ){re.escape(name)}(?= )", q):
            span = (m.start(), m.end())
            if any(span[0] < t[1] and t[0] < span[1] for t in taken):
                continue
            taken.append(span)
            if src not in found or span[0] < found[src][0]:
                found[src] = (span[0], name)
    return [(src, name) for src, (_, name) in sorted(found.items(), key=lambda kv: kv[1][0])]


def pretty(src: str) -> str:
    return re.sub(r"[_]+", " ", re.sub(r"^\d+[_-]?", "", re.sub(r"\.[A-Za-z0-9]+$", "", src)))


def paper_snippets(docs, n=200) -> dict:
    """{file: first n characters of its first page}: what the LLM planner reads besides the title."""
    first = {}
    for d in docs:
        src, page = str(d.metadata["source"]), d.metadata.get("page")
        if isinstance(page, int) and (src not in first or page < first[src][0]):
            first[src] = (page, " ".join(d.page_content.split())[:n])
    return {src: text for src, (_, text) in first.items()}


def llm_papers(question: str, snippets: dict, llm) -> list:
    """Ask the LLM which papers the question is about (it can use knowledge such as 'Transformer' = 'Attention Is All You Need').
    The reply is checked: only file names that really exist are kept. Bad or empty replies give []."""
    catalog_text = "\n".join(f"- {src}: {text}" for src, text in snippets.items())
    reply = (PLANNER_PROMPT | llm | StrOutputParser()).invoke({"catalog": catalog_text, "question": question})
    m = re.search(r"\[.*?\]", reply, re.S)
    try:
        names = json.loads(m.group(0)) if m else []
    except ValueError:
        return []
    valid = {norm(src): src for src in snippets}
    return [valid[norm(n)] for n in names if isinstance(n, str) and norm(n) in valid]


LIST_CUT = re.compile(r"^(.*)\s+(?:of|for|in|by|between|across|about|on)\s+(?:the\s+)?[\w.\-]+(?:\s+[\w.\-]+){0,2}$", re.I)


def aspect_of(question: str) -> str:
    """What is being asked, without the list of papers: 'Which GPU was used ... of the A, B, C and D?' -> 'Which GPU was used ...'.
    Only cuts when the question really ends in a list (2+ separators); otherwise the question is kept as it is."""
    q = question.strip().rstrip("?").strip()
    seps = q.count(",") + len(re.findall(r"\b(?:and|or|vs\.?|versus)\b", q, re.I))
    if q.count(",") == 0 or seps < 2:
        return question.strip()
    m = LIST_CUT.match(q.split(",")[0])
    aspect = m.group(1).strip() if m else ""
    return aspect if len(aspect.split()) >= 3 else question.strip()


def plan(question: str, catalog: dict, extra_papers=(), smart=False) -> list:
    """The plan = a list of {question, source, origin, keywords}. Empty list = not a multi-paper question.
    origin 'name' = the question names the paper (rules); 'llm' = only the LLM planner added it.
    smart=False: every sub-question is the whole question.  smart=True: only WHAT is asked plus this paper's name,
    so the other papers' names do not pull the search away from the real topic."""
    papers = [(src, name, "name") for src, name in find_papers(question, catalog)]
    have = {src for src, _, _ in papers}
    papers += [(src, name, origin) for src, name, origin in extra_papers if src not in have]
    if len(papers) < 2:
        return []
    aspect = aspect_of(question) if smart else question
    name_words = {w for _, name, _ in papers for w in words(name)}  # a paper's name says nothing about WHAT is asked
    keywords = " ".join(w for w in words(aspect) if w not in name_words)
    return [
        {"source": src, "origin": origin, "keywords": keywords,
         "question": f"{aspect}  (about {name})" if smart else f"{question}  (look for the answer about {name})"}
        for src, name, origin in papers
    ]


def show_plan(steps, question):
    print(f"\nQuestion: {question}")
    if not steps:
        print("Plan: fewer than 2 papers named -> fall back to plain Phase 2 search (phase2_ask.py).")
        return
    print(f"Plan: {len(steps)} sub-questions, one search per paper")
    for i, s in enumerate(steps, start=1):
        print(f"  {i}. [{s['source']}] ({ {'llm': 'LLM added', 'content': 'content matched'}.get(s['origin'], 'name matched') })  {s['question']}")


QUERY_STOP = STOP | {"which", "was", "were", "used", "using", "use", "be", "been", "did", "can", "could", "would", "main"}


def stem(w: str) -> str:
    """Very light stemming so 'GPU' matches 'GPUs' and 'train' matches 'trained'."""
    for suffix in ("ing", "ed", "es", "s"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            return w[: -len(suffix)]
    return w


def words(text: str) -> list:
    return [stem(w) for w in re.findall(r"[a-z0-9]+", text.lower())]  # keeps 'h800', 'a100', '65b' as single words


def bm25_rank(query: str, docs: list, k1=1.5, b=0.75) -> list:
    """Keyword ranking (BM25) of docs for the query. Returns doc indexes, best first, only docs with a hit.
    Exact tokens such as 'h800' or 'a100' score high here, which embeddings do not guarantee."""
    corpus = [words(d.page_content) for d in docs]
    q = [w for w in words(query) if w not in QUERY_STOP]
    n, avg = len(corpus), (sum(len(c) for c in corpus) / max(len(corpus), 1)) or 1
    df = Counter(w for c in corpus for w in set(c))
    scores = []
    for i, c in enumerate(corpus):
        tf = Counter(c)
        sc = 0.0
        for w in set(q):
            if tf[w]:
                idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                sc += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(c) / avg))
        scores.append(sc)
    return [i for i in sorted(range(n), key=lambda i: -scores[i]) if scores[i] > 0]


def hybrid_search(store, step, k):
    """Embedding search + keyword search inside ONE paper, merged with reciprocal rank fusion (RRF):
    a chunk that ranks high in either list (best: in both) comes first."""
    src, wide = step["source"], max(k * 4, 12)
    emb = store.similarity_search_with_score(step["question"], k=wide, filter={"source": src})
    got = store.get(where={"source": src}, include=["documents", "metadatas"])
    pool = [Document(page_content=t, metadata=m) for t, m in zip(got["documents"], got["metadatas"])]
    kw = [pool[i] for i in bm25_rank(step["keywords"], pool)[:wide]]
    fused, label = {}, {}
    for name, ranked in (("emb", [d for d, _ in emb]), ("kw", kw)):
        for rank, d in enumerate(ranked, start=1):
            key = d.page_content
            fused.setdefault(key, [d, 0.0])[1] += 1 / (60 + rank)
            label.setdefault(key, []).append(f"{name} #{rank}")
    best = sorted(fused.values(), key=lambda x: -x[1])[:k]
    return [(d, f"rrf={sc:.4f} ({', '.join(label[d.page_content])})") for d, sc in best]


def retrieve(steps, store, k, smart=False):
    """One search per paper, restricted to that paper. Returns [(step, [(doc, score), ...]), ...]."""
    if smart:
        return [(s, hybrid_search(store, s, k)) for s in steps]
    return [(s, store.similarity_search_with_score(s["question"], k=k, filter={"source": s["source"]})) for s in steps]


CARD_CHARS = 1500  # how much of a paper's first page goes into its card
MARGIN = 1.25  # the best paper must score this much higher than the second one to be taken without asking the LLM

TIEBREAK_PROMPT = ChatPromptTemplate.from_template(
    """Which ONE of these papers introduces or is mainly about "{mention}"?

{options}

Reply with the file name only, copied exactly from the list."""
)


def unresolved_mentions(question: str, catalog: dict) -> list:
    """Items of a list in the question that matched no paper by name: 'the Transformer, RAG, LLaMA' -> ['Transformer'].
    Only works for questions that END in a list (the same shape aspect_of() cuts)."""
    q = question.strip().rstrip("?").strip()
    aspect = aspect_of(question)
    if aspect == question.strip():
        return []
    rest = re.sub(r"^\s*(?:of|for|in|by|between|across|about|on)\s+", "", q[len(aspect.rstrip('?').strip()):].strip(), flags=re.I)
    items = [re.sub(r"^(?:the|a|an)\s+", "", i.strip(), flags=re.I) for i in re.split(r",|\s+and\s+|\s+or\s+|\s+vs\.?\s+", rest)]
    return [i for i in items if i and not find_papers(i, catalog)]


def card_docs(docs) -> list:
    """One short 'card' per paper: its title plus the start of its first page (title, authors, abstract)."""
    first = {}
    for d in docs:
        src, page = str(d.metadata["source"]), d.metadata.get("page")
        if isinstance(page, int) and (src not in first or page < first[src][0]):
            first[src] = (page, d)
    return [
        Document(page_content=f"{pretty(src)}. {' '.join(d.page_content.split())[:CARD_CHARS]}", metadata={"source": src})
        for src, (_, d) in sorted(first.items())
    ]


def get_cards_store(cards, embeddings, persist_dir):
    """A second Chroma collection ('cards') in the same folder. Only papers without a card yet are embedded,
    so adding a paper costs one embedding, not a rebuild."""
    from langchain_chroma import Chroma

    store = Chroma(collection_name="cards", embedding_function=embeddings, persist_directory=str(persist_dir),
                   collection_metadata={"hnsw:space": "cosine"})
    have = set(store.get()["ids"])
    new = [c for c in cards if c.metadata["source"] not in have]
    if new:
        store.add_documents(new, ids=[c.metadata["source"] for c in new])
    return store


def rank_cards(mention: str, cards: list, store) -> list:
    """Which papers does this name belong to, judged by CONTENT? Returns [(file, score)], best first.
    Three embedding searches (the name alone, 'we propose X', 'X introduced in this paper') plus a keyword
    search over the cards, merged with reciprocal rank fusion."""
    valid = {c.metadata["source"] for c in cards}
    fused = Counter()
    for q in (mention, f"we propose {mention}", f"{mention} introduced in this paper"):
        for rank, (d, _) in enumerate(store.similarity_search_with_score(q, k=5), start=1):
            if d.metadata["source"] in valid:
                fused[d.metadata["source"]] += 1 / (60 + rank)
    for rank, i in enumerate(bm25_rank(mention, cards)[:5], start=1):
        fused[cards[i].metadata["source"]] += 1 / (60 + rank)
    # Definition cue: "we propose ... the Transformer", "called BERT": the paper that INTRODUCES the name says so near it.
    # Many papers merely USE the name, so this separates the source paper from the rest. Counts double.
    cue = re.compile(rf"\b(?:propos\w*|introduc\w*|present\w*|called|named|dubbed)\b[^.]{{0,70}}?\b{re.escape(mention)}", re.I)
    found = []
    for c in cards:
        m = cue.search(c.page_content)
        if m:
            found.append((len(m.group(0)), c.metadata["source"]))
    for rank, (_, src) in enumerate(sorted(found), start=1):
        fused[src] += 2 / (60 + rank)
    return fused.most_common()


def resolve_mention(mention: str, cards: list, store, llm=None):
    """Returns (file or None, ranked candidates). Takes the best paper if it clearly leads;
    if the top two are close, lets the LLM choose among the top 3 (answer checked against the real file names)."""
    ranked = rank_cards(mention, cards, store)
    if not ranked:
        return None, ranked
    if len(ranked) == 1 or ranked[0][1] >= MARGIN * ranked[1][1]:
        return ranked[0][0], ranked
    if llm is None:
        return None, ranked
    text = {c.metadata["source"]: c.page_content for c in cards}
    options = "\n".join(f"- {src}: {text[src][:300]}" for src, _ in ranked[:3])
    reply = (TIEBREAK_PROMPT | llm | StrOutputParser()).invoke({"mention": mention, "options": options})
    for src, _ in ranked[:3]:
        if norm(src) in norm(reply):
            return src, ranked
    return None, ranked


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
            score = f"dist={dist:.3f}" if isinstance(dist, float) else dist
            print(f"  #{rank} {score} p.{d.metadata['page']}: {text[:250]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--show-aliases", action="store_true", help="print the names found for every paper and stop")
    ap.add_argument("--llm-plan", action="store_true",
                    help="also ask the LLM which papers the question is about (catches names like 'the Transformer')")
    ap.add_argument("--smart", action="store_true",
                    help="cleaner sub-questions, embeddings + keyword search, and names that match no title are resolved by paper CONTENT")
    ap.add_argument("--plan-only", action="store_true", help="show the plan and stop (needs only docs.jsonl)")
    ap.add_argument("--retrieve-only", action="store_true", help="plan + retrieve, no LLM")
    ap.add_argument("--k", type=int, default=PER_PAPER_K, help="chunks per paper")
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()

    all_docs = read_docs(config.DOCS_FILE)
    catalog = build_catalog([d for d in all_docs if d.metadata.get("file_type") in CATALOG_TYPES])
    if args.show_aliases:
        for src, names in catalog.items():
            print(f"{src}\n    {', '.join(names)}")
        return
    if not args.question:
        raise SystemExit("Give a question, or use --show-aliases.")
    extra = []
    if args.llm_plan:
        from rag.models import get_llm

        snippets = paper_snippets([d for d in all_docs if d.metadata.get("file_type") in CATALOG_TYPES])
        picked = llm_papers(args.question, snippets, get_llm(args.fake))
        print(f"LLM planner suggested: {picked or 'nothing usable'}")
        extra += [(src, pretty(src), "llm") for src in picked]
    if args.smart:
        missing = unresolved_mentions(args.question, catalog)
        if missing and not config.chroma_dir(args.fake).exists():
            print(f"Names with no paper match: {missing}. No index yet, so content matching is skipped (run phase2_index.py).")
        elif missing:
            from rag.models import get_embeddings, get_llm

            cards = card_docs([d for d in all_docs if d.metadata.get("file_type") in CATALOG_TYPES])
            cstore = get_cards_store(cards, get_embeddings(args.fake), config.chroma_dir(args.fake))
            llm = None
            for mention in missing:
                src, ranked = resolve_mention(mention, cards, cstore)
                if src is None and ranked:  # top two too close: ask the LLM, once
                    llm = llm or get_llm(args.fake)
                    src, ranked = resolve_mention(mention, cards, cstore, llm)
                shown = " | ".join(f"{f} ({sc:.3f})" for f, sc in ranked[:3])
                print(f"Content match for '{mention}': {shown or 'no candidates'}  ->  {src or 'AMBIGUOUS, not added'}")
                if src:
                    extra.append((src, mention, "content"))
    steps = plan(args.question, catalog, extra, args.smart)
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

    evidence = retrieve(steps, store, args.k, args.smart)
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