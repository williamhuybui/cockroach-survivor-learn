"""One entry point for everything: question in -> answer + context out.

Used by the API (FastAPI) and by eval_dev.py, so both run exactly the same code.
It reuses the planner and search pieces that live in phase3_ask.py.

    from rag.agent import Agent
    agent = Agent()                       # loads catalog, index, keyword index (once)
    result = agent.ask("What is LoRA?")   # {"answer": str, "context": [str, ...], "trace": {...}}

Routing (decided from the papers the question names):
    2+ papers   multi    one hybrid search per paper, one answer covering every paper
    1 paper     single   hybrid search inside that paper
    0 papers    global   hybrid search over every chunk of every paper
"Hybrid" = embedding search + keyword search (BM25), merged with reciprocal rank fusion.
"""
import math
import re
import time
from collections import Counter

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from phase3_ask import (CATALOG_TYPES, QUERY_STOP, SYNTH_PROMPT, aspect_of, build_catalog, card_docs, find_papers,
                        format_evidence, get_cards_store, hybrid_search, plan, resolve_mention, retrieve,
                        unresolved_mentions, words)
from rag import config
from rag.loaders import read_docs
from rag.models import get_embeddings, get_llm, get_vectorstore

NOT_FOUND = "I could not find this in the documents."
K_SINGLE = 5  # chunks for a question about one paper, or about no named paper
K_PER_PAPER = 5  # chunks per paper, per-paper mode (each LLM call sees only ONE paper, so there is room)
K_COMBINED = 3  # chunks per paper, combined mode (all papers share one prompt)
AUTHOR_GAP = 30  # characters: how much earlier the best paper must show the surname than the next one
AUTHOR = re.compile(r"\b([A-Z][a-z]{2,})(?:\s+and\s+[A-Z][a-z]+)?\s+et\s+al\b")  # "Kaplan et al."

RULES = """Rules:
- Answer directly and briefly. Copy exact numbers, names and units as written in the context.
- Cite the source after each claim like [file p.page].
- Use ONLY the context. No outside knowledge.
- If the context does not state what the question asks (for example the question names a different year,
  dataset, model or quantity than the one in the context), reply exactly: """ + NOT_FOUND

SINGLE_PROMPT = ChatPromptTemplate.from_template(
    "You answer questions about research papers.\n" + RULES.replace("{", "{{").replace("}", "}}") + """

Context:
{context}

Question: {question}

Answer:"""
)

# Multi-paper answers use SYNTH_PROMPT from phase3_ask.py, the prompt that was tested on real data (D13).
# An earlier "answer briefly" prompt made the model return a terse list without paper names.

PER_PAPER_PROMPT = ChatPromptTemplate.from_template(
    """You answer a question using ONLY the context below. The context comes from ONE paper: {name}.
Rules:
- Answer only for {name}, in one short sentence. Copy exact numbers, names and units as written in the context.
- Cite the source like [file p.page].
- Use ONLY the context. If it does not state what the question asks, reply exactly: not stated in the documents

Context:
{context}

Question: {question}

Answer for {name}:"""
)

VERIFY_PROMPT = ChatPromptTemplate.from_template(
    """Context:
{context}

Question: {question}

Does the context explicitly state the specific fact the question asks for? Answer with one word: YES or NO."""
)


def surface(alias: str, query: str) -> str:
    """The alias as the user wrote it: 'llama 3' -> 'Llama 3', 'deepseek v3' -> 'DeepSeek-V3', 'rag' -> 'RAG'."""
    pattern = r"[\W_]*".join(re.escape(w) for w in alias.split())
    m = re.search(rf"\b{pattern}\b", query, re.I)
    return m.group(0) if m else alias


class KeywordIndex:
    """BM25 over ALL chunks, tokens computed once, so a global keyword search takes milliseconds."""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.tf = [Counter(words(d.page_content)) for d in docs]
        self.len = [sum(c.values()) for c in self.tf]
        self.avg = (sum(self.len) / max(len(docs), 1)) or 1
        self.df = Counter(w for c in self.tf for w in c)

    def top(self, query: str, n: int) -> list:
        q = [w for w in set(words(query)) if w not in QUERY_STOP]
        total = len(self.docs)
        scores = []
        for tf, length in zip(self.tf, self.len):
            sc = 0.0
            for w in q:
                f = tf.get(w)
                if f:
                    idf = math.log(1 + (total - self.df[w] + 0.5) / (self.df[w] + 0.5))
                    sc += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * length / self.avg))
            scores.append(sc)
        order = sorted(range(total), key=lambda i: -scores[i])[:n]
        return [self.docs[i] for i in order if scores[i] > 0]


def fuse(lists: list, k: int) -> list:
    """Reciprocal rank fusion of ranked lists of Documents. Returns the top k Documents."""
    score, keep = Counter(), {}
    for ranked in lists:
        for rank, d in enumerate(ranked, start=1):
            score[d.page_content] += 1 / (60 + rank)
            keep.setdefault(d.page_content, d)
    return [keep[t] for t, _ in score.most_common(k)]


class Agent:
    def __init__(self, fake=False, verify=False, llm=None, context_with_source=True, multi_mode="combined"):
        """verify=True adds a YES/NO check ("does the context state the answer?") before answering
        single-paper questions. It costs one more LLM call; eval_dev.py shows whether it helps.
        multi_mode: 'combined'  = one LLM call that sees the evidence of all papers at once (default, same as phase3_ask.py)
                    'per-paper' = one small LLM call per paper, answers joined as "Paper: answer" lines (experimental)"""
        persist_dir = config.chroma_dir(fake)
        if not persist_dir.exists():
            raise RuntimeError("No index found. Run phase1_ingest.py and phase2_index.py first.")
        self.verify, self.fake, self.with_source, self.multi_mode = verify, fake, context_with_source, multi_mode
        paper_docs = [d for d in read_docs(config.DOCS_FILE) if d.metadata.get("file_type") in CATALOG_TYPES]
        self.catalog = build_catalog(paper_docs)
        self.cards = card_docs(paper_docs)
        self.embeddings = get_embeddings(fake)
        self.store = get_vectorstore(self.embeddings, persist_dir)
        self.llm = llm or get_llm(fake)
        got = self.store.get(include=["documents", "metadatas"])
        self.kw = KeywordIndex([Document(page_content=t, metadata=m) for t, m in zip(got["documents"], got["metadatas"])])
        self._cards_store = None

    # ---- planning -------------------------------------------------------------------------------
    def _cstore(self):
        if self._cards_store is None:
            self._cards_store = get_cards_store(self.cards, self.embeddings, config.chroma_dir(self.fake))
        return self._cards_store

    def _resolve_extra(self, query: str, have: set) -> list:
        """Papers the question refers to without naming them: list items that match no title (the Transformer),
        and authors ("Kaplan et al."). Returns [(file, name, 'content')]."""
        extra = []
        for mention in unresolved_mentions(query, self.catalog):
            src, ranked = resolve_mention(mention, self.cards, self._cstore())
            if src is None and ranked:  # the top two are close: let the LLM choose
                src, ranked = resolve_mention(mention, self.cards, self._cstore(), self.llm)
            if src and src not in have:
                extra.append((src, mention, "content"))
                have.add(src)
        for surname in AUTHOR.findall(query):
            # "Kaplan et al." = the FIRST author. The first author's name comes right after the title, so take the paper
            # where the name appears earliest near the top, if it clearly leads the next one (other papers may list
            # the same person further down their author line, or cite them).
            hits = sorted((c.page_content[:600].lower().find(surname.lower()), c.metadata["source"])
                          for c in self.cards if surname.lower() in c.page_content[:600].lower())
            if hits and (len(hits) == 1 or hits[1][0] - hits[0][0] >= AUTHOR_GAP) and hits[0][1] not in have:
                extra.append((hits[0][1], f"{surname} et al.", "content"))
                have.add(hits[0][1])
        return extra

    # ---- answering ------------------------------------------------------------------------------
    def _invoke(self, prompt, inputs, tries=4) -> str:
        """Run prompt -> LLM. Retries with growing waits (5, 15, 30 s): after a crash a local Ollama server needs
        time to reload the model, and an immediate retry fails again."""
        chain = prompt | self.llm | StrOutputParser()
        for attempt in range(tries):
            try:
                return chain.invoke(inputs).strip()
            except Exception as e:
                if attempt == tries - 1:
                    raise
                wait = (5, 15, 30)[attempt]
                print(f"  LLM error ({type(e).__name__}: {str(e)[:150]}), retry {attempt + 1} in {wait}s", flush=True)
                time.sleep(wait)

    def _answer_per_paper(self, query: str, evidence: list, papers: list) -> str:
        """One small question per paper, so a small model never has to juggle five papers in one prompt."""
        question, lines = aspect_of(query), []
        for (step, hits), paper in zip(evidence, papers):
            label = paper["name"] if paper["origin"] == "content" else surface(paper["name"], query)
            context = "\n\n".join(f"[{d.metadata['source']} p.{d.metadata['page']}]\n{d.page_content}" for d, _ in hits)
            answer = self._invoke(PER_PAPER_PROMPT, {"name": label, "context": context, "question": question})
            lines.append(f"{label}: {answer}")
        if all("not stated" in line.lower() for line in lines):
            return NOT_FOUND
        return "\n".join(lines)

    def _answer_single(self, query: str, docs: list) -> str:
        context = "\n\n".join(f"[{d.metadata['source']} p.{d.metadata['page']}]\n{d.page_content}" for d in docs)
        if self.verify and docs:
            check = self._invoke(VERIFY_PROMPT, {"context": context, "question": query})
            if check.strip().upper().startswith("NO"):
                return NOT_FOUND
        return self._invoke(SINGLE_PROMPT, {"context": context, "question": query})

    def ask(self, query: str) -> dict:
        t0 = time.time()
        named = find_papers(query, self.catalog)
        have = {src for src, _ in named}
        extra = self._resolve_extra(query, have)
        papers = [{"source": s, "name": n, "origin": "name"} for s, n in named]
        papers += [{"source": s, "name": n, "origin": o} for s, n, o in extra]
        steps = plan(query, self.catalog, extra, smart=True)  # [] unless 2+ papers

        if steps:
            route = "multi"
            per_paper = self.multi_mode == "per-paper"
            evidence = retrieve(steps, self.store, K_PER_PAPER if per_paper else K_COMBINED, smart=True)
            docs = [d for _, hits in evidence for d, _ in hits]
            if per_paper and len(evidence) == len(papers):
                answer = self._answer_per_paper(query, evidence, papers)
            else:
                answer = self._invoke(SYNTH_PROMPT, {"evidence": format_evidence(evidence), "question": query})
        else:
            if len(papers) == 1:
                route = "single"
                skip = set(words(papers[0]["name"]))
                step = {"source": papers[0]["source"], "question": query,
                        "keywords": " ".join(w for w in words(query) if w not in skip)}
                docs = [d for d, _ in hybrid_search(self.store, step, K_SINGLE)]
            else:
                route = "global"
                emb = [d for d, _ in self.store.similarity_search_with_score(query, k=20)]
                docs = fuse([emb, self.kw.top(query, 20)], K_SINGLE)
            answer = self._answer_single(query, docs)

        seen, context, chunks = set(), [], []
        for d in docs:
            if d.page_content in seen:
                continue
            seen.add(d.page_content)
            src, page = d.metadata["source"], d.metadata["page"]
            context.append(f"[{src} p.{page}] {d.page_content}" if self.with_source else d.page_content)
            chunks.append({"source": src, "page": page, "text": d.page_content})
        return {"answer": answer, "context": context,
                "trace": {"route": route, "mode": self.multi_mode if route == "multi" else "-", "papers": papers, "chunks": chunks, "seconds": round(time.time() - t0, 2)}}