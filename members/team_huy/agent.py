"""Huy's agent (team3): a ReAct agent over the research papers, with weighted hybrid retrieval.

api_view.py calls answer(question, ask_llm, load_documents) and expects {"answer": str, "sources": [file, ...]}.

How it works (LangGraph loop; LangChain tools, prompts, Ollama chat model and embeddings):

    plan -> search -> extract -+-> search (retry with a new query, or the next sub-question)
                               +-> final -> answer

  plan     the LLM splits the question into small sub-questions, one per paper or fact
           ("compare A, B and C" becomes three searches; a simple question stays one)
  search   Action: runs a LangChain tool (search_documents / search_in_paper) with the weighted retriever
  extract  Observation -> Thought: the LLM reads the results and either records the fact (Found) or says
           NOT FOUND and proposes a new query. This is the "does the question need more search?" decision:
           a sub-question is retried up to MAX_TRIES times; a retry searches all papers
  final    the recorded facts are turned into a short answer. No facts at all -> "I don't know"
           (a made-up answer costs points; abstaining does not)

A recorded fact is kept only if every number in it appears in the search results (guards against a small
model inventing figures). Sources are the papers whose results produced the facts.

Retrieval = weighted sum of three scores, each min-max scaled to 0..1 over all chunks:
    WEIGHTS: embedding (EmbeddingGemma) + BM25 + exact keyword match.
The chunks come from chunking_output/ (run chunking.py first; Ollama needs gemma3 and embeddinggemma).

Try it:  python3 members/team_huy/agent.py "Which GPUs trained the Transformer and DeepSeek-V3?"
"""
import difflib
import json
import re
import sys
import threading
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import END, START, StateGraph

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from keyword_search import keyword_pattern, question_keywords  # noqa: E402
from tfidf_bm25 import Bm25Index  # noqa: E402

CHUNKS_CSV = HERE / "chunking_output" / "chunks.csv"
DOCS_CSV = HERE / "chunking_output" / "documents.csv"
SECTIONS_CSV = HERE / "chunking_output" / "sections.csv"

LLM_MODEL = "gemma3"  # same model as ask_llm in api_view.py
EMBED_MODEL = "embeddinggemma"
NUM_CTX = 8192  # Ollama's default window is smaller and silently cuts the start of long prompts

WEIGHTS = {"embedding": 0.5, "bm25": 0.3, "keyword": 0.2}  # sum to 1
TOP_K = 5  # chunks returned per search
MAX_PER_PAPER = 3  # so one paper cannot fill every slot of a comparison question
CHUNK_CHARS_SHOWN = 1500
MAX_TRIES = 3  # searches per sub-question before giving up on it
MAX_SUBQUESTIONS = 6
IDK = "I don't know - this is not in the documents."


# ============================================================================ retrieval
def unit(v):
    return v / (np.linalg.norm(v) or 1.0)


def scaled(scores):
    """Min-max scale to 0..1 so scores of different methods can be added."""
    lo, hi = scores.min(), scores.max()
    return (scores - lo) / (hi - lo) if hi > lo else np.zeros_like(scores)


class Retriever:
    """Weighted hybrid search over the chunks (embedding + BM25 + keyword)."""

    def __init__(self):
        if not CHUNKS_CSV.exists():
            raise FileNotFoundError(f"{CHUNKS_CSV} not found - run members/team_huy/chunking.py first")
        chunks = pd.read_csv(CHUNKS_CSV)
        docs = pd.read_csv(DOCS_CSV).set_index("doc_id")
        sections = pd.read_csv(SECTIONS_CSV).set_index("section_id")
        self.doc_id = chunks.doc_id.to_numpy()
        self.text = chunks.chunk_text.tolist()
        self.lower = [t.lower() for t in self.text]
        self.title = chunks.doc_id.map(docs.title).to_numpy()
        self.file = chunks.doc_id.map(docs.file).to_numpy()
        self.section = chunks.section_id.map(sections.section_title).to_numpy()
        self.papers = {d: (r.title, r.file) for d, r in docs.iterrows()}
        self.embeddings = np.array([json.loads(e) for e in chunks.chunk_embedding])
        self.embeddings /= np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.bm25 = Bm25Index(chunks[["embed_text"]])
        self.embedder = OllamaEmbeddings(model=EMBED_MODEL)

    def keyword_scores(self, query):
        terms = question_keywords(query)
        patterns = [keyword_pattern(t) for t in terms]
        out = np.zeros(len(self.text))
        for i, text in enumerate(self.text):
            counts = [len(p.findall(text)) for p in patterns]
            out[i] = sum(c > 0 for c in counts) + 0.1 * min(sum(counts), 10) / 10
        return out

    def scores(self, query):
        vec = unit(np.array(self.embedder.embed_query(f"task: search result | query: {query}")))
        parts = {"embedding": self.embeddings @ vec, "bm25": self.bm25.scores(query),
                 "keyword": self.keyword_scores(query)}
        return sum(WEIGHTS[name] * scaled(s) for name, s in parts.items())

    def search(self, query, doc_id=None, k=TOP_K):
        scores = self.scores(query)
        if doc_id:
            scores = np.where(self.doc_id == doc_id, scores, -1.0)
        hits, per_paper = [], {}
        for i in np.argsort(-scores):
            if scores[i] < 0 or len(hits) == k:
                break
            if per_paper.get(self.doc_id[i], 0) >= (k if doc_id else MAX_PER_PAPER):
                continue
            per_paper[self.doc_id[i]] = per_paper.get(self.doc_id[i], 0) + 1
            hits.append({"doc_id": self.doc_id[i], "title": self.title[i], "file": self.file[i],
                         "section": self.section[i], "text": self.text[i], "score": float(scores[i])})
        return hits

    def resolve_paper(self, name):
        """doc_id of the paper that best matches a title the LLM wrote, or None."""
        norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
        want = norm(name)
        if not want:
            return None
        best, best_score = None, 0.0
        for d, (title, file) in self.papers.items():
            t = norm(title)
            if want == t:
                score = 1.0
            elif want in t:  # the LLM wrote part of the title: "Llama 3" -> "The Llama 3 Herd of Models"
                score = 0.9
            elif t in want:  # the LLM wrote more than the title; the longer the title, the better the match
                score = 0.8 + 0.1 * len(t) / len(want)
            else:
                score = difflib.SequenceMatcher(None, want, t).ratio()
            if score > best_score:
                best, best_score = d, score
        return best if best_score >= 0.6 else None


_retriever = None
_retriever_lock = threading.Lock()  # the evaluation asks several questions at once


def get_retriever():
    global _retriever
    with _retriever_lock:
        if _retriever is None:
            _retriever = Retriever()
        return _retriever


def format_hits(hits):
    if not hits:
        return "(no results)"
    return "\n\n".join(f"[{i}] {h['title']} > {h['section']}\n{h['text'][:CHUNK_CHARS_SHOWN]}"
                       for i, h in enumerate(hits, 1))


# ============================================================================ LangChain tools
@tool(response_format="content_and_artifact")
def search_documents(query: str):
    """Search all research papers. Use specific words: model names, numbers, benchmark names."""
    hits = get_retriever().search(query)
    return format_hits(hits), hits


@tool(response_format="content_and_artifact")
def search_in_paper(paper: str, query: str):
    """Search inside one research paper, given its title."""
    r = get_retriever()
    doc_id = r.resolve_paper(paper)
    if doc_id is None:
        return f"No paper called '{paper}'. Use a title from the paper list.", []
    hits = r.search(query, doc_id=doc_id)
    return format_hits(hits), hits


def run_tool(tool_fn, args):
    msg = tool_fn.invoke({"name": tool_fn.name, "args": args, "id": "call", "type": "tool_call"})
    return msg.content, msg.artifact


# ============================================================================ prompts and LLM
PLAN_PROMPT = PromptTemplate.from_template("""You answer questions about a fixed set of research papers.
Papers:
{papers}

Question: {question}

Break the question into the smallest self-contained sub-questions needed to answer it: one per paper or per fact.
A simple question needs only one sub-question. Never invent papers: name a paper only if the question mentions it.
Write one line per sub-question, in this format:  <paper title from the list, or any> | <self-contained sub-question>
At most {max_subquestions} lines. Output only the lines.""")

EXTRACT_PROMPT = PromptTemplate.from_template("""Sub-question: {subquestion}

Search results:
{observation}

Find the answer to the sub-question in the search results. Use ONLY the results, never outside knowledge.
Answer only if the results state it about exactly the thing asked (same model, paper, dataset, metric);
otherwise write NOT FOUND. Copy numbers and names exactly.

Reply in exactly this format:
Thought: <one sentence>
Found: <one or two short sentences with the exact numbers/names, copied from the results> or NOT FOUND
Retry query: <only if NOT FOUND: a different search query that uses other keywords>""")

FINAL_PROMPT = PromptTemplate.from_template("""Answer the question using ONLY the facts below. Be concise and specific: exact numbers and names,
and say which paper each fact comes from. Cover every part of the question that the facts support;
for a part the facts do not cover, say it was not found in the documents. Write 1-3 sentences. Do not mention
"facts" or "provided text", and add no citations or reference numbers.

Facts:
{facts}

Question: {question}""")

_llm = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOllama(model=LLM_MODEL, temperature=0, num_ctx=NUM_CTX, client_kwargs={"timeout": 300})
    return _llm


def call_llm(prompt, **values):
    return (prompt | get_llm() | StrOutputParser()).invoke(values).strip()


def paper_list():
    return "\n".join(f"- {title}" for title, _ in get_retriever().papers.values())


def words(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


EXTRA_ALIASES = {  # names people use for a paper besides its title (lowercase, words separated by spaces)
    "Attention Is All You Need": ["transformer"],
    "Neural Machine Translation by Jointly Learning to Align and Translate": ["bahdanau", "neural machine translation"],
    "GPT-3 Language Models are Few-Shot Learners": ["gpt 3", "gpt3"],
    "Scaling Laws for Neural Language Models": ["kaplan", "scaling laws"],
    "Chinchilla Training Compute-Optimal LLMs": ["chinchilla"],
    "DPO Direct Preference Optimization": ["dpo", "direct preference optimization"],
    "Chain-of-Thought Prompting": ["chain of thought", "cot"],
    "DeepSeek-R1": ["deepseek r1", "deepseek r1 zero"],
    "LLaMA": ["llama"],
    "Mixtral of Experts": ["mixtral"],
    "Retrieval-Augmented Generation": ["rag", "retrieval augmented generation"],
    "The Llama 3 Herd of Models": ["llama 3", "llama3"],
    "DeepSeekMath (GRPO)": ["deepseekmath", "grpo"],
    "Scaling LLM Test-Time Compute Optimally": ["test time compute"],
    "DeepSeek-V3": ["deepseek v3"],
    "Alignment Faking in Large Language Models": ["alignment faking"],
    "Does RL Really Incentivize Reasoning Capacity Beyond the Base Model": ["rlvr"],
    "Qwen3 Technical Report": ["qwen3", "qwen 3"],
    "Gemini 2.5": ["gemini 2 5", "gemini"],
    "MAS-FIRE": ["mas fire"],
    "How Controllable Are Large Language Models (SteerEval)": ["steereval"],
    "Harnessing the Universal Geometry of Embeddings": ["vec2vec", "universal geometry"],
}


def norm_text(text):
    return " " + " ".join(re.findall(r"[a-z0-9]+", text.lower())) + " "


def mentioned_papers(text):
    """doc_ids of the papers a text names, by title or nickname. 'llama' inside 'Llama 3' counts only for Llama 3."""
    papers = get_retriever().papers
    text = norm_text(text)
    found = []  # (start, end, alias, doc_id)
    for doc_id, (title, _) in papers.items():
        aliases = {norm_text(title).strip(), norm_text(re.sub(r"\(.*?\)", "", title)).strip()}
        aliases |= {norm_text(m).strip() for m in re.findall(r"\((.*?)\)", title)}
        aliases |= set(EXTRA_ALIASES.get(title, []))
        for alias in aliases - {""}:
            for m in re.finditer(r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])", text):
                found.append((m.start(), m.end(), alias, doc_id))
    docs = []
    for a in found:  # drop a match that lies inside a longer match of another paper
        inside = any(b[3] != a[3] and b[0] <= a[0] and a[1] <= b[1] and (b[1] - b[0]) > (a[1] - a[0]) for b in found)
        if not inside and a[3] not in docs:
            docs.append(a[3])
    return docs


def parse_plan(text, question):
    """[{"q": sub-question, "doc": doc_id or None}]. A paper hint is kept only if the question names that paper;
    papers the question names but the plan forgot get a sub-question of their own."""
    r, subs = get_retriever(), []
    named = mentioned_papers(question)
    for line in text.splitlines():
        if "|" in line:
            paper, sub = line.split("|", 1)
            paper, sub = re.sub(r"^[\s\-*•\d.)]+", "", paper), sub.split("|")[0]
        else:
            paper, sub = "", line
        sub = sub.strip(" -*•")
        if len(sub) <= 10 or not ("|" in line or sub.endswith("?")):
            continue
        doc = r.resolve_paper(paper) if paper else None
        doc = doc if doc in named else None
        if doc is None and len(mentioned_papers(sub)) == 1:
            doc = mentioned_papers(sub)[0]
        if any(o["doc"] == doc and len(words(o["q"]) & words(sub)) >= 0.6 * len(words(sub)) for o in subs):
            continue  # near-duplicate of a sub-question already planned
        subs.append({"q": sub, "doc": doc})
    for doc in named:
        if not any(sq["doc"] == doc for sq in subs):
            subs.append({"q": question, "doc": doc})
    return subs[:MAX_SUBQUESTIONS] or [{"q": question, "doc": None}]


def parse_extract(text):
    """(found answer or None, retry query) from the LLM reply."""
    found = re.search(r"Found:\s*(.*?)(?=\n\s*Retry query:|\Z)", text, re.S | re.I)
    retry = re.search(r"Retry query:\s*(.*)", text, re.I)
    found = found.group(1).strip() if found else ""
    if not found or found.lower().startswith("not found"):
        found = None
    return found, (retry.group(1).strip() if retry else "")


def grounded(line, observation):
    """A fact is kept only if every number in it occurs in the search results."""
    squash = lambda t: re.sub(r"[\s,]", "", t.lower())
    numbers = re.findall(r"\d+(?:\.\d+)?", line)
    return all(n in squash(observation) for n in numbers if len(n) > 1 or "." in n)


def supported(sub, found, observation, hits):
    """Reject a 'found' answer that the results do not support:
    - its words must mostly occur in the results (a small model sometimes invents text),
    - terms with digits in the sub-question (2019, 8x7B, 405B) must occur in the results,
    - if the sub-question names papers, the fact must come from one of them."""
    obs = norm_text(observation)
    content = [w for w in words(found) if len(w) > 3]
    if content and sum(f" {w} " in obs or w in obs for w in content) / len(content) < 0.7:
        return False
    squash = re.sub(r"[^a-z0-9]", "", observation.lower())
    if any(re.sub(r"[^a-z0-9]", "", t.lower()) not in squash
           for t in question_keywords(sub) if any(c.isdigit() for c in t)):
        return False
    named = mentioned_papers(sub)
    doc = source_of(found, hits)
    return not (named and doc and doc not in named)


def source_of(line, hits):
    """doc_id of the result whose text shares the most words with a fact."""
    mine = {w for w in words(line) if len(w) > 3}
    best = max(hits, key=lambda h: len(mine & words(h["text"])), default=None)
    return best["doc_id"] if best and mine & words(best["text"]) else None


# ============================================================================ LangGraph agent
class AgentState(TypedDict):
    question: str
    subs: list  # sub-questions: {"q", "doc"}
    idx: int  # sub-question being worked on
    tries: int  # searches already done for it
    query: str  # query for the next search ("" = the sub-question itself)
    observation: str
    hits: list  # latest results (doc_id, text)
    facts: list
    cited: list  # doc_ids whose results produced facts
    answer: str
    trace: list


def plan_node(state):
    subs = parse_plan(call_llm(PLAN_PROMPT, papers=paper_list(), question=state["question"],
                               max_subquestions=MAX_SUBQUESTIONS), state["question"])
    papers = get_retriever().papers
    return {"subs": subs, "idx": 0, "tries": 0, "query": "",
            "trace": [f"plan: {[(papers[s['doc']][0] if s['doc'] else 'any', s['q']) for s in subs]}"]}


def search_node(state):
    sub = state["subs"][state["idx"]]
    query = state["query"] or sub["q"]
    doc = sub["doc"] if state["tries"] == 0 else None  # a retry searches every paper (the hint may be wrong)
    if doc:
        title = get_retriever().papers[doc][0]
        observation, hits = run_tool(search_in_paper, {"paper": title, "query": query})
        label = f"search_in_paper[{title} | {query}]"
    else:
        observation, hits = run_tool(search_documents, {"query": query})
        label = f"search[{query}]"
    return {"observation": observation, "hits": [{"doc_id": h["doc_id"], "text": h["text"]} for h in hits],
            "trace": state["trace"] + [f"{label} -> {[h['title'] for h in hits]}"]}


def extract_node(state):
    sub = state["subs"][state["idx"]]
    found, retry = parse_extract(call_llm(EXTRACT_PROMPT, subquestion=sub["q"], observation=state["observation"]))
    if found and not (grounded(found, state["observation"])
                      and supported(sub["q"], found, state["observation"], state["hits"])):
        found = None  # not backed by the results (invented number or text, or from the wrong paper)
    if found:
        found = re.sub(r"\s*\(?(?:source:?\s*)?\[\d+\]\)?", "", found).strip()  # result markers like "[3]"
    trace = state["trace"] + [f"extract: {('FOUND ' + found[:160]) if found else 'NOT FOUND'}"]
    if found:
        cited = list(state["cited"])
        doc = source_of(found, state["hits"])
        if doc and doc not in cited:
            cited.append(doc)
        return {"facts": state["facts"] + [f"{sub['q']} -> {found}"], "cited": cited,
                "idx": state["idx"] + 1, "tries": 0, "query": "", "trace": trace}
    if state["tries"] + 1 >= MAX_TRIES:  # give up on this sub-question
        return {"idx": state["idx"] + 1, "tries": 0, "query": "", "trace": trace + ["gave up on this sub-question"]}
    return {"tries": state["tries"] + 1, "query": retry or sub["q"], "trace": trace}


def final_node(state):
    if not state["facts"]:
        return {"answer": IDK}
    return {"answer": call_llm(FINAL_PROMPT, facts="\n".join(f"- {f}" for f in state["facts"]),
                               question=state["question"])}


def route_after_extract(state):
    return "final" if state["idx"] >= len(state["subs"]) else "search"


def build_graph():
    g = StateGraph(AgentState)
    for name, fn in [("plan", plan_node), ("search", search_node), ("extract", extract_node), ("final", final_node)]:
        g.add_node(name, fn)
    g.add_edge(START, "plan")
    g.add_edge("plan", "search")
    g.add_edge("search", "extract")
    g.add_conditional_edges("extract", route_after_extract, {"search": "search", "final": "final"})
    g.add_edge("final", END)
    return g.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_agent(question):
    start = {"question": question, "subs": [], "idx": 0, "tries": 0, "query": "", "observation": "", "hits": [],
             "facts": [], "cited": [], "answer": "", "trace": []}
    state = get_graph().invoke(start, {"recursion_limit": 100})
    answer = re.sub(r"\s*[\[(](?:Source:?\s*)?\d+(?:,\s*\d+)*[\])]", "", state["answer"]).strip()  # "[2, 4]", "(Source 1)"
    if not state["facts"] or not answer or "don't know" in answer.lower():
        return {"answer": IDK, "sources": [], "trace": state["trace"]}
    papers = get_retriever().papers
    return {"answer": answer, "sources": [papers[d][1] for d in state["cited"]], "trace": state["trace"]}


def answer(question: str, ask_llm, load_documents) -> dict:
    """Entry point for api_view.py. ask_llm / load_documents are not used: the agent calls gemma3
    through LangChain (bigger context window) and reads the chunk index built by chunking.py."""
    result = run_agent(question)
    return {"answer": result["answer"], "sources": result["sources"]}


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Llama 3 405B and DeepSeek-V3 were both pre-trained at large scale. How many tokens did each use?"
    out = run_agent(q)
    print("\n".join(f"  {t}" for t in out["trace"]))
    print(f"\nAnswer: {out['answer']}\nSources: {out['sources']}")
