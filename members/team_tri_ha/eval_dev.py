"""Score the agent on the competition dev set (data/dev_set.csv). Run it after every change.

Run:
    python eval_dev.py                          # all questions
    python eval_dev.py --only D13 D05           # just these ids
    python eval_dev.py --multi per-paper        # multi-paper questions: one small LLM call per paper, to compare
    python eval_dev.py --verify                 # also check "does the context state the answer?" before answering
    python eval_dev.py --judge                  # let the LLM grade each answer (slower, a small model is a rough judge)
    python eval_dev.py --csv path/to/dev_set.csv --fake     # code check only, no Ollama

What is measured (per question, then averaged by type and difficulty):
    doc_recall    share of the expected source papers that appear in the returned context (retrieval check)
    score         share of the key facts of ground_truth found in the answer: numbers / names with digits /
                  ALL-CAPS words (P100, 41.8, BLEU...). If ground_truth has none, share of its long words.
                  A cheap proxy: it can miss a correct answer worded differently. Read the answers too.
    refused       the answer says it could not find the information.
                  unanswerable questions SHOULD be refused; answerable ones should NOT.
Files written: data/eval/dev_<time>.csv (numbers) and .jsonl (answers, context, trace for error analysis).
"""
import argparse
import csv
import json
import re
import time
from collections import defaultdict
from pathlib import Path

from rag import config

REFUSE = re.compile(
    r"could not find|cannot find|can't find|not (?:found|stated|mentioned|reported|provided|specified)"
    r"|not in the (?:documents|context|provided)|no information|does not (?:report|state|provide|mention|specify)"
    r"|doesn't (?:report|state|provide|mention)|unable to find",
    re.I,
)
KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9.,%/+\-]*\d[A-Za-z0-9.,%/+\-]*|\b[A-Z]{2,}[A-Za-z0-9\-]*\b")
STOPWORDS = {"about", "which", "their", "there", "these", "those", "while", "where", "being", "other", "within",
             "using", "uses", "used", "with", "from", "that", "this", "have", "such", "than", "also", "over"}


def clean_numbers(text: str) -> str:
    return re.sub(r"(?<=\d),(?=\d)", "", text.lower())  # 2,500M -> 2500m


def key_tokens(ground_truth: str) -> list:
    out = []
    for t in KEY.findall(ground_truth):
        t = clean_numbers(t).strip(".,-")
        if t and t not in out:
            out.append(t)
    return out


def long_words(text: str) -> list:
    return sorted({w for w in re.findall(r"[a-z]{5,}", text.lower()) if w not in STOPWORDS})


def answer_score(answer: str, ground_truth: str) -> float:
    a = clean_numbers(answer)
    keys = key_tokens(ground_truth)
    if keys:
        return sum(k in a for k in keys) / len(keys)
    words = long_words(ground_truth)
    return sum(w in a for w in words) / len(words) if words else 0.0


def doc_recall(expected: list, chunks: list) -> float:
    got = {re.sub(r"\.[A-Za-z0-9]+$", "", c["source"]) for c in chunks}
    return sum(e in got for e in expected) / len(expected)


def judge(llm, question, truth, answer) -> str:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_template(
        "Question: {q}\nGround truth: {t}\nAnswer: {a}\n\n"
        "Is the answer correct and complete compared with the ground truth? "
        "Reply with one word: CORRECT, PARTIAL or WRONG.")
    reply = (prompt | llm | StrOutputParser()).invoke({"q": question, "t": truth, "a": answer}).upper()
    for word in ("CORRECT", "PARTIAL", "WRONG"):
        if word in reply:
            return word
    return "UNKNOWN"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fmt(x):
    return "  -  " if x is None else f"{x:5.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(config.ROOT / "data" / "dev_set.csv"))
    ap.add_argument("--only", nargs="+", help="question ids, e.g. D13 D05")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--multi", choices=["per-paper", "combined"], default="combined",
                    help="multi-paper questions: one call for all papers (default, as phase3_ask.py) or one small call per paper")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--show", action="store_true", help="print every answer, not only the failures")
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()

    path = Path(args.csv)
    if not path.exists():
        raise SystemExit(f"Dev set not found: {path}\nPut the competition CSV there, or pass --csv.")
    with path.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("question")]
    if args.only:
        rows = [r for r in rows if r["id"] in set(args.only)]

    from rag.agent import Agent  # imported late so --help is instant

    agent = Agent(fake=args.fake, verify=args.verify, multi_mode=args.multi)
    results = []
    print(f"{'id':4} {'type':13} {'diff':8} {'route':6} {'recall':>6} {'score':>6} {'refused':>8} {'sec':>6}")
    print("-" * 66)
    for r in rows:
        expected = [s.strip() for s in r["source_docs"].split(";") if s.strip()]
        try:
            res = agent.ask(r["question"])
        except Exception as e:  # one failed question must not stop the whole run
            print(f"{r['id']:4} ERROR {type(e).__name__}: {str(e)[:300]}")
            res = {"answer": f"ERROR: {type(e).__name__}: {str(e)[:300]}", "context": [],
                   "trace": {"route": "error", "papers": [], "chunks": [], "seconds": 0.0}}
        trace = res["trace"]
        refused = bool(REFUSE.search(res["answer"]))
        unanswerable = r["type"] == "unanswerable"
        rec = None if unanswerable else doc_recall(expected, trace["chunks"])
        score = None if unanswerable else answer_score(res["answer"], r["ground_truth"])
        verdict = judge(agent.llm, r["question"], r["ground_truth"], res["answer"]) if args.judge else ""
        row = {"id": r["id"], "type": r["type"], "difficulty": r["difficulty"], "route": trace["route"],
               "doc_recall": rec, "score": score, "refused": refused, "judge": verdict, "seconds": trace["seconds"],
               "question": r["question"], "ground_truth": r["ground_truth"], "answer": res["answer"],
               "papers": [p["source"] for p in trace["papers"]], "context": res["context"]}
        results.append(row)
        print(f"{r['id']:4} {r['type']:13} {r['difficulty']:8} {trace['route']:6} {fmt(rec):>6} {fmt(score):>6} "
              f"{'yes' if refused else 'no':>8} {trace['seconds']:6.1f} {verdict}")

    print("\n=== Summary ===")
    print(f"{'group':22} {'n':>3} {'doc_recall':>11} {'score':>7} {'refused%':>9}   (unanswerable: refused% should be 100)")
    groups = defaultdict(list)
    for row in results:
        groups[f"type: {row['type']}"].append(row)
        groups[f"difficulty: {row['difficulty']}"].append(row)
    groups["ALL"] = results
    for name, rs in sorted(groups.items()):
        print(f"{name:22} {len(rs):3} {fmt(mean([x['doc_recall'] for x in rs])):>11} "
              f"{fmt(mean([x['score'] for x in rs])):>7} {100 * mean([float(x['refused']) for x in rs]):8.0f}%")

    print("\n=== Look at these ===")
    for row in results:
        problems = []
        if row["type"] == "unanswerable" and not row["refused"]:
            problems.append("should have refused")
        if row["type"] != "unanswerable":
            if row["refused"]:
                problems.append("refused but answerable")
            if row["doc_recall"] is not None and row["doc_recall"] < 1:
                problems.append(f"missing source paper(s), recall {row['doc_recall']:.2f}, papers found: {row['papers']}")
            if row["score"] is not None and row["score"] < 0.5:
                problems.append(f"low score {row['score']:.2f}")
        if problems or args.show:
            print(f"\n{row['id']} [{row['type']}/{row['difficulty']}] {row['question']}")
            for p in problems:
                print(f"  ! {p}")
            print(f"  truth : {row['ground_truth']}")
            print(f"  answer: {row['answer'][:600]}")

    out_dir = config.ROOT / "data" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    cols = ["id", "type", "difficulty", "route", "doc_recall", "score", "refused", "judge", "seconds"]
    with (out_dir / f"dev_{stamp}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    with (out_dir / f"dev_{stamp}.jsonl").open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nSaved data/eval/dev_{stamp}.csv and .jsonl")


if __name__ == "__main__":
    main()