"""Run a team's submitted service against the hidden test set and update the leaderboard.

Usage (from the repo root):
  python Huy_code/evaluate.py team1 --mode practice        # weekly practice run
  python Huy_code/evaluate.py team2 --mode final           # Nov 1 final run
  python Huy_code/evaluate.py team1 --mode final --rescore Huy_code/eval_results/<file>.csv
      # re-score after you fixed verdicts by hand in the results CSV

Each team's service must expose POST <url> {"question": "..."} -> {"answer": "...", "sources": [...]}.
URLs live in Huy_code/teams.json. The judge is a local Ollama model (default gemma3).
Per-question results (they contain the hidden answers) go to Huy_code/eval_results/ (gitignored);
docs/leaderboard.json (public) gets only team, score and date; the full breakdown stays in eval_results/.
"""
import argparse
import csv
import json
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TEST_SET = HERE / "test_set.csv"
TEAMS = HERE / "teams.json"
BOARD = ROOT / "docs" / "leaderboard.json"
RESULTS = HERE / "eval_results"
OLLAMA = "http://localhost:11434/api/chat"

POINTS = {"correct": 1.0, "partial": 0.5, "idk": 0.0, "hallucinated": -0.5}
LABELS = list(POINTS)

JUDGE_PROMPT = """You are grading an answer from a question-answering system.

Question: {question}
Ground truth: {truth}
{note}
System answer: {answer}

Pick exactly one verdict:
- correct: the answer contains the key facts of the ground truth and nothing contradicting it
- partial: the answer is right but incomplete or vague, missing an important part of the ground truth
- idk: the system declines to answer, e.g. "I don't know" or "not in the documents"
- hallucinated: the answer is wrong, made up, contradicts the ground truth, or is unrelated

Reply with JSON only: {{"verdict": "correct|partial|idk|hallucinated"}}"""

UNANSWERABLE_NOTE = ("This question is UNANSWERABLE from the documents. If the system gives any "
                     "answer at all, the verdict is hallucinated. Only an explicit statement that it "
                     "is not in the documents counts as idk.")


def load_questions():
    rows = []
    for r in csv.DictReader(open(TEST_SET, encoding="utf-8")):
        docs = [d.strip() for d in r["source docs"].split(";") if d.strip()]
        qtype = "unanswerable" if not docs else "multi" if len(docs) > 1 else "single"
        rows.append({"id": r["ID"], "question": r["question"], "truth": r["ground truth"],
                     "docs": docs, "type": qtype})
    return rows


def ask_team(url, question, timeout):
    t0 = time.time()
    try:
        resp = requests.post(url, json={"question": question}, timeout=timeout)
        resp.raise_for_status()
        body = resp.json()
        return str(body.get("answer", "")).strip(), body.get("sources") or [], time.time() - t0, ""
    except Exception as e:  # a crashed or slow answer scores as "idk" and is flagged
        return "", [], time.time() - t0, f"{type(e).__name__}: {e}"


def judge(q, answer, model):
    if not answer:
        return "idk"
    prompt = JUDGE_PROMPT.format(
        question=q["question"], truth=q["truth"], answer=answer,
        note=UNANSWERABLE_NOTE if q["type"] == "unanswerable" else "")
    resp = requests.post(OLLAMA, json={
        "model": model, "stream": False, "format": "json",
        "options": {"temperature": 0},
        "messages": [{"role": "user", "content": prompt}]}, timeout=300)
    resp.raise_for_status()
    text = resp.json()["message"]["content"]
    try:
        verdict = json.loads(text)["verdict"]
    except (ValueError, KeyError):
        m = re.search("|".join(LABELS), text)
        verdict = m.group(0) if m else "hallucinated"
    return verdict if verdict in POINTS else "hallucinated"


def retrieval_hit(q, sources):
    if not q["docs"] or not sources:
        return None
    got = " ".join(sources).lower()
    return all(Path(d).stem.lower()[:12] in got for d in q["docs"])


def score_unanswerable(verdict):
    # unanswerable: correct abstention +1, any answer -0.5
    return 1.0 if verdict == "idk" else -0.5


def summarize(results, mode):
    by_type, counts = {}, dict.fromkeys(LABELS, 0)
    total = 0.0
    for r in results:
        pts = score_unanswerable(r["verdict"]) if r["type"] == "unanswerable" else POINTS[r["verdict"]]
        total += pts
        counts[r["verdict"]] += 1
        t = by_type.setdefault(r["type"], {"score": 0.0, "max": 0, "n": 0})
        t["score"] += pts
        t["max"] += 1
        t["n"] += 1
    hits = [h for h in (r.get("retrieval_hit") for r in results) if h is not None]
    return {
        "date": datetime.now().strftime("%Y-%m-%d"), "mode": mode,
        "score": total, "max": len(results), "n": len(results), "counts": counts,
        "by_type": by_type,
        "retrieval_hit_rate": sum(hits) / len(hits) if hits else None,
        "errors": sum(1 for r in results if r.get("error")),
    }


def update_board(team_id, run, questions):
    board = json.loads(BOARD.read_text())
    team = next(t for t in board["teams"] if t["id"] == team_id)
    # public file: score and date only, so teams can't infer the test set's size or makeup
    team["runs"].append({"date": run["date"], "mode": run["mode"], "score": run["score"]})
    board["updated"] = datetime.now().astimezone().isoformat(timespec="seconds")
    board.pop("test_counts", None)
    BOARD.write_text(json.dumps(board, indent=2) + "\n")
    return board


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("team", help="team id from teams.json, e.g. team1")
    ap.add_argument("--mode", choices=["practice", "final"], default="practice")
    ap.add_argument("--judge-model", default="gemma3")
    ap.add_argument("--timeout", type=int, default=120, help="seconds per question")
    ap.add_argument("--limit", type=int, help="only the first N questions (smoke test; not saved to the board)")
    ap.add_argument("--rescore", help="results CSV with hand-edited verdicts; skips calling the team")
    ap.add_argument("--force", action="store_true", help="ignore the once-per-week practice limit")
    args = ap.parse_args()

    questions = load_questions()
    if args.limit:
        questions = questions[:args.limit]
    RESULTS.mkdir(exist_ok=True)

    if args.mode == "practice" and not args.force and not args.limit and BOARD.exists():
        team = next(t for t in json.loads(BOARD.read_text())["teams"] if t["id"] == args.team)
        last = [r["date"] for r in team["runs"] if r["mode"] == "practice"]
        if last and datetime.now() - datetime.strptime(last[-1], "%Y-%m-%d") < timedelta(days=7):
            raise SystemExit(f"{args.team} already had a practice run on {last[-1]} (once per week). Use --force to override.")

    if args.rescore:
        by_id = {q["id"]: q for q in questions}
        results = []
        for r in csv.DictReader(open(args.rescore, encoding="utf-8")):
            q = by_id[r["id"]]
            results.append({**q, "verdict": r["verdict"], "error": r.get("error", ""),
                            "retrieval_hit": {"True": True, "False": False}.get(r.get("retrieval_hit"))})
        out = Path(args.rescore)
    else:
        url = json.loads(TEAMS.read_text())[args.team]["url"]
        results = []
        for i, q in enumerate(questions, 1):
            answer, sources, secs, err = ask_team(url, q["question"], args.timeout)
            verdict = judge(q, answer, args.judge_model)
            results.append({**q, "answer": answer, "sources": sources, "seconds": round(secs, 1),
                            "verdict": verdict, "error": err, "retrieval_hit": retrieval_hit(q, sources)})
            print(f"[{i}/{len(questions)}] {q['id']} {q['type']:<12} {verdict:<12} {secs:5.1f}s {err}")
        out = RESULTS / f"{args.team}_{args.mode}_{datetime.now():%Y%m%d_%H%M%S}.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "type", "verdict", "question", "ground_truth", "answer", "sources",
                        "retrieval_hit", "seconds", "error"])
            for r in results:
                w.writerow([r["id"], r["type"], r["verdict"], r["question"], r["truth"], r["answer"],
                            ";".join(r["sources"]), r["retrieval_hit"], r["seconds"], r["error"]])

    run = summarize(results, args.mode)
    print(json.dumps(run, indent=2))
    print(f"Per-question results: {out}")
    if args.limit:
        print("--limit used: leaderboard NOT updated.")
        return
    update_board(args.team, run, questions)
    (RESULTS / f"{out.stem}_summary.json").write_text(json.dumps(run, indent=2))
    print(f"Leaderboard updated: {BOARD}\nNow: git add docs/leaderboard.json && git commit && git push")


if __name__ == "__main__":
    main()
