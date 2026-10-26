"""Run a team's submitted service against the hidden test set and update the leaderboard.

Usage (from the repo root):
  python evaluation/evaluate.py team1 --mode practice        # weekly practice run
  python evaluation/evaluate.py team2 --mode final           # Nov 1 final run
  python evaluation/evaluate.py team1 --test-set dev_set.csv --limit 5   # self-check on the public dev set
  python evaluation/evaluate.py team1 --mode final --rescore evaluation/eval_results/<file>.csv
      # re-score after you fixed verdicts by hand in the results CSV

Start the gateway first (uvicorn api_view:app from the repo root); it exposes
POST /<team>/rag-agent {"question": "..."} -> {"answer": "...", "sources": [...]}
and calls the team's agent function defined in api_view.py. The judge is an OpenAI model (default gpt-4o-mini; needs OPENAI_API_KEY).
Per-question results (they contain the hidden answers) go to evaluation/eval_results/ (gitignored);
docs/scores.csv (public, read by the leaderboard page) gets one row per full test-set run: time, date, team, mode, trial, questions, score; the full breakdown stays in eval_results/.
"""
import argparse
import csv
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import requests
from openai import OpenAI

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def load_env(path):
    """Put KEY=value lines from the repo's .env into os.environ (existing variables win)."""
    if path.exists():
        for line in path.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and not key.lstrip().startswith("#"):
                os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


load_env(ROOT / ".env")
TEST_SET = HERE / "test_set.csv"
SCORES = ROOT / "docs" / "scores.csv"
RESULTS = HERE / "eval_results"

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


def load_questions(path=TEST_SET):
    rows = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        # test_set.csv: ID / ground truth / source docs; dev_set.csv: id / ground_truth / source_docs
        r = {k.strip().lower().replace("_", " "): v for k, v in r.items()}
        docs = [d.strip() for d in r["source docs"].split(";") if d.strip()]
        qtype = "unanswerable" if not docs else "multi" if len(docs) > 1 else "single"
        rows.append({"id": r["id"], "question": r["question"], "truth": r["ground truth"],
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
    resp = OpenAI().chat.completions.create(
        model=model, temperature=0, response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}], timeout=120)
    text = resp.choices[0].message.content or ""
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


def read_scores():
    if not SCORES.exists():
        return []
    return list(csv.DictReader(open(SCORES, encoding="utf-8")))


SCORE_FIELDS = ["timestamp", "date", "team", "mode", "trial", "questions", "score"]


def update_board(team_id, run, questions=None):
    """Append one row to docs/scores.csv, which the leaderboard page reads.

    trial = how many full runs this team has had in this mode (this one included). Public file: only time, team,
    mode, trial, number of questions and score, so teams can't infer the test set's makeup.
    """
    rows = read_scores()
    trial = 1 + sum(1 for r in rows if r["team"] == team_id and r["mode"] == run["mode"])
    now = datetime.now().astimezone()
    row = {"timestamp": now.isoformat(timespec="seconds"), "date": now.strftime("%Y-%m-%d"), "team": team_id,
           "mode": run["mode"], "trial": trial, "questions": run["n"], "score": run["score"]}
    new = not SCORES.exists()
    with open(SCORES, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SCORE_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def check_practice_limit(team_id):
    last = [r["timestamp"] for r in read_scores() if r["team"] == team_id and r["mode"] == "practice"]
    if last and datetime.now().astimezone() - datetime.fromisoformat(last[-1]) < timedelta(days=7):
        raise RuntimeError(f"{team_id} already had a practice run on {last[-1][:10]} (once per week).")


def run_evaluation(team_id, ask, mode="practice", test_set=TEST_SET, limit=None,
                   judge_model="gpt-4o-mini", workers=4, log=print):
    """Send every question in the set to ask(question) -> (answer, sources, seconds, error), judge, save, score.

    The leaderboard is only updated for a full run of the hidden test set. Returns (run summary, results CSV path, updated).
    """
    custom_set = Path(test_set).resolve() != TEST_SET
    questions = load_questions(test_set)
    if limit:
        questions = questions[:limit]
    RESULTS.mkdir(exist_ok=True)
    official = not limit and not custom_set
    if mode == "practice" and official:
        check_practice_limit(team_id)

    done, lock = 0, threading.Lock()

    def work(q):  # ask + judge one question; runs in a worker thread
        nonlocal done
        answer, sources, secs, err = ask(q["question"])
        verdict = judge(q, answer, judge_model)
        with lock:
            done += 1
            log(f"[{done}/{len(questions)}] {q['id']} {q['type']:<12} {verdict:<12} {secs:5.1f}s {err}")
        return {**q, "answer": answer, "sources": sources, "seconds": round(secs, 1),
                "verdict": verdict, "error": err, "retrieval_hit": retrieval_hit(q, sources)}

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(work, questions))  # results keep the question order
    out = RESULTS / f"{team_id}_{mode}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    write_results(out, results)
    run = summarize(results, mode)
    if official:
        update_board(team_id, run, questions)
        (RESULTS / f"{out.stem}_summary.json").write_text(json.dumps(run, indent=2))
    return run, out, official


def write_results(out, results):
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "type", "verdict", "question", "ground_truth", "answer", "sources",
                    "retrieval_hit", "seconds", "error"])
        for r in results:
            w.writerow([r["id"], r["type"], r["verdict"], r["question"], r["truth"], r["answer"],
                        ";".join(r["sources"]), r["retrieval_hit"], r["seconds"], r["error"]])


def rescore(team_id, path, mode, test_set):
    by_id = {q["id"]: q for q in load_questions(test_set)}
    results = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        q = by_id[r["id"]]
        results.append({**q, "verdict": r["verdict"], "error": r.get("error", ""),
                        "retrieval_hit": {"True": True, "False": False}.get(r.get("retrieval_hit"))})
    run = summarize(results, mode)
    out = Path(path)
    update_board(team_id, run, list(by_id.values()))
    (RESULTS / f"{out.stem}_summary.json").write_text(json.dumps(run, indent=2))
    return run, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("team", choices=["team1", "team2", "team3"], help="calls <gateway>/<team>/rag-agent")
    ap.add_argument("--gateway", default="http://localhost:8000", help="where api_view.py is running")
    ap.add_argument("--mode", choices=["practice", "final"], default="practice")
    ap.add_argument("--judge-model", default="gpt-4o-mini")
    ap.add_argument("--workers", type=int, default=4, help="questions asked in parallel (1 = one at a time)")
    ap.add_argument("--timeout", type=int, default=120, help="seconds per question")
    ap.add_argument("--limit", type=int, help="only the first N questions (smoke test; not saved to the board)")
    ap.add_argument("--test-set", default=str(TEST_SET), help="questions CSV (default: hidden test_set.csv)")
    ap.add_argument("--rescore", help="results CSV with hand-edited verdicts; skips calling the team")
    args = ap.parse_args()

    if args.rescore:
        run, out = rescore(args.team, args.rescore, args.mode, args.test_set)
        updated = True
    else:
        base = f"{args.gateway.rstrip('/')}/{args.team}"
        try:  # fail fast if the gateway is down or the team has not plugged in an agent
            health = requests.get(f"{base}/health", timeout=10)
        except requests.RequestException as e:
            raise SystemExit(f"gateway not reachable at {args.gateway}: {e}")
        if not health.ok:
            raise SystemExit(f"{args.team} is not ready: {health.status_code} {health.text}")
        try:
            run, out, updated = run_evaluation(
                args.team, lambda q: ask_team(f"{base}/rag-agent", q, args.timeout), args.mode,
                args.test_set, args.limit, args.judge_model, args.workers)
        except RuntimeError as e:
            raise SystemExit(str(e))
    print(json.dumps(run, indent=2))
    print(f"Per-question results: {out}")
    if updated:
        print(f"Score saved to {SCORES}\nNow: git add docs/scores.csv && git commit && git push")
    else:
        print("--limit/--test-set used: leaderboard NOT updated.")


if __name__ == "__main__":
    main()
