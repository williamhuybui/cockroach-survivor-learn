"""Grade a benchmark run with the official grader's judge (gpt-4o-mini, same prompt and points).

Usage (from members/team_hieu_trang):
  python scripts/judge.py data/evaluations/smoke_set_<time>.json --dataset smoke_set.csv

Uses evaluation/evaluate.py's judge() and scoring, so verdicts match the dashboard grader.
Needs OPENAI_API_KEY in the environment or in this team's .env; the key is never printed.
Use public question sets only (dev, smoke, unseen); never the hidden test set.
"""

import argparse
import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path

TEAM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TEAM_ROOT.parents[1]


def load_key():
    if os.environ.get("OPENAI_API_KEY"):
        return
    env = TEAM_ROOT / ".env"
    for line in env.read_text().splitlines() if env.exists() else []:
        key, sep, value = line.partition("=")
        if sep and key.strip() == "OPENAI_API_KEY":
            os.environ["OPENAI_API_KEY"] = value.strip().strip("\"'")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="JSON written by scripts/benchmark.py")
    parser.add_argument("--dataset", type=Path, default=REPO_ROOT / "dev_set.csv",
                        help="CSV with the reference answers (default: repo dev_set.csv)")
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    args = parser.parse_args()

    load_key()
    sys.path.insert(0, str(REPO_ROOT / "evaluation"))
    import evaluate as grader

    questions = {q["id"]: q for q in grader.load_questions(args.dataset)}
    rows = json.loads(args.run.read_text(encoding="utf-8"))
    verdicts, total = Counter(), 0.0
    for row in rows:
        question = questions[row["id"]]
        verdict = grader.judge(question, (row.get("answer") or "").strip(), args.judge_model)
        points = (
            grader.score_unanswerable(verdict)
            if question["type"] == "unanswerable"
            else grader.POINTS[verdict]
        )
        row["verdict"], row["points"] = verdict, points
        verdicts[verdict] += 1
        total += points
        print(f"{row['id']}: {verdict:12} {points:+.1f}  {row['seconds']:6.1f}s")
    seconds = [row["seconds"] for row in rows]
    print(f"\nScore {total:.1f} / {len(rows)}  {dict(verdicts)}")
    print(f"Latency mean {statistics.mean(seconds):.1f}s, max {max(seconds):.1f}s, "
          f"under 120s: {sum(s < 120 for s in seconds)}/{len(seconds)}")
    args.run.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
