"""Check the judge against answers whose verdict is known, and print a confusion matrix.

Usage (from the repo root):
  python evaluation/judge_check.py                       # synthetic answers built from dev_set.csv
  python evaluation/judge_check.py --labeled mine.csv    # your hand-labelled answers: id, answer, expected

Synthetic answers: correct = the ground truth, partial = its first half, idk = a refusal, hallucinated = another
question's ground truth. It is a sanity check; hand-labelled answers from real team runs are the better test.
"""
import argparse
import csv
from collections import Counter

import evaluate as ev

DEV_SET = ev.ROOT / "dev_set.csv"
REFUSAL = "I don't know. This is not in the documents."


def synthetic_cases(questions):
    cases = []
    for i, q in enumerate(questions):
        other = questions[(i + 1) % len(questions)]["truth"]
        if q["type"] == "unanswerable":
            cases += [(q, REFUSAL, "idk"), (q, other, "hallucinated")]
            continue
        words = q["truth"].split()
        cases += [(q, q["truth"], "correct"), (q, " ".join(words[:len(words) // 2]), "partial"),
                  (q, REFUSAL, "idk"), (q, other, "hallucinated")]
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-set", default=str(DEV_SET))
    ap.add_argument("--labeled", help="CSV with id, answer, expected (correct|partial|idk|hallucinated)")
    ap.add_argument("--judge-model", default="gpt-4o-mini")
    args = ap.parse_args()

    questions = ev.load_questions(args.test_set)
    if args.labeled:
        by_id = {q["id"]: q for q in questions}
        cases = [(by_id[r["id"]], r["answer"], r["expected"]) for r in csv.DictReader(open(args.labeled, encoding="utf-8"))]
    else:
        cases = synthetic_cases(questions)

    pairs = Counter((exp, ev.judge(q, ans, args.judge_model)) for q, ans, exp in cases)
    labels = ev.LABELS
    print(f"{'expected / judge':<18}" +"".join(f"{l:>14}" for l in labels))
    for exp in ev.JUDGE_LABELS:
        print(f"{exp:<18}" + "".join(f"{pairs[(exp, l)]:>14}" for l in labels))
    agree = sum(n for (e, g), n in pairs.items() if e == g)
    print(f"\nagreement: {agree}/{len(cases)} = {agree / len(cases):.0%}")


if __name__ == "__main__":
    main()
