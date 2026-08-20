"""
Show a saved evaluation run in plain language.

    python -m scripts.show_results                    # list saved runs
    python -m scripts.show_results dense-v2           # one run in detail
    python -m scripts.show_results dense-v2 hybrid-v2 # what changed between two

Reads eval/results/*.json, which scripts/evaluate.py wrote. No database, no
API calls, no quota - the numbers are already on disk.

Answers the question "where does 0.95 come from?" by showing the count it is
made of, and naming every question that failed.
"""

import json
import sys
from pathlib import Path

RESULTS_DIR = Path("eval/results")
TOP_K = 5


def load(label: str) -> dict:
    path = RESULTS_DIR / f"{label}.json"
    if not path.exists():
        available = sorted(p.stem for p in RESULTS_DIR.glob("*.json"))
        sys.exit(f"No run called '{label}'. Available: {', '.join(available)}")
    return json.loads(path.read_text())


def found(outcome: dict) -> bool:
    """Was the right chunk inside the top TOP_K results?"""
    return outcome["rank"] is not None and outcome["rank"] <= TOP_K


def place(outcome: dict) -> str:
    return f"rank {outcome['rank']}" if outcome["rank"] else "not found at all"


def show_one(label: str) -> None:
    data = load(label)
    outcomes = data["outcomes"]
    hits = [o for o in outcomes if found(o)]
    misses = [o for o in outcomes if not found(o)]

    print(f"\n{'=' * 74}")
    print(f"  {label}   ({data['mode']} search, {data['corpus_chunks']} chunks)")
    print("=" * 74)

    print(f"\n  Asked {len(outcomes)} questions. For each one: was the right")
    print(f"  page in the top {TOP_K} results?\n")
    print(f"      YES  {len(hits):>3} questions")
    print(f"      NO   {len(misses):>3} questions")
    print(f"           {'-' * 14}")
    print(f"      score = {len(hits)} / {len(outcomes)} = "
          f"{len(hits) / len(outcomes):.2f}      <-- this is recall@{TOP_K}")

    # Same count, split by how the question was worded.
    for style in sorted({o["style"] for o in outcomes}):
        subset = [o for o in outcomes if o["style"] == style]
        ok = [o for o in subset if found(o)]
        print(f"\n      {style:<12} {len(ok)} / {len(subset)} = "
              f"{len(ok) / len(subset):.2f}")

    if misses:
        print(f"\n  The {len(misses)} question(s) that FAILED:\n")
        for o in misses:
            print(f"    {o['id']}  [{o['style']}]  {place(o)}")
            print(f"          {o['question']}")
            print(f"          wanted: {o['expected']}\n")


def show_diff(before: str, after: str) -> None:
    a = {o["id"]: o for o in load(before)["outcomes"]}
    b = {o["id"]: o for o in load(after)["outcomes"]}

    print(f"\n{'=' * 74}")
    print(f"  What changed:  {before}  ->  {after}")
    print("=" * 74 + "\n")

    fixed, broke, moved = [], [], []
    for qid in sorted(a):
        if a[qid]["rank"] == b[qid]["rank"]:
            continue
        # Only a move across the top-K line changes the score.
        if not found(a[qid]) and found(b[qid]):
            fixed.append(qid)
        elif found(a[qid]) and not found(b[qid]):
            broke.append(qid)
        else:
            moved.append(qid)

    for title, ids in (("FIXED - now in the top 5", fixed),
                       ("BROKE - fell out of the top 5", broke),
                       ("moved, but still in the top 5", moved)):
        if not ids:
            continue
        print(f"  {title}:\n")
        for qid in ids:
            print(f"    {qid}  [{a[qid]['style']}]  "
                  f"{place(a[qid])}  ->  {place(b[qid])}")
            print(f"          {a[qid]['question'][:66]}\n")

    net = len(fixed) - len(broke)
    total = len(a)
    print(f"  Net: {len(fixed)} fixed - {len(broke)} broken = "
          f"{net:+d} question(s)")
    print(f"  So the score moves by {net}/{total} = {net / total:+.2f}\n")


def main() -> None:
    if not RESULTS_DIR.exists():
        sys.exit("No eval/results/ directory. Run scripts.evaluate first.")

    args = sys.argv[1:]

    if not args:
        runs = sorted(p.stem for p in RESULTS_DIR.glob("*.json"))
        print("\n  Saved evaluation runs:\n")
        for r in runs:
            data = json.loads((RESULTS_DIR / f"{r}.json").read_text())
            overall = data["summary"]["overall"]
            print(f"    {r:<14} {data['mode']:<7} "
                  f"recall@5 {overall['recall@5']:.2f}   "
                  f"({overall['n']} questions)")
        print(f"\n  Detail:   python -m scripts.show_results {runs[0]}")
        if len(runs) > 1:
            print(f"  Compare:  python -m scripts.show_results "
                  f"{runs[0]} {runs[1]}")
        print()
    elif len(args) == 1:
        show_one(args[0])
    else:
        show_diff(args[0], args[1])


if __name__ == "__main__":
    main()
