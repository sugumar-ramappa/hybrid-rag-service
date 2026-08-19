"""
Review golden set candidates one at a time.

    python -m scripts.review_golden_set
    python -m scripts.review_golden_set --style paraphrase   # one style only
    python -m scripts.review_golden_set --all                # include decided ones

Shows each question with the chunk it is supposed to retrieve, and any warning
signs already detected, then asks whether it is a fair test.

Saves after every decision, so quitting loses nothing.

No API calls - this is reading and judgement only.
"""

import argparse
import json
import re
import textwrap
from pathlib import Path

from scripts.check_golden_set import leakage, longest_shared_run

WRAP = 88


def warnings_for(q: dict) -> list[str]:
    """Things worth knowing before deciding. Not verdicts - just flags."""
    flags = []
    chunk = q["chunk_text"]

    if "whatsnext" in chunk:
        flags.append("chunk contains a 'What's next' link list")
    if chunk.count("](") >= 4 and len(chunk) < 1200:
        flags.append("chunk is mostly links, little prose")
    if re.search(r"\{\{[%<]", chunk):
        flags.append("chunk contains Hugo template markup")

    overlap = leakage(q["question"], chunk)
    run = longest_shared_run(q["question"], chunk)
    if overlap > 0.75:
        flags.append(f"question reuses {overlap:.0%} of the chunk's vocabulary")
    if run >= 5:
        flags.append(f"question lifts {run} consecutive words from the chunk")

    if len(chunk) < 500:
        flags.append(f"chunk is short ({len(chunk)} chars)")

    return flags


def show(q: dict, position: int, total: int) -> None:
    print("\n" + "=" * WRAP)
    print(f"{q['id']}   [{q['style']}]   {position} of {total}")
    print(f"{q['source']}  ·  chunk {q['chunk_index']}")
    print("=" * WRAP)

    print("\nQUESTION")
    print(textwrap.fill(q["question"], WRAP, initial_indent="  ", subsequent_indent="  "))

    print("\nCHUNK IT SHOULD RETRIEVE")
    body = q["chunk_text"].strip()
    shown = body if len(body) <= 1100 else body[:1100] + f"\n  ... [{len(body) - 1100} more chars]"
    for line in shown.split("\n"):
        print(textwrap.fill(line, WRAP, initial_indent="  ", subsequent_indent="  ") or "")

    flags = warnings_for(q)
    if flags:
        print("\nFLAGS")
        for f in flags:
            print(f"  ! {f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Review golden set candidates.")
    parser.add_argument("--file", default="eval/golden_set.json")
    parser.add_argument("--style", help="review only this style")
    parser.add_argument("--all", action="store_true",
                        help="include entries already marked verified")
    args = parser.parse_args()

    path = Path(args.file)
    data = json.loads(path.read_text())
    questions = data["questions"]

    queue = [q for q in questions
             if (args.all or not q["verified"])
             and (not args.style or q["style"] == args.style)]

    if not queue:
        print("Nothing to review.")
        return

    print(f"\n{len(queue)} to review.\n")
    print("Keep it only if BOTH are true:")
    print("  1. this chunk genuinely answers the question")
    print("  2. no OTHER chunk in the corpus answers it as well or better")
    print("\n  [y] keep   [n] reject   [s] skip   [q] save and quit\n")

    kept = 0
    for i, q in enumerate(queue, start=1):
        show(q, i, len(queue))

        while True:
            choice = input("\n  keep? [y/n/s/q] ").strip().lower()
            if choice in {"y", "n", "s", "q"}:
                break

        if choice == "q":
            break
        if choice == "s":
            continue

        q["verified"] = choice == "y"
        if choice == "y":
            kept += 1
        else:
            note = input("  why? (optional, Enter to skip) ").strip()
            if note:
                q["notes"] = note

        # Save after every decision so quitting never loses work.
        path.write_text(json.dumps(data, indent=2) + "\n")

    total_verified = sum(1 for q in questions if q["verified"])
    print(f"\n{'=' * WRAP}")
    print(f"kept this session : {kept}")
    print(f"verified in total : {total_verified} of {len(questions)}")
    if total_verified < 20:
        print("\nFewer than 20 verified questions makes the metrics noisy -")
        print("a single result swings recall by several points. Aim for 25-30.")
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
