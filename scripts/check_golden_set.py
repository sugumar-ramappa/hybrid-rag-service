"""
Check golden set questions for leakage before spending time reviewing them.

    python -m scripts.check_golden_set
    python -m scripts.check_golden_set --regenerate exact_term

A question that reuses its chunk's wording does not test retrieval - it tests
string matching. Both dense and keyword search find it trivially, so it inflates
recall while proving nothing. Worth catching mechanically before hand-review,
because it is a property of how the question was generated, not a judgement call.

No API calls unless --regenerate is passed.
"""

import argparse
import json
import re
import time
from pathlib import Path

STOPWORDS = set(
    "a an the is are was were be been being do does did doing how what when "
    "where which who whom why can could will would shall should may might must "
    "i my me you your yours it its this that these those to of in on for with "
    "and or but if not no nor as at by from about into over after before "
    "there here have has had".split()
)


def content_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9_.\-]+", text.lower()) if w not in STOPWORDS]


def leakage(question: str, chunk: str) -> float:
    """Fraction of the question's content words that also appear in the chunk."""
    qw = set(content_words(question))
    if not qw:
        return 0.0
    return len(qw & set(content_words(chunk))) / len(qw)


def longest_shared_run(question: str, chunk: str) -> int:
    """Longest run of consecutive words the question lifts verbatim from the chunk."""
    q = re.findall(r"[a-z0-9]+", question.lower())
    c = " ".join(re.findall(r"[a-z0-9]+", chunk.lower()))
    best = 0
    for start in range(len(q)):
        for end in range(start + best + 1, len(q) + 1):
            if " ".join(q[start:end]) in c:
                best = end - start
            else:
                break
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Check golden set questions for leakage.")
    parser.add_argument("--file", default="eval/golden_set.json")
    parser.add_argument("--regenerate", metavar="STYLE",
                        help="regenerate questions of this style in place (costs API calls)")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="flag questions whose vocabulary overlap exceeds this")
    args = parser.parse_args()

    path = Path(args.file)
    data = json.loads(path.read_text())
    questions = data["questions"]

    scored = [(leakage(q["question"], q["chunk_text"]), q) for q in questions]

    print(f"{len(questions)} questions in {path}\n")

    for style in sorted({q["style"] for q in questions}):
        vals = [s for s, q in scored if q["style"] == style]
        flagged = sum(1 for v in vals if v > args.threshold)
        print(f"  {style:<12} mean overlap {sum(vals) / len(vals):>4.0%}   "
              f"{flagged}/{len(vals)} over {args.threshold:.0%}")

    print("\nMost verbatim:")
    for score, q in sorted(scored, key=lambda t: -t[0])[:5]:
        run = longest_shared_run(q["question"], q["chunk_text"])
        print(f"  {score:>4.0%} overlap, {run} words lifted  [{q['style']}]")
        print(f"        {q['question'][:88]}")

    print("\nLeast verbatim (these are the good ones):")
    for score, q in sorted(scored, key=lambda t: t[0])[:5]:
        print(f"  {score:>4.0%} overlap  [{q['style']}]  {q['question'][:78]}")

    if not args.regenerate:
        print("\nRe-run with --regenerate STYLE to rewrite one style in place.")
        return

    # ------------------------------------------------------------ regenerate --
    from google import genai

    from src.config import get_config
    from scripts.generate_golden_set import PROMPTS, generate_question

    config = get_config()
    client = genai.Client(api_key=config.gemini.api_key)
    targets = [q for q in questions if q["style"] == args.regenerate]

    print(f"\nRegenerating {len(targets)} '{args.regenerate}' questions...\n")
    replaced = 0

    for q in targets:
        new = generate_question(client, config.gemini.model_name,
                                q["chunk_text"], args.regenerate)
        if new:
            before = leakage(q["question"], q["chunk_text"])
            after = leakage(new, q["chunk_text"])
            q["question"] = new
            q["verified"] = False  # a rewritten question is unreviewed again
            replaced += 1
            print(f"  {before:>4.0%} -> {after:>4.0%}  {new[:74]}")
        time.sleep(4.5)  # generation model allows 15 rpm

    path.write_text(json.dumps(data, indent=2) + "\n")

    after_vals = [leakage(q["question"], q["chunk_text"])
                  for q in questions if q["style"] == args.regenerate]
    print(f"\nReplaced {replaced}. New mean overlap for '{args.regenerate}': "
          f"{sum(after_vals) / len(after_vals):.0%}")


if __name__ == "__main__":
    main()
