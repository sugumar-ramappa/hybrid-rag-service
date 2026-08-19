"""
Generate candidate questions for the retrieval golden set.

    python -m scripts.generate_golden_set                  # 60 candidates
    python -m scripts.generate_golden_set --count 40
    python -m scripts.generate_golden_set --out eval/candidates.json

Writes eval/golden_set.json with every candidate marked "verified": false.
NOTHING COUNTS UNTIL YOU SET THAT TO TRUE BY HAND. The eval harness ignores
unverified entries, so skipping the review produces an empty result rather than
a flattering one.

Two design decisions worth understanding:

SAMPLING BY DOCUMENT, NOT BY CHUNK
    The corpus is skewed - one page produces 59 chunks while several produce 1.
    Sampling chunks uniformly would draw ~8% of questions from a single
    document, so the resulting number would largely measure retrieval quality
    on that one page. Sampling a document first, then a chunk within it, gives
    coverage of the corpus instead.

TWO QUESTION STYLES
    exact_term  - phrased using the identifiers a reader would actually type
                  (PersistentVolumeClaim, reclaimPolicy)
    paraphrase  - describes the need in plain language, sharing few or no words
                  with the source text

    Dense retrieval is strong on the second and weak on the first; keyword
    search is the reverse. Labelling each question means recall can be reported
    per style, which shows *where* hybrid search helps rather than just that it
    does.
"""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

from google import genai

from src.config import get_config

logger = logging.getLogger(__name__)

PROMPTS = {
    "exact_term": """You are building an evaluation set for a documentation search system.

Below is one chunk from the Kubernetes documentation. Write ONE question that:
- is answerable ONLY from this specific chunk, not from general Kubernetes knowledge
- uses the exact technical identifiers that appear in the text (API field names,
  resource kinds, flags) - the way an engineer searching for this would type it
- is a single sentence, under 20 words
- does not mention "the chunk", "the text", "the document", or "above"

Return ONLY the question. No preamble, no quotes, no explanation.

CHUNK:
{chunk}""",

    "paraphrase": """You are building an evaluation set for a documentation search system.

Below is one chunk from the Kubernetes documentation. Write ONE question that:
- is answerable ONLY from this specific chunk, not from general Kubernetes knowledge
- describes the problem in PLAIN LANGUAGE, deliberately avoiding the technical
  terms used in the text - as a newcomer who does not yet know the vocabulary
  would phrase it
- is a single sentence, under 20 words
- does not mention "the chunk", "the text", "the document", or "above"

Return ONLY the question. No preamble, no quotes, no explanation.

CHUNK:
{chunk}""",
}

# Chunks shorter than this rarely contain enough to ask a specific question about.
MIN_CHUNK_CHARS = 400


def fetch_candidates(database_url: str, count: int, seed: int) -> list[dict]:
    """
    Pick `count` chunks, sampling documents evenly rather than chunks.

    Done in SQL with a window function: number the chunks within each document,
    shuffle documents, then take the lowest-numbered chunk from each. Documents
    are visited round-robin, so a 59-chunk page contributes no more than a
    1-chunk page until every document has been used once.
    """
    import psycopg

    with psycopg.connect(database_url) as conn:
        conn.execute("SELECT setseed(%s)", (seed / 2**31,))
        rows = conn.execute(
            """
            WITH numbered AS (
                SELECT chunk_id, source, chunk_index, content,
                       row_number() OVER (PARTITION BY source ORDER BY random()) AS pick_order,
                       dense_rank()  OVER (ORDER BY source) AS doc_rank
                FROM chunks
                WHERE length(content) >= %s
            )
            SELECT chunk_id, source, chunk_index, content
            FROM numbered
            ORDER BY pick_order, random()
            LIMIT %s
            """,
            (MIN_CHUNK_CHARS, count),
        ).fetchall()

    return [
        {"chunk_id": r[0], "source": r[1], "chunk_index": r[2], "content": r[3]}
        for r in rows
    ]


def generate_question(client, model: str, chunk_text: str, style: str) -> str | None:
    """One generation call. Returns None if it fails - one bad chunk is not fatal."""
    try:
        response = client.models.generate_content(
            model=model,
            contents=PROMPTS[style].format(chunk=chunk_text),
        )
        question = (response.text or "").strip().strip('"').strip()
        return question or None
    except Exception as exc:  # noqa: BLE001 - keep going, report at the end
        logger.warning("Generation failed: %s", str(exc)[:120])
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate golden set candidates.")
    parser.add_argument("--count", type=int, default=60,
                        help="candidates to generate (default 60; expect to keep ~half)")
    parser.add_argument("--out", default="eval/golden_set.json")
    parser.add_argument("--seed", type=int, default=42,
                        help="sampling seed, so the same corpus gives the same candidates")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = get_config()

    candidates = fetch_candidates(config.database.url, args.count, args.seed)
    if not candidates:
        print("No chunks found. Ingest the corpus first.")
        sys.exit(1)

    print(f"Sampled {len(candidates)} chunks from "
          f"{len({c['source'] for c in candidates})} documents\n")

    client = genai.Client(api_key=config.gemini.api_key)
    questions = []

    for i, chunk in enumerate(candidates, start=1):
        # Alternate styles so both are represented evenly.
        style = "exact_term" if i % 2 else "paraphrase"
        question = generate_question(client, config.gemini.model_name,
                                     chunk["content"], style)
        if not question:
            continue

        questions.append({
            "id": f"q{len(questions) + 1:03d}",
            "question": question,
            "style": style,
            "expected_chunk_id": chunk["chunk_id"],
            "source": chunk["source"],
            "chunk_index": chunk["chunk_index"],
            "chunk_text": chunk["content"],
            "verified": False,
            "notes": "",
        })
        print(f"  [{len(questions):>2}] {style:<11} {question[:76]}")

        # RPM is 15 for the generation model; stay comfortably under it.
        time.sleep(4.5)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "corpus": {
            "chunks_sampled_from": len(candidates),
            "documents": len({c["source"] for c in candidates}),
        },
        "questions": questions,
    }, indent=2) + "\n")

    print(f"\nWrote {len(questions)} candidates to {out_path}")
    print("\nEvery one is marked \"verified\": false. Review each against its")
    print("chunk_text and set verified to true only if:")
    print("  - the chunk genuinely answers it")
    print("  - no OTHER chunk in the corpus answers it as well or better")
    print("  - it is not answerable from general knowledge without the docs")
    print("\nExpect to reject roughly half. That rejection is what makes the")
    print("numbers trustworthy - the eval harness counts only verified entries.")


if __name__ == "__main__":
    main()
