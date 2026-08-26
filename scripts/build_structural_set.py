"""Derive a large evaluation set from the corpus's own structure.

    python -m scripts.build_structural_set --out eval/structural_set.json

WHY THIS EXISTS
The hand-verified golden set has 43 questions. At that size `recall@5 = 1.000`
is 43 of 43, and one awkward new question drops it to 0.977 - so a two-point
difference between configurations is one question, which is noise. Every
conclusion drawn from 43 questions carries that caveat.

The usual fix is a public benchmark with human relevance judgements. This does
something different, because the corpus already contains the labels:

    ## Reclaiming
    When a user is done with their volume, they can delete the PVC...

A heading names a topic. The prose beneath it is, by construction, the passage
that covers that topic. So `(heading, the chunk containing that prose)` is a
labelled retrieval pair - and there are 1,816 of them in this corpus, obtained
with no annotation, no LLM and no quota.

WHAT THIS MEASURES, AND WHAT IT DOES NOT
Being honest about the difference matters more than the sample size.

A heading is a *topic label*, not a user question. "Configuring the kubelet" is
not how anyone asks; they ask "how do I change kubelet settings". So this set
measures **topical retrieval** - can the system find the section a topic belongs
to - which is related to but not the same as question answering.

That makes the two sets complementary rather than competing:

    43 hand-written questions   real phrasing, real intent, too small to trust
    1,816 structural pairs      thin phrasing, no intent, large enough to trust

A configuration that wins on both is winning for real. One that wins on the
small set only is probably winning on noise, and this set is how you find out.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

# Headings that name a document's furniture rather than a topic. A query of
# "Overview" is not a retrieval test - every document has one, so there is no
# correct answer to find.
GENERIC = {
    "overview", "example", "examples", "introduction", "summary", "next steps",
    "what's next", "whats next", "see also", "before you begin", "feedback",
    "references", "reference", "prerequisites", "notes", "note", "usage",
    "syntax", "options", "synopsis", "description", "see", "further reading",
    "objectives", "cleaning up", "clean up", "conclusion", "limitations",
    "troubleshooting", "faq", "api reference", "configuration", "installation",
}

HEADING = re.compile(r"^(#{2,4})\s+(.+?)\s*$", re.M)

# Hugo shortcodes, front matter and link lists. Present in this corpus at a
# level worth stripping: 7% of it is build markup and 29 of 56 documents end
# with a list of links, neither of which is prose that answers anything.
NOISE = re.compile(r"\{\{[<%].*?[>%]\}\}|^\s*[-*]\s*\[.*?\]\(.*?\)\s*$", re.M | re.S)

# Hugo explicit-anchor syntax: "## kube-proxy (optional) {#kube-proxy}". The
# anchor is build metadata, not part of the topic, and leaving it in makes the
# query contain its own answer's slug - a leak, and one that would flatter
# keyword search specifically.
ANCHOR = re.compile(r"\s*\{#[^}]*\}\s*$")


@dataclass
class Pair:
    id: str
    question: str          # the heading, used as a topical query
    source: str            # path relative to the corpus root
    chunk_index: int       # which chunk of that source holds the section
    style: str             # "topic" or "topic_question"
    heading_level: int
    body_chars: int        # how much prose sat under the heading
    # NOT "verified". These pairs were derived from document structure and never
    # read by a person. Calling them verified would let a 301-pair set inherit
    # the credibility of the 43 that actually were hand-checked. Last in the
    # record because a defaulted field cannot precede a non-defaulted one.
    provenance: str = "structural"


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", NOISE.sub(" ", text)).strip()


def sections(markdown: str) -> list[tuple[int, str, str]]:
    """Split into (level, heading, body) triples, body being prose only."""
    out = []
    matches = list(HEADING.finditer(markdown))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        body = clean(markdown[m.end():end])
        out.append((len(m.group(1)), m.group(2).strip(), body))
    return out


def usable(heading: str, body: str, min_words: int, min_body: int) -> bool:
    """Reject headings that cannot function as a retrieval query.

    Three rejections, each for a reason that would otherwise corrupt the set:

    - **generic furniture** - "Overview" appears in every document, so no single
      chunk is the right answer
    - **too short** - a one or two word heading is a keyword, and matching a
      keyword is not the retrieval behaviour under test
    - **too little prose** - a heading followed by a code block or a table has
      no text for the section to be identified by, so the pair cannot be
      grounded to a chunk with any confidence
    """
    # A heading that is itself a Hugo shortcode is build markup, not a topic.
    # `{{% heading "whatsnext" %}}` appeared 32 times in a 301-pair set - 11%
    # of it - as 32 identical queries pointing at 32 different answers, which
    # can only ever score as misses. The noise filter stripped shortcodes from
    # the BODY and never from the heading, so they passed straight through.
    if "{{" in heading or "}}" in heading:
        return False

    words = heading.split()
    if len(words) < min_words:
        return False
    if heading.strip().lower().rstrip("?:").strip() in GENERIC:
        return False
    if len(body) < min_body:
        return False
    # A heading that is itself a code identifier is a lookup, not a question.
    if heading.startswith("`") and heading.endswith("`"):
        return False
    return True


def ground_to_chunk(body: str, chunks: list[tuple[int, str]]) -> int | None:
    """Which chunk holds this section's prose.

    Matched on a distinctive slice of the body rather than on the heading,
    because a heading is often repeated across a document while its opening
    sentence is not. Long enough to be unique, short enough to survive the
    chunker having split mid-paragraph.

    Returns None when nothing matches - which is the honest outcome for a
    section whose prose was reshaped by chunking, and those pairs are dropped
    rather than guessed at.
    """
    for probe_len in (120, 80, 50):
        probe = body[:probe_len].strip()
        if len(probe) < 30:
            continue
        for index, content in chunks:
            if probe in content:
                return index
        # Try a slice from further in: the chunker may have cut the opening.
        mid = body[probe_len:probe_len * 2].strip()
        if len(mid) >= 40:
            for index, content in chunks:
                if mid in content:
                    return index
    return None


def load_chunks(database_url: str) -> dict[str, list[tuple[int, str]]]:
    import psycopg

    by_source: dict[str, list[tuple[int, str]]] = {}
    with psycopg.connect(database_url) as conn:
        for source, index, content in conn.execute(
                "select metadata->>'source', (metadata->>'chunk_index')::int, content "
                "from chunks order by 1, 2"):
            by_source.setdefault(source, []).append((index, content))
    return by_source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="documents")
    parser.add_argument("--out", default="eval/structural_set.json")
    parser.add_argument("--min-heading-words", type=int, default=3)
    parser.add_argument("--min-body-chars", type=int, default=200)
    parser.add_argument("--limit", type=int, default=0, help="0 = no limit")
    args = parser.parse_args()

    from src.config import get_config
    chunks_by_source = load_chunks(get_config().database.url)
    print(f"  {sum(len(v) for v in chunks_by_source.values())} chunks across "
          f"{len(chunks_by_source)} sources")

    root = Path(args.corpus)
    if not root.is_dir():
        print(f"  no corpus at {root}")
        return 1

    pairs: list[Pair] = []
    stats = {"headings": 0, "rejected_unusable": 0, "ungrounded": 0, "no_chunks": 0}

    for path in sorted(root.rglob("*.md")):
        rel = str(path.relative_to(root))
        chunks = chunks_by_source.get(rel)
        if not chunks:
            # The ingested source key may carry a prefix; try a suffix match.
            match = [k for k in chunks_by_source if k.endswith(rel)]
            if not match:
                stats["no_chunks"] += 1
                continue
            rel = match[0]
            chunks = chunks_by_source[rel]

        for level, heading, body in sections(path.read_text(errors="replace")):
            stats["headings"] += 1
            heading = ANCHOR.sub("", heading).strip()
            if not usable(heading, body, args.min_heading_words, args.min_body_chars):
                stats["rejected_unusable"] += 1
                continue
            index = ground_to_chunk(body, chunks)
            if index is None:
                stats["ungrounded"] += 1
                continue
            pairs.append(Pair(
                id=f"s{len(pairs) + 1:04d}",
                question=heading,
                source=rel,
                chunk_index=index,
                style="topic_question" if re.match(
                    r"^(how|what|why|when|where|can|does|should|is|do)\b",
                    heading, re.I) else "topic",
                heading_level=level,
                body_chars=len(body),
            ))
            if args.limit and len(pairs) >= args.limit:
                break
        if args.limit and len(pairs) >= args.limit:
            break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "generator": "structural (headings as topical queries)",
        "corpus": str(root),
        "note": "Ground truth derived from document structure, not human labels. "
                "Measures topical retrieval, not question answering - see the "
                "module docstring.",
        "stats": stats,
        "questions": [asdict(p) for p in pairs],
    }, indent=1))

    print(f"\n  headings seen        {stats['headings']}")
    print(f"  rejected as unusable {stats['rejected_unusable']}")
    print(f"  could not ground     {stats['ungrounded']}")
    print(f"  sources with no chunks {stats['no_chunks']}")
    print(f"\n  PAIRS WRITTEN        {len(pairs)}  ->  {out}")

    by_style: dict[str, int] = {}
    for p in pairs:
        by_style[p.style] = by_style.get(p.style, 0) + 1
    for style, count in sorted(by_style.items()):
        print(f"    {style:16} {count}")

    if pairs:
        print("\n  three examples:")
        for p in pairs[:3]:
            print(f"    {p.id}  {p.question[:56]!r}")
            print(f"          -> {p.source}#{p.chunk_index}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
