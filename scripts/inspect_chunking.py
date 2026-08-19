"""
Show exactly how a document gets chunked, without touching the database.

    python -m scripts.inspect_chunking walkthrough/storage-guide.md
    python -m scripts.inspect_chunking documents/concepts/storage/volumes.md

Answers the questions that are otherwise guesswork:
  - what blocks does the separator hierarchy produce, and how big are they?
  - how are those blocks packed into chunks?
  - how much text actually overlaps between consecutive chunks?
  - which chunk IDs would this produce?

No API calls, no database - pure local computation, so it is free to run as
often as you like while tuning CHUNK_SIZE and CHUNK_OVERLAP.
"""

import sys
from pathlib import Path

from src.config import get_config
from src.rag.document_loader import Document
from src.rag.text_splitter import TextSplitter


def shared_suffix_prefix(a: str, b: str) -> int:
    """How many characters of a's tail are literally repeated at the start of b."""
    for n in range(min(len(a), len(b)), 0, -1):
        if a[-n:] == b[:n]:
            return n
    return 0


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.is_file():
        print(f"Not a file: {path}")
        sys.exit(1)

    config = get_config()
    size = config.rag.chunk_size
    overlap = config.rag.chunk_overlap

    text = path.read_text(encoding="utf-8")

    print(f"file          : {path}")
    print(f"characters    : {len(text)}")
    print(f"CHUNK_SIZE    : {size}")
    print(f"CHUNK_OVERLAP : {overlap}")

    # ---------------------------------------------------------------- blocks --
    # The splitter tries separators in order, starting with the paragraph break.
    # Whatever this produces is the smallest unit it can move around - overlap
    # can never be finer-grained than these.
    blocks = text.split("\n\n")
    print(f"\nBLOCKS after splitting on a blank line ({len(blocks)}):\n")
    for i, block in enumerate(blocks):
        head = block.strip().split("\n")[0][:56]
        flag = "  <- exceeds CHUNK_SIZE, will be split further" if len(block) > size else ""
        print(f"  {i:>2}  {len(block):>5} chars  {head}{flag}")

    # ---------------------------------------------------------------- chunks --
    splitter = TextSplitter(chunk_size=size, chunk_overlap=overlap)
    doc = Document(content=text, metadata={"source": path.name})
    chunks = splitter.split_documents([doc])

    print(f"\nCHUNKS produced ({len(chunks)}):\n")
    for chunk in chunks:
        head = chunk.content.strip().split("\n")[0][:48]
        print(f"  {chunk.chunk_index:>2}  {len(chunk.content):>5} chars  "
              f"{chunk.chunk_id}  {head}")

    # --------------------------------------------------------------- overlap --
    print("\nACTUAL OVERLAP between consecutive chunks:\n")
    for a, b in zip(chunks, chunks[1:]):
        n = shared_suffix_prefix(a.content, b.content)
        shared = b.content[:n].replace("\n", " ")[:44]
        pct = f"{n / overlap * 100:.0f}%" if overlap else "n/a"
        print(f"  {a.chunk_index} -> {b.chunk_index}:  {n:>4} chars "
              f"({pct} of the {overlap} budget)  {shared!r}")

    total = sum(len(c.content) for c in chunks)
    print(f"\n  sum of chunk lengths : {total}")
    print(f"  original length      : {len(text)}")
    print(f"  duplicated by overlap: {total - len(text)}")

    print(
        "\nCHUNK_OVERLAP is a ceiling, not a target. The splitter carries back "
        "\nwhole blocks only - if the next block back would exceed the budget, it "
        "\nstops, even a long way short."
    )


if __name__ == "__main__":
    main()
