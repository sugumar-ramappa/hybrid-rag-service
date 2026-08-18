"""Tests for TextSplitter."""

from src.rag.document_loader import Document
from src.rag.text_splitter import TextSplitter


def test_split_single_document() -> None:
    """Test splitting a document into chunks."""
    # Create a document longer than chunk_size
    content = "This is sentence one. " * 100  # ~2200 chars
    doc = Document(content=content, metadata={"source": "test.txt"})

    splitter = TextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents([doc])

    assert len(chunks) > 1
    # Each chunk should be <= chunk_size (approximately)
    for chunk in chunks:
        assert len(chunk.content) <= 600  # Some tolerance for word boundaries
    # Metadata should be preserved
    assert all(c.metadata["source"] == "test.txt" for c in chunks)


def test_split_short_document() -> None:
    """Test that a short document produces one chunk."""
    doc = Document(content="Short text.", metadata={"source": "short.txt"})

    splitter = TextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents([doc])

    assert len(chunks) == 1
    assert chunks[0].content == "Short text."


def test_split_empty_list() -> None:
    """Test splitting an empty list returns empty list."""
    splitter = TextSplitter()
    chunks = splitter.split_documents([])

    assert chunks == []


def test_chunk_indices_restart_per_document() -> None:
    """Chunk index is a position within its own document, not a global counter."""
    docs = [
        Document(content="A " * 500, metadata={"source": "a.txt"}),
        Document(content="B " * 500, metadata={"source": "b.txt"}),
    ]

    splitter = TextSplitter(chunk_size=200, chunk_overlap=20)
    chunks = splitter.split_documents(docs)

    for source in ("a.txt", "b.txt"):
        indices = [c.chunk_index for c in chunks if c.metadata["source"] == source]
        assert indices == list(range(len(indices))), f"{source} indices should start at 0"


def test_chunk_ids_are_stable_when_another_document_is_added() -> None:
    """
    REGRESSION: chunk IDs must not depend on what else is in the corpus.

    The original implementation numbered chunks with a global counter, so
    ingesting a new document shifted every later chunk's ID. Because the vector
    store upserts by ID, that silently overwrote unrelated documents' chunks.
    """
    doc_a = Document(content="A " * 500, metadata={"source": "a.txt"})
    doc_new = Document(content="N " * 500, metadata={"source": "new.txt"})

    splitter = TextSplitter(chunk_size=200, chunk_overlap=20)

    ids_alone = [c.chunk_id for c in splitter.split_documents([doc_a])]
    # new.txt sorts before a.txt, so it is loaded first and shifts a.txt's position
    ids_after = [
        c.chunk_id
        for c in splitter.split_documents([doc_new, doc_a])
        if c.metadata["source"] == "a.txt"
    ]

    assert ids_alone == ids_after


def test_chunk_ids_are_unique_across_documents() -> None:
    """Two documents must never produce colliding chunk IDs."""
    docs = [
        Document(content="A " * 500, metadata={"source": "a.txt"}),
        Document(content="B " * 500, metadata={"source": "b.txt"}),
    ]

    chunks = TextSplitter(chunk_size=200, chunk_overlap=20).split_documents(docs)
    ids = [c.chunk_id for c in chunks]

    assert len(set(ids)) == len(ids)


def test_identical_pdf_pages_do_not_collide() -> None:
    """Same source, same text, different page must still be distinct chunks."""
    docs = [
        Document(content="Same text.", metadata={"source": "doc.pdf", "page": 1}),
        Document(content="Same text.", metadata={"source": "doc.pdf", "page": 2}),
    ]

    chunks = TextSplitter().split_documents(docs)

    assert chunks[0].chunk_id != chunks[1].chunk_id


def test_metadata_includes_chunk_index() -> None:
    """Test that chunk metadata includes chunk_index."""
    doc = Document(content="Some content.", metadata={"source": "test.txt"})

    splitter = TextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents([doc])

    assert "chunk_index" in chunks[0].metadata
    assert chunks[0].metadata["source"] == "test.txt"
