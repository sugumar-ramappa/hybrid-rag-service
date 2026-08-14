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


def test_chunk_indices_are_sequential() -> None:
    """Test that chunk indices are sequential across documents."""
    docs = [
        Document(content="A " * 500, metadata={"source": "a.txt"}),
        Document(content="B " * 500, metadata={"source": "b.txt"}),
    ]

    splitter = TextSplitter(chunk_size=200, chunk_overlap=20)
    chunks = splitter.split_documents(docs)

    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_metadata_includes_chunk_index() -> None:
    """Test that chunk metadata includes chunk_index."""
    doc = Document(content="Some content.", metadata={"source": "test.txt"})

    splitter = TextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents([doc])

    assert "chunk_index" in chunks[0].metadata
    assert chunks[0].metadata["source"] == "test.txt"
