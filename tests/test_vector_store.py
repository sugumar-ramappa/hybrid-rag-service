"""Tests for VectorStore (using ChromaDB locally - no API calls needed)."""

import tempfile
from pathlib import Path

from src.rag.text_splitter import TextChunk
from src.rag.vector_store import VectorStore


def test_add_and_search() -> None:
    """Test adding chunks and searching."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = VectorStore(persist_dir=tmp_dir, collection_name="test_col")

        # Create test chunks with fake embeddings
        chunks = [
            TextChunk(content="Python is great", metadata={"source": "a.txt"}, chunk_index=0),
            TextChunk(content="Java is also great", metadata={"source": "b.txt"}, chunk_index=1),
        ]
        # Simple fake embeddings (3 dimensions)
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]

        count = store.add_chunks(chunks, embeddings)
        assert count == 2
        assert store.get_document_count() == 2

        # Search with a vector close to the first embedding
        results = store.search(query_embedding=[0.9, 0.1, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0].content == "Python is great"


def test_empty_store_search() -> None:
    """Test searching an empty store returns empty list."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = VectorStore(persist_dir=tmp_dir, collection_name="empty_test")
        results = store.search(query_embedding=[1.0, 0.0, 0.0], top_k=5)
        assert results == []


def test_clear_store() -> None:
    """Test clearing the vector store."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = VectorStore(persist_dir=tmp_dir, collection_name="clear_test")

        chunks = [
            TextChunk(content="test", metadata={"source": "a.txt"}, chunk_index=0),
        ]
        store.add_chunks(chunks, [[1.0, 0.0]])

        assert store.get_document_count() == 1
        store.clear()
        assert store.get_document_count() == 0


def test_mismatched_chunks_and_embeddings() -> None:
    """Test that mismatched lengths raise an error."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = VectorStore(persist_dir=tmp_dir, collection_name="mismatch_test")

        chunks = [
            TextChunk(content="test", metadata={}, chunk_index=0),
        ]
        try:
            store.add_chunks(chunks, [[1.0], [2.0]])  # 1 chunk, 2 embeddings
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected


def test_metadata_preserved() -> None:
    """Test that metadata is preserved in search results."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = VectorStore(persist_dir=tmp_dir, collection_name="meta_test")

        chunks = [
            TextChunk(
                content="Test content",
                metadata={"source": "doc.pdf", "page": 5},
                chunk_index=0,
            ),
        ]
        store.add_chunks(chunks, [[1.0, 0.0, 0.0]])

        results = store.search(query_embedding=[1.0, 0.0, 0.0], top_k=1)
        assert results[0].metadata["source"] == "doc.pdf"
        assert results[0].metadata["page"] == 5
