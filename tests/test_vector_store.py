"""
Tests for VectorStore against a real Postgres with pgvector.

These are integration tests, not unit tests - they exercise real SQL, real
indexes and real upsert semantics, which is the whole point. Mocking the
database here would test nothing worth testing.

Start a database first:
    docker run -d --name ragdb -p 5432:5432 \
      -e POSTGRES_PASSWORD=dev -e POSTGRES_DB=ragdb pgvector/pgvector:pg16

WARNING: every test truncates the chunks table. Point DATABASE_URL at a
development database, never a real one.
"""

import os

import psycopg
import pytest
from dotenv import load_dotenv

from src.rag.text_splitter import TextChunk, _make_chunk_id
from src.rag.vector_store import VectorStore

load_dotenv()

DIM = 768
DATABASE_URL = os.getenv("DATABASE_URL", "")


def vec(*leading: float) -> list[float]:
    """A DIM-length vector with the given leading values, zero-padded."""
    values = list(leading) + [0.0] * DIM
    return values[:DIM]


def chunk(content: str, source: str = "test.txt", index: int = 0) -> TextChunk:
    """Build a TextChunk exactly as TextSplitter would."""
    return TextChunk(
        content=content,
        metadata={"source": source, "chunk_index": index},
        chunk_index=index,
        chunk_id=_make_chunk_id(source, index, content),
    )


@pytest.fixture
def store() -> VectorStore:
    """A clean store for each test."""
    if not DATABASE_URL:
        pytest.skip("DATABASE_URL not set")

    # Truncating is destructive, so refuse anything that is not obviously local.
    if not any(host in DATABASE_URL for host in ("localhost", "127.0.0.1")):
        pytest.skip(f"refusing to truncate a non-local database: {DATABASE_URL.split('@')[-1]}")

    # Skip ONLY when the database is unreachable. Catching every exception here
    # would turn real bugs in VectorStore into green "skipped" runs - which is
    # exactly what hid the register_vector bootstrap bug on the first run.
    try:
        psycopg.connect(DATABASE_URL, connect_timeout=3).close()
    except psycopg.OperationalError as exc:
        pytest.skip(f"Postgres not reachable: {exc}")

    s = VectorStore(DATABASE_URL, embedding_dim=DIM, embedding_model="test-model")
    s.clear()
    return s


def test_add_and_search(store: VectorStore) -> None:
    chunks = [
        chunk("Python is great", source="a.txt"),
        chunk("Java is also great", source="b.txt"),
    ]
    embeddings = [vec(1.0, 0.0), vec(0.0, 1.0)]

    assert store.add_chunks(chunks, embeddings) == 2
    assert store.get_document_count() == 2

    results = store.search(query_embedding=vec(0.9, 0.1), top_k=1)
    assert len(results) == 1
    assert results[0].content == "Python is great"


def test_empty_store_search(store: VectorStore) -> None:
    assert store.search(query_embedding=vec(1.0), top_k=5) == []


def test_clear_store(store: VectorStore) -> None:
    store.add_chunks([chunk("test", source="a.txt")], [vec(1.0)])
    assert store.get_document_count() == 1

    store.clear()
    assert store.get_document_count() == 0


def test_mismatched_chunks_and_embeddings(store: VectorStore) -> None:
    with pytest.raises(ValueError, match="Mismatch"):
        store.add_chunks([chunk("test")], [vec(1.0), vec(2.0)])


def test_duplicate_chunk_ids_rejected(store: VectorStore) -> None:
    """Two identical chunks in one batch is a bug, not something to silently absorb."""
    duplicate = chunk("same", source="a.txt", index=0)
    with pytest.raises(ValueError, match="Duplicate chunk IDs"):
        store.add_chunks([duplicate, duplicate], [vec(1.0), vec(1.0)])


def test_metadata_preserved(store: VectorStore) -> None:
    c = chunk("Test content", source="doc.pdf")
    c.metadata["page"] = 5
    store.add_chunks([c], [vec(1.0)])

    results = store.search(query_embedding=vec(1.0), top_k=1)
    assert results[0].metadata["source"] == "doc.pdf"
    assert results[0].metadata["page"] == 5


def test_metadata_filter(store: VectorStore) -> None:
    """The `where` filter existed in the old implementation but was never used."""
    md = chunk("markdown content", source="guide.md")
    md.metadata["type"] = ".md"
    pdf = chunk("pdf content", source="manual.pdf")
    pdf.metadata["type"] = ".pdf"

    store.add_chunks([md, pdf], [vec(1.0, 0.0), vec(0.9, 0.1)])

    results = store.search(query_embedding=vec(1.0, 0.0), top_k=5, where={"type": ".pdf"})
    assert len(results) == 1
    assert results[0].content == "pdf content"


def test_delete_by_source(store: VectorStore) -> None:
    """Removing one document must leave the others intact."""
    store.add_chunks(
        [chunk("doc a", source="a.txt"), chunk("doc b", source="b.txt")],
        [vec(1.0, 0.0), vec(0.0, 1.0)],
    )

    assert store.delete_by_source("a.txt") == 1
    assert store.get_document_count() == 1

    remaining = store.search(query_embedding=vec(0.0, 1.0), top_k=5)
    assert remaining[0].content == "doc b"


def test_reingest_is_idempotent(store: VectorStore) -> None:
    """
    Re-ingesting unchanged content must not duplicate rows.

    This is the payoff of content-derived chunk IDs: the second write targets
    the same primary keys, so it updates in place instead of accumulating.
    """
    chunks = [chunk("stable content", source="a.txt")]
    embeddings = [vec(1.0)]

    store.add_chunks(chunks, embeddings)
    store.add_chunks(chunks, embeddings)
    store.add_chunks(chunks, embeddings)

    assert store.get_document_count() == 1


def test_existing_chunk_ids_reports_what_is_already_embedded(store: VectorStore) -> None:
    stored = chunk("already here", source="a.txt")
    store.add_chunks([stored], [vec(1.0)])

    fresh = chunk("brand new", source="b.txt")
    found = store.existing_chunk_ids(
        [stored.chunk_id, fresh.chunk_id], "test-model", DIM
    )

    assert found == {stored.chunk_id}


def test_existing_chunk_ids_ignores_a_different_model(store: VectorStore) -> None:
    """
    Vectors from another model live in a different space and are not
    comparable, so they must be re-embedded rather than reused.
    """
    stored = chunk("content", source="a.txt")
    store.add_chunks([stored], [vec(1.0)])

    assert store.existing_chunk_ids([stored.chunk_id], "some-other-model", DIM) == set()
    assert store.existing_chunk_ids([stored.chunk_id], "test-model", 1536) == set()


def test_delete_orphans_removes_only_superseded_chunks(store: VectorStore) -> None:
    """Editing a document leaves its old chunks behind unless they are swept."""
    old_version = chunk("version one", source="doc.md")
    other_doc = chunk("untouched", source="other.md")
    store.add_chunks([old_version, other_doc], [vec(1.0, 0.0), vec(0.0, 1.0)])

    # doc.md is re-ingested with different content, so a different chunk_id
    new_version = chunk("version two", source="doc.md")
    store.add_chunks([new_version], [vec(0.5, 0.5)])
    assert store.get_document_count() == 3  # old version still lingering

    deleted = store.delete_orphans({"doc.md"}, {new_version.chunk_id})

    assert deleted == 1
    assert store.get_document_count() == 2
    remaining = {r.content for r in store.search(vec(1.0, 0.0), top_k=10)}
    assert remaining == {"version two", "untouched"}


def test_delete_orphans_leaves_untouched_sources_alone(store: VectorStore) -> None:
    """Ingesting part of a corpus must not delete the rest of it."""
    a = chunk("doc a", source="a.md")
    b = chunk("doc b", source="b.md")
    store.add_chunks([a, b], [vec(1.0, 0.0), vec(0.0, 1.0)])

    # Only a.md was re-ingested this run
    assert store.delete_orphans({"a.md"}, {a.chunk_id}) == 0
    assert store.get_document_count() == 2


def test_dimension_mismatch_is_rejected(store: VectorStore) -> None:
    """Configured dimension must agree with the schema's vector(N) column."""
    with pytest.raises(ValueError, match="Schema declares vector"):
        VectorStore(DATABASE_URL, embedding_dim=1536, embedding_model="test-model")
