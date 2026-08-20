"""
Tests for EmbeddingCache.

These exist because the cache shipped without them and broke on first read: the
write path worked, and nothing had ever read a value back until an eval run
tried to. pgvector returns its own Vector type rather than a list, which no
amount of reasoning about the write path would have revealed.

A cache is exactly the kind of component where write and read are separate code
paths that can disagree, so both need exercising.

Requires TEST_DATABASE_URL - see tests/test_vector_store.py.
"""

import os

import psycopg
import pytest
from dotenv import load_dotenv

from src.rag.embedding_cache import EmbeddingCache
from src.rag.vector_store import VectorStore

load_dotenv()

DIM = 768
TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", "")


def vec(*leading: float) -> list[float]:
    values = list(leading) + [0.0] * DIM
    return values[:DIM]


@pytest.fixture
def cache() -> EmbeddingCache:
    if not TEST_DATABASE_URL:
        pytest.skip("TEST_DATABASE_URL not set")

    db_name = TEST_DATABASE_URL.rsplit("/", 1)[-1].split("?")[0]
    if "test" not in db_name.lower():
        pytest.skip(f"refusing to write to database '{db_name}'")

    try:
        psycopg.connect(TEST_DATABASE_URL, connect_timeout=3).close()
    except psycopg.OperationalError as exc:
        pytest.skip(f"test database not reachable: {exc}")

    # VectorStore applies the schema, which creates embedding_cache too.
    VectorStore(TEST_DATABASE_URL, embedding_dim=DIM, embedding_model="test-model")

    with psycopg.connect(TEST_DATABASE_URL) as conn:
        conn.execute("TRUNCATE embedding_cache")

    return EmbeddingCache(TEST_DATABASE_URL)


def test_round_trip_returns_a_usable_list(cache: EmbeddingCache) -> None:
    """
    The bug this file exists for: pgvector returns a Vector object, so the
    value that comes back must still be a plain list of floats that callers
    can pass straight into a query.
    """
    original = vec(0.1, -0.2, 0.3)
    cache.put("what is a persistent volume", "m", DIM, "RETRIEVAL_QUERY", original)

    retrieved = cache.get("what is a persistent volume", "m", DIM, "RETRIEVAL_QUERY")

    assert isinstance(retrieved, list)
    assert len(retrieved) == DIM
    assert all(isinstance(x, float) for x in retrieved[:5])
    assert retrieved[:3] == pytest.approx([0.1, -0.2, 0.3], abs=1e-6)


def test_miss_returns_none(cache: EmbeddingCache) -> None:
    assert cache.get("never stored", "m", DIM, "RETRIEVAL_QUERY") is None


def test_task_type_is_part_of_the_key(cache: EmbeddingCache) -> None:
    """
    The same text embedded as a document and as a query produces different
    vectors. A key ignoring task_type would serve one where the other belongs -
    silently degrading retrieval in a way no other test would catch.
    """
    cache.put("storage", "m", DIM, "RETRIEVAL_DOCUMENT", vec(1.0))

    assert cache.get("storage", "m", DIM, "RETRIEVAL_DOCUMENT") is not None
    assert cache.get("storage", "m", DIM, "RETRIEVAL_QUERY") is None


def test_model_and_dimensions_are_part_of_the_key(cache: EmbeddingCache) -> None:
    """Vectors from different models occupy different spaces and are not comparable."""
    cache.put("storage", "model-a", DIM, "RETRIEVAL_QUERY", vec(1.0))

    assert cache.get("storage", "model-b", DIM, "RETRIEVAL_QUERY") is None
    assert cache.get("storage", "model-a", 1536, "RETRIEVAL_QUERY") is None


def test_hit_and_miss_counts_are_tracked(cache: EmbeddingCache) -> None:
    cache.put("cached", "m", DIM, "RETRIEVAL_QUERY", vec(1.0))

    cache.get("cached", "m", DIM, "RETRIEVAL_QUERY")     # hit
    cache.get("cached", "m", DIM, "RETRIEVAL_QUERY")     # hit
    cache.get("absent", "m", DIM, "RETRIEVAL_QUERY")     # miss

    stats = cache.stats
    assert stats["hits"] == 2
    assert stats["misses"] == 1
    assert stats["hit_rate"] == pytest.approx(2 / 3)


def test_put_is_idempotent(cache: EmbeddingCache) -> None:
    """Storing the same key twice updates rather than duplicating."""
    cache.put("same", "m", DIM, "RETRIEVAL_QUERY", vec(1.0))
    cache.put("same", "m", DIM, "RETRIEVAL_QUERY", vec(1.0))

    assert cache.size() == 1
