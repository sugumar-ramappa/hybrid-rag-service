"""
Tests for the semantic answer cache.

This is the component where a bug is least visible. A wrong cache hit returns a
fluent, confident answer to a question nobody asked - no error, no warning, and
nothing downstream can tell it apart from a correct response.

So the tests focus on the boundary: what must hit, what must NOT hit, and what
must stop hitting when the corpus changes underneath it.

Synthetic vectors are used deliberately - they let the distance between two
"questions" be set exactly, which real embeddings cannot.

Requires TEST_DATABASE_URL.
"""

import math
import os

import psycopg
import pytest
from dotenv import load_dotenv

from src.rag.answer_cache import AnswerCache
from src.rag.vector_store import VectorStore

load_dotenv()

DIM = 768
THRESHOLD = 0.12
TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", "")


def unit(*leading: float) -> list[float]:
    """A DIM-length unit vector, so cosine distances are predictable."""
    values = list(leading) + [0.0] * DIM
    values = values[:DIM]
    norm = math.sqrt(sum(v * v for v in values))
    return [v / norm for v in values]


def at_distance(d: float) -> list[float]:
    """
    A unit vector exactly `d` cosine-distance from unit(1.0).

    cosine distance = 1 - cos(angle), so an angle of arccos(1-d) gives
    exactly the distance we want. That makes threshold behaviour testable
    rather than approximate.
    """
    angle = math.acos(1 - d)
    return unit(math.cos(angle), math.sin(angle))


@pytest.fixture
def cache() -> AnswerCache:
    if not TEST_DATABASE_URL:
        pytest.skip("TEST_DATABASE_URL not set")

    db_name = TEST_DATABASE_URL.rsplit("/", 1)[-1].split("?")[0]
    if "test" not in db_name.lower():
        pytest.skip(f"refusing to write to database '{db_name}'")

    try:
        psycopg.connect(TEST_DATABASE_URL, connect_timeout=3).close()
    except psycopg.OperationalError as exc:
        pytest.skip(f"test database not reachable: {exc}")

    VectorStore(TEST_DATABASE_URL, embedding_dim=DIM, embedding_model="test-model")

    with psycopg.connect(TEST_DATABASE_URL) as conn:
        conn.execute("TRUNCATE answer_cache")

    return AnswerCache(TEST_DATABASE_URL, threshold=THRESHOLD)


def test_identical_question_hits(cache: AnswerCache) -> None:
    v = unit(1.0)
    cache.store("Why is Kubernetes called K8s?", v, "Because of the 8 letters.", [], "v1")

    hit = cache.lookup(v, "v1")

    assert hit is not None
    assert hit.answer == "Because of the 8 letters."
    assert hit.distance == pytest.approx(0.0, abs=1e-6)


def test_close_rephrasing_hits(cache: AnswerCache) -> None:
    """
    A question 0.08 away is a rephrasing - measured on the real corpus,
    rewordings of the same question land between 0.009 and 0.070.
    """
    cache.store("original", unit(1.0), "the answer", [], "v1")

    hit = cache.lookup(at_distance(0.08), "v1")

    assert hit is not None
    assert hit.matched_question == "original"


def test_different_question_does_not_hit(cache: AnswerCache) -> None:
    """
    0.25 is beyond the threshold. On the real corpus the closest two genuinely
    different questions were 0.211 apart, which is why the threshold is 0.12
    and not higher - this is the case that must never produce a hit.
    """
    cache.store("original", unit(1.0), "the answer", [], "v1")

    assert cache.lookup(at_distance(0.25), "v1") is None


def test_threshold_boundary(cache: AnswerCache) -> None:
    """Just inside hits, just outside does not."""
    cache.store("original", unit(1.0), "the answer", [], "v1")

    assert cache.lookup(at_distance(THRESHOLD - 0.01), "v1") is not None
    assert cache.lookup(at_distance(THRESHOLD + 0.01), "v1") is None


def test_corpus_change_invalidates_everything(cache: AnswerCache) -> None:
    """
    The most dangerous failure this guards against: the corpus is re-ingested,
    the documents now say something different, and a cached answer citing the
    old text is served as though still current.
    """
    v = unit(1.0)
    cache.store("question", v, "answer from the old corpus", [], "corpus-v1")

    assert cache.lookup(v, "corpus-v1") is not None      # same corpus, fine
    assert cache.lookup(v, "corpus-v2") is None          # corpus changed, refuse


def test_nearest_match_wins(cache: AnswerCache) -> None:
    """With several candidates inside the threshold, the closest one is used."""
    cache.store("far",  at_distance(0.10), "far answer",  [], "v1")
    cache.store("near", at_distance(0.02), "near answer", [], "v1")

    hit = cache.lookup(unit(1.0), "v1")

    assert hit is not None
    assert hit.answer == "near answer"


def test_invalidate_stale_removes_old_corpus_entries(cache: AnswerCache) -> None:
    cache.store("a", unit(1.0), "x", [], "corpus-v1")
    cache.store("b", unit(0.0, 1.0), "y", [], "corpus-v2")

    assert cache.invalidate_stale("corpus-v2") == 1
    assert cache.size() == 1


def test_hit_and_miss_counts_are_tracked(cache: AnswerCache) -> None:
    cache.store("q", unit(1.0), "a", [], "v1")

    cache.lookup(unit(1.0), "v1")          # hit
    cache.lookup(at_distance(0.5), "v1")   # miss

    stats = cache.stats
    assert stats["hits"] == 1
    assert stats["misses"] == 1


def test_matched_question_is_returned(cache: AnswerCache) -> None:
    """
    Callers must be able to see WHICH question an answer was reused from.
    A wrong hit is silent otherwise - this is what makes it auditable.
    """
    cache.store("What is a Pod?", unit(1.0), "A Pod is...", [], "v1")

    hit = cache.lookup(at_distance(0.05), "v1")

    assert hit.matched_question == "What is a Pod?"
    assert 0 < hit.distance < THRESHOLD
