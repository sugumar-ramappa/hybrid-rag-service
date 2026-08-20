"""
Semantic answer cache - reuse a previous answer when a new question means the
same thing.

WHY THIS AND NOT THE EMBEDDING CACHE
Embedding a question is about 1% of a request's cost. Retrieval and generation
are the other 99%. Caching embeddings saves the cheap part; caching answers
saves the expensive part.

WHY "SEMANTIC" RATHER THAN EXACT MATCH
Users do not repeat themselves byte for byte. "Why is Kubernetes abbreviated as
K8s?" and "What history lies behind the name K8s?" are the same question and
share almost no words, so a hash of the text would miss. Comparing embeddings
catches both.

Note this does NOT avoid embedding the question - the embedding is what we look
up WITH. It avoids retrieval and generation, which is where the cost is.

THE THRESHOLD, MEASURED ON THIS CORPUS

    same question, punctuation/case only    0.0093
    same question, fully reworded           0.0529
    same question, different words again    0.0704
                                            ------
    closest two DIFFERENT questions         0.2110

0.12 sits in that gap. Too high and unrelated questions serve each other's
answers - silently, because a wrong cache hit looks exactly like a right one.
Too low and rephrasings miss, which merely costs money.

Err low. A miss is expensive; a false hit is wrong.
"""

import logging
from dataclasses import dataclass

import psycopg
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

logger = logging.getLogger(__name__)


@dataclass
class CachedAnswer:
    """A previous answer judged close enough to reuse."""
    answer: str
    sources: list
    matched_question: str  # what it actually matched - keep this visible
    distance: float


class AnswerCache:
    """Look up previous answers by meaning."""

    def __init__(self, database_url: str, threshold: float = 0.12) -> None:
        self._url = database_url
        self._threshold = threshold
        self._hits = 0
        self._misses = 0

    def _connect(self) -> psycopg.Connection:
        conn = psycopg.connect(self._url)
        register_vector(conn)
        return conn

    def corpus_version(self) -> str:
        """
        A cheap fingerprint of the corpus.

        Changes whenever chunks are added, removed or re-embedded, which is
        exactly when previously generated answers stop being trustworthy. Not
        cryptographic - it only has to change when the corpus does.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT count(*), coalesce(max(created_at)::text, '') FROM chunks"
            ).fetchone()
        return f"{row[0]}:{row[1]}"

    def lookup(self, embedding: list[float], corpus_version: str) -> CachedAnswer | None:
        """
        Find the nearest previously answered question, if it is close enough.

        Filtered by corpus_version first: an answer generated against a
        different corpus may cite documents that have since changed, and
        serving it would be wrong in a way nothing downstream could detect.
        """
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, question, answer, sources,
                       embedding <=> %s::vector AS distance
                FROM answer_cache
                WHERE corpus_version = %s
                ORDER BY embedding <=> %s::vector
                LIMIT 1
                """,
                (embedding, corpus_version, embedding),
            ).fetchone()

            if row is None or row[4] > self._threshold:
                self._misses += 1
                if row is not None:
                    logger.debug(
                        "Nearest cached question was %.4f away, over the %.2f threshold",
                        row[4], self._threshold,
                    )
                return None

            conn.execute(
                """
                UPDATE answer_cache
                SET last_used_at = now(), hit_count = hit_count + 1
                WHERE id = %s
                """,
                (row[0],),
            )

        self._hits += 1
        logger.info(
            "Answer cache hit at distance %.4f - reusing the answer to: %s",
            row[4], row[1][:70],
        )
        return CachedAnswer(
            answer=row[2],
            sources=row[3],
            matched_question=row[1],
            distance=row[4],
        )

    def store(self, question: str, embedding: list[float], answer: str,
              sources: list, corpus_version: str) -> None:
        """Remember an answer so a similar question can reuse it."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO answer_cache
                    (question, embedding, answer, sources, corpus_version)
                VALUES (%s, %s::vector, %s, %s, %s)
                """,
                (question, embedding, answer, Jsonb(sources), corpus_version),
            )

    def invalidate_stale(self, current_version: str) -> int:
        """
        Drop answers generated against an older corpus.

        Not strictly required - lookup filters by version anyway - but without
        it the table grows forever with rows that can never be hit again.
        """
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM answer_cache WHERE corpus_version <> %s", (current_version,)
            )
            deleted = cur.rowcount
        if deleted:
            logger.info("Dropped %d answers cached against an older corpus", deleted)
        return deleted

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total else 0.0,
        }

    def size(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT count(*) FROM answer_cache").fetchone()
        return row[0] if row else 0
