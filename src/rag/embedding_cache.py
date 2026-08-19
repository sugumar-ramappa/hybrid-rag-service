"""
Postgres-backed cache for query embeddings.

WHY THIS EXISTS
The eval harness runs the same questions against every retrieval configuration.
Without a cache, each run re-embeds text that has not changed - 42 requests of a
1,000/day free-tier quota, plus the wall-clock cost of a 100/minute rate limit.
With it, run one pays and every run after is free.

WHY POSTGRES AND NOT A FILE
A local file cache dies with the container and is not shared between instances.
The database is already there, already persistent, and already shared. Redis
would be faster, but the lookup is dominated by the API call it avoids, so the
extra service buys latency nobody would notice.

The interface is deliberately two methods, so swapping the backend later is a
new class rather than a refactor.
"""

import hashlib
import logging

import psycopg
from pgvector.psycopg import register_vector

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Look up previously computed embeddings by exact text."""

    def __init__(self, database_url: str) -> None:
        self._url = database_url
        self._hits = 0
        self._misses = 0

    @staticmethod
    def key(text: str, model: str, dimensions: int, task_type: str) -> str:
        """
        Identity of a cached embedding.

        All four inputs matter. The same text embedded by a different model, at
        a different size, or under a different task type produces a different
        vector - and vectors from different models are not comparable at all. A
        key on text alone would silently serve incompatible values after any
        config change.
        """
        raw = f"{model}:{dimensions}:{task_type}:{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _connect(self) -> psycopg.Connection:
        conn = psycopg.connect(self._url)
        register_vector(conn)
        return conn

    def get(self, text: str, model: str, dimensions: int, task_type: str) -> list[float] | None:
        """Return a cached embedding, or None. Records the hit for reporting."""
        cache_key = self.key(text, model, dimensions, task_type)

        with self._connect() as conn:
            row = conn.execute(
                "SELECT embedding FROM embedding_cache WHERE cache_key = %s",
                (cache_key,),
            ).fetchone()

            if row is None:
                self._misses += 1
                return None

            # Touch it, so LRU eviction and hit-rate reporting have data.
            conn.execute(
                """
                UPDATE embedding_cache
                SET last_used_at = now(), hit_count = hit_count + 1
                WHERE cache_key = %s
                """,
                (cache_key,),
            )

        self._hits += 1
        return list(row[0])

    def put(self, text: str, model: str, dimensions: int, task_type: str,
            embedding: list[float]) -> None:
        """Store an embedding. Overwrites silently - the key already pins identity."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO embedding_cache
                    (cache_key, embedding, model, dimensions, task_type)
                VALUES (%s, %s::vector, %s, %s, %s)
                ON CONFLICT (cache_key) DO UPDATE SET
                    embedding    = EXCLUDED.embedding,
                    last_used_at = now()
                """,
                (self.key(text, model, dimensions, task_type), embedding,
                 model, dimensions, task_type),
            )

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
            row = conn.execute("SELECT count(*) FROM embedding_cache").fetchone()
        return row[0] if row else 0
