"""
Vector Store - stores and searches document chunks in Postgres with pgvector.

WHY POSTGRES INSTEAD OF A DEDICATED VECTOR DATABASE?
Because retrieval needs two kinds of search, not one:

  - semantic  ("how do I keep data after a pod restarts")  -> vector similarity
  - keyword   ("PersistentVolumeClaim")                    -> exact term match

Embeddings are weak at exact technical terms; keyword search is weak at
paraphrase. Postgres holds both in the same row - a `vector` column and a
generated `tsvector` column - so a hybrid query is one SQL statement rather
than two systems merged in application code.

PYTHON CONCEPTS FOR JAVA DEVS:
- `with conn:`             = try-with-resources; commits on exit, rolls back on error
- %s placeholders          = like JDBC PreparedStatement '?' - never string-format SQL
- Json(dict)               = tells psycopg to send a dict as a JSONB column
- Path(__file__).parent    = the directory this source file lives in
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import psycopg
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

from src.rag.text_splitter import TextChunk

logger = logging.getLogger(__name__)

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


@dataclass
class SearchResult:
    """A search result from the vector store."""
    content: str
    metadata: dict
    score: float  # cosine distance: lower = more similar


class VectorStore:
    """
    Postgres-backed chunk store - like a Spring @Repository.

    Applies its own schema on construction, so a fresh database becomes usable
    without a separate migration step.
    """

    def __init__(
        self,
        database_url: str,
        embedding_dim: int = 768,
        embedding_model: str = "unknown",
    ) -> None:
        self._url = database_url
        self._dim = embedding_dim
        self._model = embedding_model

        self._apply_schema()
        self._verify_dimension()

        logger.info(
            "Vector store ready: %d chunks, %d dimensions",
            self.get_document_count(), self._dim,
        )

    # ------------------------------------------------------------ internals --

    def _connect(self) -> psycopg.Connection:
        """
        Open a connection with pgvector's type adapters registered.

        One connection per operation. Fine against local Postgres and at this
        corpus size; a managed database over the network would want a pool
        (psycopg_pool.ConnectionPool) to avoid paying TLS setup per query.
        """
        conn = psycopg.connect(self._url)
        register_vector(conn)
        return conn

    def _apply_schema(self) -> None:
        """
        Create the extension, table and indexes if absent. Safe to run repeatedly.

        Deliberately uses a plain connection rather than _connect(): pgvector's
        register_vector() looks up the `vector` type and fails if it is absent,
        and this is the code that creates it. Registering here would mean the
        store could never bootstrap a fresh database.
        """
        with psycopg.connect(self._url) as conn:
            conn.execute(_SCHEMA_PATH.read_text())

    def _verify_dimension(self) -> None:
        """
        Fail fast if the schema's vector(N) disagrees with configured N.

        Otherwise the mismatch surfaces as an opaque insert error midway through
        ingestion, long after the actual mistake.
        """
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT atttypmod
                FROM pg_attribute
                WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'
                """
            ).fetchone()

        if row and row[0] > 0 and row[0] != self._dim:
            raise ValueError(
                f"Schema declares vector({row[0]}) but EMBEDDING_DIM is {self._dim}. "
                f"Change one to match the other - and note that altering the column "
                f"requires re-embedding every chunk, since vectors of different "
                f"sizes are not comparable."
            )

    # --------------------------------------------------------------- writes --

    def add_chunks(
        self,
        chunks: list[TextChunk],
        embeddings: list[list[float]],
    ) -> int:
        """
        Insert or update chunks by their stable chunk_id.

        Because chunk_id is content-derived, re-ingesting an unchanged document
        rewrites identical rows - a no-op in effect, not a destructive shuffle.

        Returns: number of chunks written
        """
        if not chunks:
            return 0

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings"
            )

        ids = [c.chunk_id for c in chunks]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate chunk IDs in batch - refusing to write")

        rows = [
            (
                chunk.chunk_id,
                chunk.metadata.get("source", "unknown"),
                chunk.chunk_index,
                chunk.content,
                Jsonb(chunk.metadata),
                embedding,
                self._model,
                self._dim,
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]

        with self._connect() as conn:
            conn.cursor().executemany(
                """
                INSERT INTO chunks (
                    chunk_id, source, chunk_index, content,
                    metadata, embedding, embedding_model, embedding_dim
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    source          = EXCLUDED.source,
                    chunk_index     = EXCLUDED.chunk_index,
                    content         = EXCLUDED.content,
                    metadata        = EXCLUDED.metadata,
                    embedding       = EXCLUDED.embedding,
                    embedding_model = EXCLUDED.embedding_model,
                    embedding_dim   = EXCLUDED.embedding_dim
                """,
                rows,
            )

        logger.info("Wrote %d chunks", len(rows))
        return len(rows)

    def existing_chunk_ids(
        self,
        chunk_ids: list[str],
        embedding_model: str,
        embedding_dim: int,
    ) -> set[str]:
        """
        Of the given chunk IDs, which are already stored by THIS model at THIS
        size - and therefore do not need embedding again.

        Model and dimension are part of the question, not decoration: the same
        text embedded by a different model produces a vector in a different
        space, and the two are not comparable. Reusing across a model change
        would quietly corrupt the index.

        This is what content-derived chunk IDs bought us. Under the original
        positional scheme, "have I already embedded this?" was unanswerable -
        the IDs moved whenever the corpus changed.
        """
        if not chunk_ids:
            return set()

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT chunk_id FROM chunks
                WHERE chunk_id = ANY(%s)
                  AND embedding_model = %s
                  AND embedding_dim = %s
                """,
                (list(chunk_ids), embedding_model, embedding_dim),
            ).fetchall()

        return {row[0] for row in rows}

    def delete_orphans(self, sources: set[str], keep_chunk_ids: set[str]) -> int:
        """
        Drop chunks belonging to the given sources that the corpus no longer
        produces.

        Editing a document changes its chunks' content, and therefore their IDs.
        The new versions are inserted, but without this the previous versions
        stay behind and keep surfacing in search results. Deleting a document
        entirely has the same problem.

        Only touches the sources just ingested, so it is safe to run against a
        partial corpus - documents that were not loaded this time are untouched.

        Returns: number of chunks deleted
        """
        if not sources:
            return 0

        with self._connect() as conn:
            cur = conn.execute(
                """
                DELETE FROM chunks
                WHERE source = ANY(%s)
                  AND NOT (chunk_id = ANY(%s))
                """,
                (list(sources), list(keep_chunk_ids)),
            )
            deleted = cur.rowcount

        if deleted:
            logger.info("Removed %d orphaned chunk(s)", deleted)
        return deleted

    def delete_by_source(self, source: str) -> int:
        """
        Remove every chunk belonging to one document.

        Needed before re-ingesting an edited document: its changed chunks get
        new IDs, so without this the previous versions linger as orphans and
        keep turning up in search results.

        Returns: number of chunks deleted
        """
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM chunks WHERE source = %s", (source,))
            deleted = cur.rowcount

        if deleted:
            logger.info("Deleted %d chunks from source '%s'", deleted, source)
        return deleted

    def clear(self) -> None:
        """Delete all chunks."""
        with self._connect() as conn:
            conn.execute("TRUNCATE chunks")
        logger.info("Vector store cleared")

    # --------------------------------------------------------------- reads --

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: Optional[dict] = None,
    ) -> list[SearchResult]:
        """
        Find the chunks most similar to a query vector.

        `<=>` is pgvector's cosine distance operator. Ordering by it lets the
        HNSW index serve the query; computing the distance in a WHERE clause
        instead would force a sequential scan.

        Args:
            query_embedding: the query vector
            top_k: how many results to return (like SQL LIMIT)
            where: optional JSONB containment filter, e.g. {"type": ".md"}
        """
        # The ::vector casts are required, not decoration. psycopg sends a Python
        # list as double precision[], and while pgvector defines an assignment
        # cast that lets an INSERT succeed, operator resolution does not consider
        # assignment casts - so `vector <=> double precision[]` matches nothing.
        sql = """
            SELECT content, metadata, embedding <=> %s::vector AS distance
            FROM chunks
            {filter}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        params: list = [query_embedding]

        if where:
            # Jsonb, not Json: the column is jsonb and the containment operator
            # is only defined as jsonb @> jsonb. Json() would send the `json`
            # type, which Postgres will happily accept on INSERT via an
            # assignment cast but cannot match to an operator.
            sql = sql.format(filter="WHERE metadata @> %s")
            params.append(Jsonb(where))
        else:
            sql = sql.format(filter="")

        params.extend([query_embedding, top_k])

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        results = [
            SearchResult(content=content, metadata=metadata, score=distance)
            for content, metadata, distance in rows
        ]
        logger.info("Found %d results for query", len(results))
        return results

    def hybrid_search(
        self,
        query_embedding: list[float],
        query_text: str,
        top_k: int = 5,
        rrf_k: int = 60,
        keyword_weight: float = 1.0,
    ) -> list[SearchResult]:
        """
        Search semantically AND lexically, then fuse the two rankings.

        WHY BOTH
        Dense search finds meaning but compresses away rare tokens - ask it about
        `reclaimPolicy` and it returns generally storage-ish passages. Keyword
        search nails exact identifiers but scores zero when the question shares no
        vocabulary with the answer, which is most natural-language questions.
        Each is blind exactly where the other is strong.

        WHY RECIPROCAL RANK FUSION RATHER THAN ADDING SCORES
        BM25-style relevance is unbounded and cosine distance is bounded, so the
        two numbers cannot be meaningfully combined. RRF ignores scores entirely
        and combines by POSITION:

            score(doc) = sum over lists of  1 / (rrf_k + rank_in_that_list)

        A document ranked 1st scores 1/61, 2nd scores 1/62, and appearing in both
        lists beats appearing high in one. No normalisation, no per-corpus tuning,
        and nothing to re-tune as the data drifts. rrf_k=60 is the value from the
        original paper and is not sensitive.

        WHY keyword_weight EXISTS
        Measured on this corpus, an equal-weight fusion made exact-identifier
        questions perfect (21/21 at recall@5, up from 20) and paraphrased ones
        markedly worse (10/21, down from 14) - a net loss.

        The cause is that OR-semantics keyword search matches on common words
        too. A question phrased as "how do I update my passwords and settings"
        has no rare terms, so the keyword arm returns forty loosely-related
        chunks, and RRF gives that noise the same vote as the dense arm:

            dense #1,  keyword absent   ->  1/61          = 0.0164
            dense #30, keyword #1       ->  1/90 + 1/61   = 0.0275   wins

        Weighting the keyword arm below 1.0 keeps its value on rare terms - where
        it ranks the right chunk first and dense ranks it low - while stopping a
        match on the word "update" from displacing a strong dense result.

        Both arms retrieve deeper than top_k so fusion has material to work with -
        a chunk ranked 12th by vector and 3rd by keyword should be able to surface.

        NOTE ON `score`: for dense search, lower is better (cosine distance). Here
        it is the RRF score, where HIGHER is better. Results are returned
        best-first either way, which is all the eval harness depends on.
        """
        candidate_depth = max(top_k * 4, 20)

        # plainto_tsquery joins every term with AND, so a fifteen-word question
        # requires a chunk containing all fifteen stems - which matches nothing,
        # and silently turns hybrid search back into dense-only. Rewriting the
        # operators to OR gives search semantics: any term can match, and
        # ts_rank scores documents higher for matching more of them, weighted by
        # term rarity.
        or_query = "replace(plainto_tsquery('english', %(q)s)::text, '&', '|')::tsquery"

        sql = f"""
            WITH dense AS (
                SELECT chunk_id, content, metadata,
                       row_number() OVER (ORDER BY embedding <=> %(vec)s::vector) AS rank
                FROM chunks
                ORDER BY embedding <=> %(vec)s::vector
                LIMIT %(depth)s
            ),
            keyword AS (
                SELECT chunk_id, content, metadata,
                       row_number() OVER (
                           ORDER BY ts_rank(content_tsv, {or_query}) DESC
                       ) AS rank
                FROM chunks
                WHERE content_tsv @@ {or_query}
                ORDER BY ts_rank(content_tsv, {or_query}) DESC
                LIMIT %(depth)s
            )
            SELECT
                COALESCE(d.content, k.content)   AS content,
                COALESCE(d.metadata, k.metadata) AS metadata,
                COALESCE(1.0 / (%(rrf_k)s + d.rank), 0)
                  + %(kw_weight)s * COALESCE(1.0 / (%(rrf_k)s + k.rank), 0)
                  AS rrf_score
            FROM dense d
            FULL OUTER JOIN keyword k ON d.chunk_id = k.chunk_id
            ORDER BY rrf_score DESC
            LIMIT %(top_k)s
        """

        with self._connect() as conn:
            rows = conn.execute(sql, {
                "vec": query_embedding,
                "q": query_text,
                "depth": candidate_depth,
                "rrf_k": rrf_k,
                "kw_weight": keyword_weight,
                "top_k": top_k,
            }).fetchall()

        results = [
            SearchResult(content=content, metadata=metadata, score=float(score))
            for content, metadata, score in rows
        ]
        logger.info("Hybrid search returned %d results", len(results))
        return results

    def get_document_count(self) -> int:
        """Total number of chunks stored."""
        with self._connect() as conn:
            row = conn.execute("SELECT count(*) FROM chunks").fetchone()
        return row[0] if row else 0
