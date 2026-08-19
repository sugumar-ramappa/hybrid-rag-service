-- Schema for hybrid retrieval: vector similarity and keyword search in one table.
--
-- Applied automatically by VectorStore on startup (idempotent), or by hand:
--     psql "$DATABASE_URL" -f src/rag/schema.sql

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    -- Stable, content-derived identity from TextSplitter. Because it does not
    -- depend on corpus order, re-ingesting an unchanged document is a no-op
    -- rather than a destructive rewrite.
    chunk_id        TEXT        PRIMARY KEY,

    source          TEXT        NOT NULL,
    chunk_index     INTEGER     NOT NULL,
    content         TEXT        NOT NULL,
    metadata        JSONB       NOT NULL DEFAULT '{}'::jsonb,

    -- Dimension must match GeminiConfig.embedding_dim. Changing it means a
    -- migration plus a full re-embed - vectors from different models are not
    -- comparable, so they cannot coexist in one index.
    embedding       vector(768) NOT NULL,

    -- Recorded per row so a model migration can run blue/green: build rows with
    -- the new model alongside the old, verify, then cut over. Without this you
    -- cannot tell which rows came from which model.
    embedding_model TEXT        NOT NULL,
    embedding_dim   INTEGER     NOT NULL,

    -- The keyword half of hybrid search. GENERATED means Postgres maintains it
    -- from content automatically, so it can never drift out of sync.
    content_tsv     tsvector    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Approximate nearest-neighbour index for the semantic half of the search.
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Inverted index for the keyword half.
CREATE INDEX IF NOT EXISTS chunks_content_tsv_idx
    ON chunks USING gin (content_tsv);

-- Supports deleting or re-ingesting a single document without touching others.
CREATE INDEX IF NOT EXISTS chunks_source_idx
    ON chunks (source);
