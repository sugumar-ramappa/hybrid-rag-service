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


-- Query embeddings, cached.
--
-- The eval harness runs the same 42 questions on every configuration, so
-- without this each run spends 42 of a 1,000/day quota re-embedding text that
-- has not changed. In Postgres rather than a local file because a local cache
-- dies with the container and is not shared between instances.
CREATE TABLE IF NOT EXISTS embedding_cache (
    -- sha256(model : dimensions : task_type : text).
    --
    -- task_type is part of the key, not decoration: chunks embed as
    -- RETRIEVAL_DOCUMENT and queries as RETRIEVAL_QUERY, and the same text
    -- under each produces a different vector. A key omitting it would serve a
    -- document vector for a query and quietly degrade retrieval.
    cache_key    TEXT        PRIMARY KEY,

    embedding    vector(768) NOT NULL,
    model        TEXT        NOT NULL,
    dimensions   INTEGER     NOT NULL,
    task_type    TEXT        NOT NULL,

    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Support LRU eviction and a reportable hit rate.
    last_used_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    hit_count    INTEGER     NOT NULL DEFAULT 0
);


-- Generated answers, cached by MEANING rather than by exact text.
--
-- This is the cache that matters economically. Embedding a question is roughly
-- 1% of a request's cost; retrieval and generation are the other 99%. Matching
-- a new question against previously answered ones lets both be skipped.
--
-- Measured on this corpus, rephrasings of the same question sit 0.009-0.070
-- apart, while the closest two genuinely different questions sit 0.211 apart.
-- The default threshold of 0.12 falls in that gap with margin on both sides.
CREATE TABLE IF NOT EXISTS answer_cache (
    id             BIGSERIAL   PRIMARY KEY,

    -- Kept for debugging and for auditing what a hit actually matched against.
    -- Serving the wrong answer is silent, so being able to see which question
    -- a response was reused from matters more than it looks.
    question       TEXT        NOT NULL,
    embedding      vector(768) NOT NULL,

    answer         TEXT        NOT NULL,
    sources        JSONB       NOT NULL,

    -- A fingerprint of the corpus at the time the answer was generated.
    -- Re-ingesting changes it, so every prior answer stops matching rather
    -- than being served against documents that no longer say the same thing.
    -- Without this, a cache hit after a corpus update is silently wrong.
    corpus_version TEXT        NOT NULL,

    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_used_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    hit_count      INTEGER     NOT NULL DEFAULT 0
);

-- Lookup is itself a vector search, so it needs the same index the chunks do.
CREATE INDEX IF NOT EXISTS answer_cache_embedding_idx
    ON answer_cache USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS answer_cache_corpus_idx
    ON answer_cache (corpus_version);
