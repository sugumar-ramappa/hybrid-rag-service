-- Inspection queries for the walkthrough in docs/walkthrough.md
--
-- Open in DBeaver (or any SQL client) connected to:
--   localhost:5432 / ragdb / postgres / dev
--
-- Run one statement at a time. In DBeaver that is Ctrl+Enter with the cursor
-- inside the statement.
--
-- NOTE: never run `SELECT * FROM chunks`. Each row carries 768 floats, so the
-- grid renders a multi-thousand-character string per row and becomes unusable.


-- ===========================================================================
-- 1. What is stored
-- ===========================================================================

SELECT chunk_index,
       chunk_id,
       source,
       length(content) AS chars,
       left(content, 60) AS preview,
       created_at
FROM chunks
ORDER BY source, chunk_index;

-- `source` is the path relative to the corpus root, not an absolute path -
-- that is what makes citations useful once the corpus is nested.


-- ===========================================================================
-- 2. Are the vectors real, and normalized?
-- ===========================================================================

SELECT c.chunk_index,
       vector_dims(c.embedding) AS dims,
       round(sqrt(sum(v.val * v.val))::numeric, 4) AS magnitude
FROM chunks c,
     LATERAL unnest(c.embedding::real[]) AS v(val)
GROUP BY c.chunk_index, c.embedding
ORDER BY c.chunk_index;

-- Expect dims = 768 and magnitude = 1.0000.
-- The magnitude is _normalize() in embeddings.py doing its job: truncating a
-- Matryoshka embedding from 3072 to 768 leaves it no longer unit length.


-- Peek at actual values without drowning the grid.
-- Not really a check - it is here so you can see what an embedding physically
-- is. Components should sit around +/- 0.01 to 0.08: a normalized 768-dim
-- vector averages 1/sqrt(768) ~ 0.036 per component. Values near 1.0 would
-- mean normalization did not run.
SELECT chunk_index,
       left(embedding::text, 70) || ' ...' AS first_few_values
FROM chunks
ORDER BY chunk_index;


-- THIS is the real check: every chunk must have its own vector.
-- duplicates must be 0.
--
-- Zero is not guaranteed by anything except correct code. The pipeline slices
-- chunks into batches, embeds each batch, and zips the results back - an
-- off-by-one there would give chunk 3 chunk 1's vector. Nothing would error:
-- ingest succeeds, search returns results, and they are quietly wrong.
SELECT count(*)                                    AS chunks,
       count(DISTINCT embedding::text)             AS distinct_vectors,
       count(*) - count(DISTINCT embedding::text)  AS duplicates
FROM chunks;


-- ===========================================================================
-- 3. The keyword column you never wrote
-- ===========================================================================

SELECT chunk_index,
       left(content_tsv::text, 90) AS tsvector_preview
FROM chunks
ORDER BY chunk_index;

-- content_tsv is GENERATED ALWAYS AS (to_tsvector('english', content)) STORED.
-- Postgres derives and maintains it, so it can never drift from content.
-- Numbers after each lexeme are token positions. 'storag':4,17 means the stem
-- "storag" appears at positions 4 and 17 - note stemming has already collapsed
-- storage/storages/storing into one lexeme.


-- ===========================================================================
-- 4. Which model produced these vectors
-- ===========================================================================

SELECT embedding_model,
       embedding_dim,
       count(*) AS chunks,
       min(created_at) AS first_written,
       max(created_at) AS last_written
FROM chunks
GROUP BY embedding_model, embedding_dim;

-- Recorded per row so a model change can be migrated blue/green: build rows
-- with the new model alongside the old, compare, then cut over. Vectors from
-- different models are not comparable, so they can never share an index.


-- ===========================================================================
-- 5. The indexes
-- ===========================================================================

SELECT indexname,
       indexdef
FROM pg_indexes
WHERE tablename = 'chunks';

-- chunks_embedding_idx    HNSW  - approximate nearest neighbour, the semantic half
-- chunks_content_tsv_idx  GIN   - inverted index, the keyword half
-- chunks_source_idx             - supports per-document delete and re-ingest
-- chunks_pkey                   - chunk_id, which is what makes upsert idempotent


-- Table and index sizes
SELECT pg_size_pretty(pg_total_relation_size('chunks')) AS total,
       pg_size_pretty(pg_relation_size('chunks'))       AS table_only,
       pg_size_pretty(pg_indexes_size('chunks'))        AS indexes;


-- ===========================================================================
-- 6. Semantic search, by hand
-- ===========================================================================

-- How close is every chunk to chunk 0? `<=>` is cosine distance:
-- 0 = identical direction, 1 = unrelated, 2 = opposite.
SELECT chunk_index,
       left(content, 55) AS preview,
       round((embedding <=> (SELECT embedding FROM chunks WHERE chunk_index = 0))::numeric, 4)
           AS distance_from_chunk_0
FROM chunks
ORDER BY distance_from_chunk_0;

-- Chunk 0 scores 0.0000 against itself. The ordering of the rest is the same
-- operator your search() method uses - retrieval, with no Python involved.


-- ===========================================================================
-- 7. Keyword search - a preview of the other half of hybrid
-- ===========================================================================

SELECT chunk_index,
       left(content, 55) AS preview,
       round(ts_rank(content_tsv, plainto_tsquery('english', 'reclaim policy'))::numeric, 4)
           AS keyword_rank
FROM chunks
WHERE content_tsv @@ plainto_tsquery('english', 'reclaim policy')
ORDER BY keyword_rank DESC;

-- Nothing in the application queries this yet. Compare its ranking against the
-- semantic one above for an exact-identifier query - that gap is the reason
-- hybrid search is step 6 of the plan.

-- Try a query with no shared vocabulary and watch keyword search return nothing:
SELECT count(*) AS keyword_matches
FROM chunks
WHERE content_tsv @@ plainto_tsquery('english', 'how do I keep data after a pod restarts');


-- ===========================================================================
-- 8. Before and after an edit (walkthrough steps 5 and 6)
-- ===========================================================================

-- Run this BEFORE editing the document and note the IDs
SELECT chunk_index, chunk_id, left(content, 40) AS preview
FROM chunks
ORDER BY chunk_index;

-- Re-ingest, then run it again. Only chunks whose text actually changed will
-- have new IDs, because chunk_id = sha256(source : index : content).


-- Proves rows were reused rather than rewritten: after a no-change re-ingest
-- the count and timestamps are identical.
SELECT count(*) AS chunks,
       min(created_at) AS oldest,
       max(created_at) AS newest
FROM chunks;


-- ===========================================================================
-- 9. Prove the orphan sweep (walkthrough step 7)
-- ===========================================================================

-- Insert a chunk the corpus does not produce, borrowing an existing vector
INSERT INTO chunks (chunk_id, source, chunk_index, content, metadata,
                    embedding, embedding_model, embedding_dim)
SELECT 'fake_orphan_001', source, 99, 'stale content that no longer exists',
       '{}'::jsonb, embedding, embedding_model, embedding_dim
FROM chunks
LIMIT 1;

SELECT chunk_id, chunk_index, left(content, 40) AS preview
FROM chunks
WHERE chunk_id = 'fake_orphan_001';

-- Now run: python -m scripts.ingest
-- Expect "1 removed", and this query to return nothing afterwards.


-- ===========================================================================
-- 10. Reset
-- ===========================================================================

-- TRUNCATE chunks;
