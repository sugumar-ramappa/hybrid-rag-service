-- =============================================================================
-- Understanding vector search, one query at a time
-- =============================================================================
--
-- Run these in order. Each builds on the one before.
-- Nothing here changes data - every query is read-only.
--
-- In DBeaver: put the cursor inside a statement and press Cmd+Enter.
-- In the terminal: docker exec -i ragdb psql -U postgres -d ragdb
--
-- =============================================================================


-- =============================================================================
-- PART 1 - What is actually stored
-- =============================================================================

-- 1.1  Look at one chunk. This is just text from a documentation page.
SELECT source, chunk_index, length(content) AS chars, left(content, 300) AS text
FROM chunks
WHERE source = 'concepts/configuration/secret.md' AND chunk_index = 1;


-- 1.2  Every chunk also has an "embedding". Look at the first few numbers.
SELECT left(embedding::text, 150) || ' ...' AS embedding_starts_like
FROM chunks
WHERE source = 'concepts/configuration/secret.md' AND chunk_index = 1;

--      You will see something like:  [0.031,-0.018,0.044,-0.009,0.056, ...
--
--      That is the embedding: a list of numbers that represents the MEANING of
--      the text. It was produced by an AI model that read the chunk.
--
--      You cannot read it. Neither can I. But the computer can compare two of
--      them, and that is the entire trick.


-- 1.3  How many numbers are in one embedding?
SELECT vector_dims(embedding) AS how_many_numbers
FROM chunks LIMIT 1;

--      768. Every chunk is represented by exactly 768 numbers.
--      Think of it as 768 different "scores" describing the text.


-- =============================================================================
-- PART 2 - The one operation that makes search work: distance
-- =============================================================================
--
-- Two chunks about similar things end up with SIMILAR numbers.
-- Two chunks about different things end up with DIFFERENT numbers.
--
-- "Distance" measures how different two embeddings are:
--      0.0  = identical meaning
--      0.5  = loosely related
--      1.0  = unrelated
--
-- The operator is  <=>  and it does all the work.


-- 2.1  Distance from a chunk to ITSELF. Must be zero.
SELECT round((a.embedding <=> a.embedding)::numeric, 4) AS distance_to_itself
FROM chunks a
WHERE a.source = 'concepts/configuration/secret.md' AND a.chunk_index = 1;


-- 2.2  Distance between two chunks about the SAME topic (both from the Secrets page).
SELECT round((a.embedding <=> b.embedding)::numeric, 4) AS distance,
       left(a.content, 60) AS chunk_a,
       left(b.content, 60) AS chunk_b
FROM chunks a, chunks b
WHERE a.source = 'concepts/configuration/secret.md' AND a.chunk_index = 1
  AND b.source = 'concepts/configuration/secret.md' AND b.chunk_index = 2;


-- 2.3  Distance between two chunks about DIFFERENT topics.
--      Compare the number to 2.2 - it should be noticeably larger.
SELECT round((a.embedding <=> b.embedding)::numeric, 4) AS distance,
       a.source AS topic_a,
       b.source AS topic_b
FROM chunks a, chunks b
WHERE a.source = 'concepts/configuration/secret.md' AND a.chunk_index = 1
  AND b.source = 'concepts/containers/images.md' AND b.chunk_index = 1;

--      THIS IS THE WHOLE IDEA. Related text = small number.
--      Unrelated text = big number. Everything else is built on it.


-- =============================================================================
-- PART 3 - Search = sort everything by distance and take the top few
-- =============================================================================

-- 3.1  "Find the 5 chunks most similar to THIS chunk."
--
--      NOTE: this uses an existing chunk as the search input, not a question.
--      That is deliberate - it keeps the SQL as simple as possible so nothing
--      distracts from the one idea: sort by distance, take the top few.
--      Query 3.4 does the same thing with a REAL question.
--
--      Read the results: the first is the chunk itself (distance 0, because
--      nothing is closer to a thing than itself - which also proves the
--      comparison is working), and the rest should obviously be about the
--      same subject.
SELECT round((c.embedding <=> q.embedding)::numeric, 4) AS distance,
       c.source,
       c.chunk_index,
       left(c.content, 70) AS text
FROM chunks c,
     (SELECT embedding FROM chunks
      WHERE source = 'concepts/configuration/secret.md' AND chunk_index = 1) q
ORDER BY c.embedding <=> q.embedding
LIMIT 5;

--      That is EXACTLY what search() does in the Python code. The only
--      difference is that a real search uses the embedding of your QUESTION
--      instead of the embedding of an existing chunk.


-- 3.2  Try it from a different starting point and see how the results change.
SELECT round((c.embedding <=> q.embedding)::numeric, 4) AS distance,
       c.source,
       left(c.content, 70) AS text
FROM chunks c,
     (SELECT embedding FROM chunks
      WHERE source = 'concepts/containers/images.md' AND chunk_index = 1) q
ORDER BY c.embedding <=> q.embedding
LIMIT 5;


-- 3.3  Why does the code say "LIMIT 5"?
--      Because only the top 5 chunks get sent to the AI to write an answer.
--      Anything at position 6 or lower is never seen. That is why the metric
--      we measure is "recall@5" - was the right chunk in the top 5?


-- 3.4  THE REAL THING: search with an actual question.
--
--      Queries 3.1 and 3.2 compared against an existing CHUNK, because turning
--      a question into an embedding normally costs an API call.
--
--      But the eval harness already embedded all 42 evaluation questions and
--      cached them in the embedding_cache table. The cache key is
--      sha256(model : dimensions : task_type : question), which Postgres can
--      recompute - so we can look up a real question's embedding and search
--      with it, for free.
--
--      This is EXACTLY what happens when someone asks the system a question.

WITH q AS (
    SELECT embedding FROM embedding_cache
    WHERE cache_key = encode(sha256((
        'gemini-embedding-001:768:RETRIEVAL_QUERY:' ||
        'What history lies behind the name K8s being used as an abbreviation for Kubernetes?'
    )::bytea), 'hex')
)
SELECT round((c.embedding <=> (SELECT embedding FROM q))::numeric, 4) AS distance,
       c.source,
       c.chunk_index,
       left(c.content, 70) AS text
FROM chunks c
ORDER BY c.embedding <=> (SELECT embedding FROM q)
LIMIT 5;

--      The top result should be concepts/overview/_index.md chunk 1 - the
--      passage explaining that "K8s" counts the eight letters between K and s.
--
--      Nobody told the database these were related. The question and the answer
--      share almost no words. The embeddings put them near each other because
--      they MEAN the same thing, and that is the entire point of vector search.


-- 3.5  Try a different question.
--
--      Open eval/golden_set.json, copy any "question" value, and paste it in
--      place of the text below. All 42 are cached, so any of them works.
--
--      Try one of each style and compare:
--        exact_term - "Which Pod quality classifications are blocked from
--                      accessing swap memory when LimitedSwap is active"
--        paraphrase - "How can I see how many copies of the main management
--                      program are currently running"
--
--      The second one is q006, which the evaluation scored as a MISS. Run it
--      and look at what comes back instead - that is what a retrieval failure
--      actually looks like.

WITH q AS (
    SELECT embedding FROM embedding_cache
    WHERE cache_key = encode(sha256((
        'gemini-embedding-001:768:RETRIEVAL_QUERY:' ||
        'Which Pod quality classifications are blocked from accessing swap memory when LimitedSwap is active'
    )::bytea), 'hex')
)
SELECT round((c.embedding <=> (SELECT embedding FROM q))::numeric, 4) AS distance,
       c.source,
       c.chunk_index,
       left(c.content, 70) AS text
FROM chunks c
ORDER BY c.embedding <=> (SELECT embedding FROM q)
LIMIT 5;

--      If a question returns nothing at all, its embedding is not cached -
--      check the text matches eval/golden_set.json exactly, including
--      punctuation. The cache key is a hash, so one different character
--      produces a completely different key and no match.


-- =============================================================================
-- PART 4 - The completely different kind of search: keywords
-- =============================================================================
--
-- Everything above compares MEANING. This part compares WORDS.
-- They are unrelated mechanisms that happen to solve the same problem.


-- 4.1  Every chunk also has a "content_tsv" column, built automatically
--      by Postgres from the text.
SELECT left(content_tsv::text, 200) AS keyword_index
FROM chunks
WHERE source = 'concepts/configuration/secret.md' AND chunk_index = 1;

--      You will see things like  'secret':3,11  'configur':7
--
--      Postgres has stripped the text down to word stems and recorded where
--      each appears. "configure", "configured" and "configuring" all become "configur",
--      which is why searching for one finds the others.


-- 4.2  Find chunks containing a specific rare word.
SELECT source, chunk_index, left(content, 70) AS text
FROM chunks
WHERE content_tsv @@ plainto_tsquery('english', 'LimitedSwap');

--      Very few chunks contain it. That is what makes it a strong signal.


-- 4.3  Now a COMMON word. Notice how many chunks match.
SELECT count(*) AS chunks_containing_the_word
FROM chunks
WHERE content_tsv @@ plainto_tsquery('english', 'kubernetes');


-- 4.4  Rank keyword matches by relevance.
--      ts_rank scores higher when a chunk contains more of the search terms,
--      and when those terms are rare across the corpus.
SELECT round(ts_rank(content_tsv, plainto_tsquery('english', 'swap memory'))::numeric, 4)
           AS keyword_score,
       source,
       left(content, 70) AS text
FROM chunks
WHERE content_tsv @@ plainto_tsquery('english', 'swap memory')
ORDER BY keyword_score DESC
LIMIT 5;


-- =============================================================================
-- PART 5 - The bug that made hybrid search do nothing
-- =============================================================================
--
-- This is worth running, because it caused a real failure in this project.

-- 5.1  What does plainto_tsquery actually build from a long question?
SELECT plainto_tsquery('english',
    'Which Pod quality classifications are blocked from accessing swap memory when LimitedSwap is active'
) AS parsed;

--      Look at the output: every term is joined with  &  which means AND.
--      A chunk must contain ALL of them to match.


-- 5.2  So how many chunks match? (Answer: none.)
SELECT count(*) AS matches_with_AND
FROM chunks
WHERE content_tsv @@ plainto_tsquery('english',
    'Which Pod quality classifications are blocked from accessing swap memory when LimitedSwap is active');


-- 5.3  Replace AND with OR, and now it matches.
SELECT count(*) AS matches_with_OR
FROM chunks
WHERE content_tsv @@ replace(plainto_tsquery('english',
    'Which Pod quality classifications are blocked from accessing swap memory when LimitedSwap is active'
)::text, '&', '|')::tsquery;

--      From 0 to over 400.
--
--      That was the bug: the keyword half of hybrid search returned NOTHING
--      for every question, so hybrid silently behaved exactly like dense.
--      The tests missed it because they used one-word queries, where
--      "all terms must match" and "any term must match" are the same thing.


-- =============================================================================
-- PART 6 - Combining both, by hand
-- =============================================================================
--
-- Dense search and keyword search each produce a ranked list.
-- Hybrid search merges them. This query does it step by step so you can
-- see the arithmetic.

WITH q AS (
    SELECT embedding FROM chunks
    WHERE source = 'concepts/configuration/secret.md' AND chunk_index = 1
),
dense AS (
    SELECT chunk_id, source, chunk_index,
           row_number() OVER (ORDER BY c.embedding <=> (SELECT embedding FROM q)) AS dense_rank
    FROM chunks c
    ORDER BY c.embedding <=> (SELECT embedding FROM q)
    LIMIT 20
),
keyword AS (
    SELECT chunk_id, source, chunk_index,
           row_number() OVER (
               ORDER BY ts_rank(content_tsv, plainto_tsquery('english', 'secret configuration data')) DESC
           ) AS keyword_rank
    FROM chunks
    WHERE content_tsv @@ plainto_tsquery('english', 'secret configuration data')
    LIMIT 20
)
SELECT
    COALESCE(d.source, k.source)           AS source,
    COALESCE(d.chunk_index, k.chunk_index) AS chunk_index,
    d.dense_rank,
    k.keyword_rank,
    -- Reciprocal Rank Fusion: position 1 scores 1/61, position 2 scores 1/62.
    -- A chunk found by BOTH gets both contributions and therefore wins.
    round((COALESCE(1.0/(60 + d.dense_rank), 0)
         + COALESCE(1.0/(60 + k.keyword_rank), 0))::numeric, 5) AS combined_score
FROM dense d
FULL OUTER JOIN keyword k ON d.chunk_id = k.chunk_id
ORDER BY combined_score DESC
LIMIT 10;

--      Read the dense_rank and keyword_rank columns.
--      Rows with a number in BOTH rise to the top - that is the whole point.
--      Rows with a number in only one column score about half as much.
--
--      Now look at the rows where keyword_rank is filled in but dense_rank is
--      empty. Are those chunks actually relevant? On this corpus, usually not -
--      they just happened to contain a common word. That noise is why hybrid
--      search scored WORSE than dense here.


-- =============================================================================
-- PART 7 - What the evaluation measured
-- =============================================================================

-- 7.1  The corpus
SELECT count(*) AS chunks, count(DISTINCT source) AS documents FROM chunks;

-- 7.2  The cached question embeddings - one per evaluation question
SELECT count(*) AS cached_questions, sum(hit_count) AS times_reused
FROM embedding_cache;

-- 7.3  For each of the 42 questions, the eval:
--        1. looked up the question's embedding (Part 1)
--        2. sorted all 784 chunks by distance from it (Part 3)
--        3. checked whether the expected chunk was in the top 5
--
--      34 of 42 succeeded with dense search.  That is recall@5 = 0.81.
