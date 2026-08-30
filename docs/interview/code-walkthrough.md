# Code walkthrough

Every class and method, what it does, and the non-obvious decision inside it.
Ordered the way data flows, not alphabetically.

Read [`architecture-flow.md`](architecture-flow.md) first for the diagrams.

---

## Map

```
src/
├── config.py                 all settings, immutable, env-backed
├── rag/
│   ├── document_loader.py    disk        → Document[]
│   ├── text_splitter.py      Document[]  → TextChunk[]   ← chunk identity
│   ├── embeddings.py         text        → vector(768)
│   ├── vector_store.py       TextChunk[] ⇄ Postgres      ← both searches
│   ├── embedding_cache.py    text        → vector (exact-key cache)
│   ├── answer_cache.py       vector      → answer (semantic cache)
│   ├── pipeline.py           orchestrates all of the above
│   └── schema.sql            three tables, two indexes
```

For a Java reader: each class is a `@Service`, `config.py` is
`@ConfigurationProperties`, `@dataclass(frozen=True)` is a `record`, and
`pipeline.py` is the `@Component` that wires them together — except the wiring is
explicit in `__init__` rather than injected.

---

## `config.py` — settings

Four frozen dataclasses: `GeminiConfig`, `DatabaseConfig`, `RAGConfig`,
`AppConfig`. Every field reads an environment variable with a default.

```python
@dataclass(frozen=True)
class GeminiConfig:
    embedding_dim: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "768"))
    )
```

`field(default_factory=...)` rather than `= os.getenv(...)` matters: a plain
default is evaluated once at **class definition** time, so the value would be
frozen at import and ignore anything `load_dotenv()` set afterwards.

**`validate()`** fails fast on a missing API key or database URL, at startup
rather than at the first query.

The values that carry decisions:

| Setting | Default | Why that value |
|---|---|---|
| `EMBEDDING_DIM` | 768 | Under pgvector's 2000-dim HNSW cap |
| `EMBED_BATCH_SIZE` | 50 | Free tier is tokens/minute; batch size decides *how much work is lost* on a failure, not throughput |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 1000 / 200 | Baseline; overlap keeps a sentence split across a boundary retrievable |
| `TOP_K_RESULTS` | 5 | What the prompt can carry — hence recall@**5** is the headline metric |
| `RETRIEVAL_MODE` | `hybrid` | Measured better: recall@5 0.95 → 0.98 on the 43-question set, 0.921 → 0.951 on the 267-pair set that superseded it |
| `KEYWORD_WEIGHT` | 1.0 | Equal vote with dense |
| `ANSWER_CACHE_THRESHOLD` | 0.12 | Calibrated: rephrasings 0.009–0.070, different questions 0.211 |
| `MAX_FILES` | 56 (in `.env`) | Pins the corpus so the golden set stays valid |

---

## `document_loader.py` — disk → `Document[]`

### `load_all() -> list[Document]`

Walks the corpus directory and returns one `Document` per file (per *page* for
PDFs).

```python
for path in sorted(root.rglob("*")):
    if not path.is_file():                       continue
    if any(p.startswith(".") for p in path.parts): continue
```

Three details, all of which were bugs first:

**`rglob` not `iterdir`.** `iterdir()` reads only the top level. On a nested
documentation tree it loaded almost nothing, and the failure looked like a small
corpus rather than a broken walk.

**Hidden check on every path component, not the filename.** Once traversal
recursed, checking `path.name.startswith(".")` let every file under `.git/`
through — those files have perfectly ordinary names like `config` and `HEAD`.

**`sorted()`.** Makes `max_files` deterministic. The same cap always selects the
same files, so a golden set built against a capped corpus stays valid, and
raising the cap later is a superset rather than a reshuffle.

### `_source_of(relative) -> str`

Records `source` as the path **relative to the corpus root**, not the bare
filename. A documentation tree contains dozens of files called `_index.md`; a
bare name makes citations useless and weakens chunk identity, which derives
partly from source.

### Failure counting

`_load_file` catches per-file exceptions, counts them, and reports the total.
Without a count, a systematic problem is indistinguishable from a small corpus —
the individual stack traces scroll past and the total merely looks lower than
expected.

---

## `text_splitter.py` — `Document[]` → `TextChunk[]`

**The most important file in the project**, because chunk identity is what
everything downstream depends on.

### `split_documents(documents) -> list[TextChunk]`

```python
for doc in documents:
    texts = self._splitter.split_text(doc.content)
    doc_key = _document_key(doc.metadata)

    for chunk_index, text in enumerate(texts):     # restarts per document
        all_chunks.append(TextChunk(
            content=text,
            metadata={**doc.metadata, "chunk_index": chunk_index},
            chunk_index=chunk_index,
            chunk_id=_make_chunk_id(doc_key, chunk_index, text),
        ))
```

`RecursiveCharacterTextSplitter` tries separators in order — paragraph breaks,
then lines, then sentences, then words — so it splits at the largest natural
boundary that fits. 784 chunks from 56 documents, averaging 794 characters.

### `_document_key(metadata) -> str`

```python
return f"{source}#p{page}" if page is not None else source
```

The page number is in the key so that **two identical pages of a PDF do not
collide**. Without it, a repeated header page would produce the same chunk ID
twice and the second would overwrite the first. Covered by
`test_identical_pdf_pages_do_not_collide`.

### `_make_chunk_id(doc_key, chunk_index, content) -> str`

```python
digest = hashlib.sha256(f"{doc_key}:{chunk_index}:{content}".encode("utf-8"))
return digest.hexdigest()[:16]
```

**The bug this replaced.** `chunk_index` used to be a global counter across the
whole corpus:

```
before adding a document:   file_A → 0,1,2   file_B → 3,4,5
after:                      file_A → 0,1,2   file_NEW → 3,4   file_B → 5,6,7
                                                              ^^^^^^^^^^^^
                                              file_B's chunks now claim IDs
                                              3 and 4 — which upsert overwrote
```

`add_chunks` upserts on `chunk_id`, so file_B silently overwrote file_NEW's rows.
No error, no warning — retrieval just started returning content from the wrong
page.

Content-derived IDs fix it three ways at once:

1. **Stable** — an ID depends only on its own document and its own text, so
   adding a document cannot disturb existing ones
2. **Change-detecting** — editing a chunk changes its content and therefore its
   ID, so the edit is visible as a new row rather than a silent overwrite
3. **Answerable** — *"have I embedded this?"* becomes a primary-key lookup

That third property is what makes incremental ingest possible at all. Under the
positional scheme the question had no answer, because the IDs moved whenever the
corpus changed.

The existing test `test_chunk_indices_are_sequential` had been **asserting the
buggy behaviour**. It was replaced with three tests that assert the property that
actually matters — see [`testing.md`](testing.md).

---

## `embeddings.py` — text → `vector(768)`

### `embed_texts(texts) -> list[list[float]]`

Documents. Task type `RETRIEVAL_DOCUMENT`, batched at `self.batch_size`.

### `embed_query(query) -> list[float]`

Questions. Task type `RETRIEVAL_QUERY`, cache-checked first.

**Why two methods for the same API.** The same model encodes text differently
depending on its declared role:

```python
chunks:    config={"task_type": "RETRIEVAL_DOCUMENT"}
questions: config={"task_type": "RETRIEVAL_QUERY"}
```

A document is something to *be found*; a question is something *doing the
finding*. Identical text under those two settings produces two different vectors.
That asymmetry is the point of the parameter — and it is why `task_type` is part
of the embedding cache key. Serving a document vector where a query vector
belongs would degrade retrieval silently.

Only queries are cached. Document embeddings are already protected from repeat
work by incremental ingest.

### `_embed_with_retry(contents, task_type)`

```python
wait = 30 * (2 ** attempt)     # 30s, 60s, 120s
```

Backoff starts at **30 seconds**, not the usual 1–2. The free-tier limit is per
*minute*, so retrying inside the same minute is guaranteed to fail again — the
window has to roll over.

It also distinguishes two failures that arrive identically as
`429 RESOURCE_EXHAUSTED`:

```python
if "PerDay" in message or "per day" in message.lower():
    raise RuntimeError("Daily embedding quota exhausted - this will not
                        recover by waiting...")
```

A per-minute limit clears in a minute. A **daily** limit clears at midnight
Pacific, so retrying it just burns minutes to fail anyway. The error message says
so, and says that everything embedded so far is already stored.

### `_check_dim(vector)` and `_normalize(vector)`

```python
if len(vector) != self._dimensions:
    raise ValueError(f"Expected {self._dimensions}-dimension embeddings ...")
return _normalize(vector)
```

`_check_dim` fails fast if the API returns a different size than the schema
expects — a mismatch would otherwise surface as an opaque Postgres error hundreds
of rows into an ingest.

`_normalize` divides by magnitude. Matryoshka truncation from 3072 to 768 leaves
the vector no longer unit-length. Cosine distance is scale-invariant and would
not notice; L2 and inner-product operators would, and switching operators later
should not silently change results.

---

## `vector_store.py` — Postgres

The largest file, and where both searches live.

### `_connect() -> psycopg.Connection`

Opens a connection and calls `register_vector(conn)` so psycopg can adapt Python
lists to pgvector's type.

### `_apply_schema()`

Runs `schema.sql`. Uses a **plain `psycopg.connect()`, not `self._connect()`** —
a bootstrap ordering problem: `register_vector` needs the `vector` type to exist,
but the statement that creates it (`CREATE EXTENSION vector`) is inside the file
being applied. Going through `_connect()` deadlocked on first run against an
empty database.

### `_verify_dimension()`

Compares the `vector(N)` column against `EMBEDDING_DIM` and refuses to start if
they disagree. Catches a config change that would otherwise corrupt data.

### `add_chunks(chunks, embeddings) -> int`

```sql
INSERT INTO chunks (chunk_id, source, chunk_index, content,
                    metadata, embedding, embedding_model, embedding_dim)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (chunk_id) DO UPDATE SET ...
```

Two guards before the write:

```python
if len(chunks) != len(embeddings):     raise ValueError("Mismatch: ...")
if len(set(ids)) != len(ids):          raise ValueError("Duplicate chunk IDs")
```

The second is the regression guard for the chunk-ID bug. If identity ever breaks
again, the ingest **fails loudly** instead of silently overwriting rows.

`embedding_model` and `embedding_dim` are stored **per row**, which is what makes
a blue/green model migration possible — you can hold two generations of vectors
in the same table and cut over.

### `existing_chunk_ids(chunk_ids, model, dim) -> set[str]`

```sql
SELECT chunk_id FROM chunks
WHERE chunk_id = ANY(%s) AND embedding_model = %s AND embedding_dim = %s
```

The entire incremental-ingest mechanism. Model and dimension are in the `WHERE`
because vectors from different models occupy different spaces and are not
comparable — reusing across a model change would corrupt the index without any
visible error.

### `delete_orphans(sources, keep_chunk_ids) -> int`

```sql
DELETE FROM chunks WHERE source = ANY(%s) AND NOT (chunk_id = ANY(%s))
```

Editing a document changes its chunks' content and therefore their IDs. The new
versions get inserted; without this the **old versions stay behind and keep
being retrieved**.

Scoped to `source = ANY(sources)` — only the documents just ingested. That is
what makes it safe to run after a partial ingest: documents not loaded this time
are untouched.

### `search(query_embedding, top_k, where) -> list[SearchResult]`

```sql
SELECT content, metadata, embedding <=> %s::vector AS distance
FROM chunks
ORDER BY embedding <=> %s::vector
LIMIT %s
```

Dense search. `<=>` is cosine distance; lower is better.

**The `::vector` casts are required, not decoration.** psycopg sends a Python
list as `double precision[]`. pgvector defines an *assignment* cast, which is why
the `INSERT` in `add_chunks` works without one — but **operator resolution does
not consider assignment casts**, so `vector <=> double precision[]` matches no
operator and the query fails. This cost real debugging time: writes worked,
reads didn't.

Optional metadata filter uses `Jsonb`, not `Json`:

```python
sql = sql.format(filter="WHERE metadata @> %s")
params.append(Jsonb(where))
```

Same class of problem. The column is `jsonb` and the containment operator is only
defined as `jsonb @> jsonb`. `Json()` sends the `json` type, which Postgres
accepts on INSERT via an assignment cast but cannot match to an operator.

### `hybrid_search(query_embedding, query_text, top_k, rrf_k=60, keyword_weight=1.0)`

One statement, two CTEs, a full outer join, and rank fusion. Walked through in
detail in [`architecture-flow.md`](architecture-flow.md#flow-3--inside-hybrid-search).

The three decisions inside it:

**`candidate_depth = max(top_k * 4, 20)`** — both arms retrieve 20 so fusion has
material. If each returned only 5, fusion could only reorder what dense already
had.

**The OR rewrite** — `plainto_tsquery` ANDs every term, so a 15-word question
required a chunk containing all 15 stems and matched **zero rows**. Hybrid
silently degraded to dense. `replace(...::text, '&', '|')::tsquery` took it from
0 matches to 416.

**`FULL OUTER JOIN` + `COALESCE(..., 0)`** — a chunk found by only one arm must
still compete, scoring zero from the arm that missed it. An inner join would
return only chunks both arms found, which is the opposite of fusion's purpose.

One API wart, documented in the docstring: `SearchResult.score` means **cosine
distance (lower better)** from `search()` and **RRF score (higher better)** from
`hybrid_search()`. Both return best-first, which is all the eval harness depends
on.

---

## `embedding_cache.py` — exact-key cache

### `key(text, model, dimensions, task_type) -> str`

```python
sha256(f"{model}:{dimensions}:{task_type}:{text}")
```

All four components are load-bearing. Omitting `task_type` would serve a document
vector for a query. Omitting model or dimensions would serve a vector from a
different space.

### `get(...) -> list[float] | None`

The read path had a bug worth knowing:

```python
value = row[0]
if hasattr(value, "to_list"):     # psycopg returns a pgvector Vector object
    return value.to_list()
return [float(x) for x in value]
```

`register_vector` makes reads return a `Vector` object, not a list —
`TypeError: 'Vector' object is not iterable`. I had written and shipped the cache
having only ever tested the **write** path. `test_embedding_cache.py` exists
because of this.

### `put(...)`, `stats()`, `size()`

`put` upserts and bumps `hit_count` / `last_used_at`, which support LRU eviction
and give a reportable hit rate.

---

## `answer_cache.py` — semantic cache

### `lookup(embedding, corpus_version) -> CachedAnswer | None`

```sql
SELECT question, answer, sources, embedding <=> %s::vector AS distance
FROM answer_cache
WHERE corpus_version = %s
ORDER BY embedding <=> %s::vector
LIMIT 1
```

then, in Python:

```python
if distance > self._threshold:      # 0.12
    return None
```

**`WHERE corpus_version` comes first, then nearest-neighbour.** Order matters: a
cached answer was generated from a specific index. Serving it after a re-ingest
would return conclusions drawn from documents that no longer exist.

### `corpus_version() -> str`

A hash over the current chunk set. Changing the corpus changes the version, which
makes every existing cache row unmatchable — invalidation without a
cache-busting job.

### `store(question, embedding, answer, sources, corpus_version)`

Writes the question **text**, its vector, the answer, and the sources it came
from. The question vector is stored — not looked up from `embedding_cache` —
because the two tables answer different questions. `embedding_cache` answers
*"what is the vector for this exact string?"*; `answer_cache` needs a vector
**column it can run a nearest-neighbour scan over**. Same numbers, different job.

### `invalidate_stale(current_version) -> int`

Deletes rows from previous corpus versions. Housekeeping — `lookup` already
filters them out.

---

## `pipeline.py` — orchestration

### `ingest_documents(dry_run=False) -> IngestResult`

Five steps: load → split → diff → embed-and-store per batch → delete orphans.

```python
for start in range(0, len(new_chunks), batch_size):
    batch = new_chunks[start : start + batch_size]
    vectors = self._embeddings.embed_texts([c.content for c in batch])
    embedded += self._vector_store.add_chunks(batch, vectors)
```

**Write inside the loop, not after it.** Deliberately not "embed everything, then
store everything" — a rate-limit failure forty minutes into a run costs one batch
instead of the whole run, and re-running resumes because stored chunks are
skipped by the diff.

`dry_run` returns the counts without embedding anything. Every step before the
embed call is local, so the answer to *"what will this cost?"* is already known
by then.

Returns `IngestResult(total_chunks, embedded, skipped, deleted)` — a real record
of what happened, not a bare count.

### `query(question) -> QueryResult`

```
embed  →  answer-cache lookup  →  [hit? return]
       →  hybrid or dense search
       →  _build_context  →  _build_prompt  →  _generate_with_retry
       →  answer-cache store
```

The embedding happens **before** the cache check because the embedding is what
the cache searches with. It is also ~1% of the cost of what it can save.

Mode selection is a config read, not a code change:

```python
if self._config.rag.retrieval_mode == "hybrid":
    results = self._vector_store.hybrid_search(...)
else:
    results = self._vector_store.search(...)
```

Kept switchable rather than hardcoded because hybrid's advantage depends on the
**question distribution**, not just on the technique. It wins here because users
of Kubernetes docs type identifiers; a corpus whose users never do would see the
keyword arm contribute nothing, and `RETRIEVAL_MODE=dense` is one env var away.
Measuring is what tells you which case you are in.

### `_build_prompt(question, context)`

Instructs the model to answer **from the context only** and to say so when the
context does not contain the answer. That constraint is what makes the system
extractive rather than generative — and it is why the evaluation measures
retrieval, not answer text: if retrieval puts the right chunk in front of the
model, the answer follows.

### `_generate_with_retry(prompt, max_retries=3)`

Backoff of 5s / 10s / 20s on `ServerError`. Shorter than the embedding backoff
because generation limits are per-request, not per-minute-of-tokens.

---

## `schema.sql`

```sql
CREATE TABLE chunks (
    chunk_id        TEXT PRIMARY KEY,
    source          TEXT NOT NULL,
    chunk_index     INTEGER NOT NULL,
    content         TEXT NOT NULL,
    metadata        JSONB NOT NULL DEFAULT '{}',
    embedding       vector(768) NOT NULL,
    content_tsv     tsvector GENERATED ALWAYS AS
                    (to_tsvector('english', content)) STORED,
    embedding_model TEXT NOT NULL,
    embedding_dim   INTEGER NOT NULL
);

CREATE INDEX ... USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ... USING gin  (content_tsv);
```

**`GENERATED ALWAYS AS ... STORED`** means Postgres maintains `content_tsv` from
`content`. It cannot drift out of sync, because the application is not allowed to
set it. That is one line of schema in exchange for the entire keyword arm — on
Pinecone or Qdrant the same capability means implementing BM25, managing a
vocabulary, and keeping sparse vectors in sync yourself.

**The HNSW index is not used at 784 rows, and that is correct.** Measured with
`EXPLAIN ANALYZE`: sequential scan **2.3 ms**, forced index scan **73 ms**. 784
vectors is ~2.3 MB — it fits in memory and scanning it beats traversing a graph.
The index is built because the crossover is a corpus-size question: at a few
hundred thousand rows Postgres starts choosing it with no application change.
What the 768-dimension decision bought was **keeping that option open**, since
3072 would have made the index impossible to build at all.

---

## Interview questions this file answers

| Question | Where |
|---|---|
| Walk me through your ingestion pipeline | `pipeline.ingest_documents()` |
| How do you avoid re-embedding on every run | `existing_chunk_ids()` + content-derived IDs |
| What happens when you edit a document | ID changes → insert new, `delete_orphans()` removes old |
| What was the hardest bug | Chunk-ID collision — silent data corruption, and the test asserted the bug |
| How does hybrid search work | `hybrid_search()`, two CTEs, RRF |
| Why RRF instead of adding scores | Unbounded `ts_rank` vs bounded cosine — no common scale |
| How do you handle rate limits | `_embed_with_retry`, 30s floor, per-day detected separately |
| What is cached, and how | Two caches: exact-key for embeddings, semantic for answers |
| How do you invalidate the cache | `corpus_version` in the lookup predicate |
| Does the index actually help | No, and I measured it: 2.3 ms vs 73 ms |
