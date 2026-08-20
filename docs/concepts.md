# Concepts, explained

Questions asked while building this system, answered with numbers measured from
this corpus rather than generic examples.

Run the queries yourself with [`learn-search.sql`](learn-search.sql).

---

## Vectors and embeddings

### Is an embedding the same as a vector?

Yes. The schema says it literally:

```sql
embedding   vector(768)
   ↑            ↑
column name    its type
```

- **vector** — the general term: a list of numbers
- **embedding** — a vector that represents the *meaning* of something

Every embedding is a vector. In this project they're used interchangeably.

### What are the 768 numbers?

A representation of the text's meaning, produced by an AI model that read it.
You can't read them. Neither can I. The point is that a computer can compare two
of them.

```sql
SELECT vector_dims(embedding) FROM chunks LIMIT 1;   -- 768
```

### How does comparing them work?

The `<=>` operator returns the **cosine distance** — how far apart two
embeddings point.

```
0.0  identical meaning
1.0  unrelated
```

Measured on this corpus:

| Comparison | Distance |
|---|---:|
| Two chunks from the Secrets page | **0.0797** |
| Secrets page vs container images page | **0.2274** |

Related text produces a small number, unrelated text a big one. **That is the
entire mechanism.** Everything else is built on it.

---

## How search works

### What is search actually doing?

Sorting by that distance and keeping the top few.

```sql
ORDER BY c.embedding <=> q.embedding
LIMIT 5
```

That single line *is* `search()`. The rest of the Python is plumbing —
connections, type conversion, packaging results.

### Are we writing SQL manually, or is that just for learning?

The SQL **is** production. `vector_store.py` sends the same query, with `%s`
placeholders filled in at runtime — the same contract as a JDBC
`PreparedStatement`. Postgres can't tell the difference.

### Does it really compare against all 784 chunks?

Conceptually yes, mechanically no. The HNSW index navigates a graph and only
computes distance for a few dozen candidates.

```sql
CREATE INDEX ... ON chunks USING hnsw (embedding vector_cosine_ops);
```

At 784 chunks it barely matters. At 10 million it's the difference between
milliseconds and minutes. It's also why the 768-dimension decision mattered:
pgvector's HNSW caps at 2000 dimensions, so the model's native 3072 would have
left the table unindexed.

### Why only the top 5?

Because only those 5 chunks are sent to the model that writes the answer.
Anything at rank 6 or lower is never seen.

Configured in `.env` as `TOP_K_RESULTS=5`, read by `config.py`, passed by
`pipeline.py`, and ending up as a SQL `LIMIT`.

That is why the metric is **recall@5** — was the right chunk in the part that
actually reaches the model?

### Why does the eval fetch 10 instead of 5?

To learn *how badly* a miss missed. Rank 6 and rank 47 are different problems:
rank 6 means ranking needs a nudge, absent from the top 10 means retrieval never
found it at all.

---

## Keyword search

### What is `content_tsv`?

Postgres's word-based search format. It converts text into word stems with
positions:

```
INPUT:  'Kubernetes Secrets are created independently of the Pods'
OUTPUT: 'creat':4 'independ':5 'kubernet':1 'pod':8 'secret':2
```

Three things happened: common words dropped (`are`, `of`, `the`), words reduced
to stems (`created` → `creat`), positions recorded.

### Why stemming?

```
'create' → 'creat'      'created' → 'creat'      'creating' → 'creat'
```

All the same stem, so searching one finds the others. Without it, keyword search
would only match exact word forms.

### Do we write to that column?

No.

```sql
content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
```

`GENERATED ALWAYS` means Postgres maintains it from `content`. It cannot drift
out of sync, because you aren't allowed to set it.

### Is `tsvector` Postgres-only? What do other vector databases do?

`tsvector` is Postgres-specific, but keyword search exists everywhere. The
difference is who does the work.

**They build the index from your text:** Postgres, Elasticsearch/OpenSearch,
Weaviate. You send text, they handle tokenizing, stemming, ranking.

**You build it yourself:** Pinecone, Qdrant, Milvus. These only understand
vectors, so keyword search means computing *sparse vectors* yourself with BM25 or
SPLADE, then sending two vectors per chunk.

**Nothing at all:** ChromaDB. Which is why this project migrated away from it.

On Postgres, hybrid search cost one line of schema. On Pinecone it would have
meant a tokenizer, a BM25 implementation, vocabulary management, and keeping
sparse vectors in sync with the corpus.

### What would we implement with Pinecone instead?

One class instead of another. Everything else — loading, chunking, chunk IDs,
embedding, the pipeline, all five interfaces, the golden set, the harness — stays
identical.

```python
# Postgres
conn.execute("SELECT ... ORDER BY embedding <=> %s::vector LIMIT %s", ...)

# Pinecone
index.query(vector=query_embedding, top_k=5, include_metadata=True)
```

You'd lose hybrid search being one query, ad-hoc SQL inspection, and expressive
metadata filtering. You'd gain not having to tune an index. Worth it beyond
roughly 10 million vectors.

---

## The three tables

### What does each hold?

| Table | Rows | Holds | Embedded as |
|---|---:|---|---|
| `chunks` | 784 | **document text** from the Kubernetes docs | `RETRIEVAL_DOCUMENT` |
| `embedding_cache` | 48 | **question text** | `RETRIEVAL_QUERY` |
| `answer_cache` | 2 | **question text** + its answer | `RETRIEVAL_QUERY` |

Documents in one, questions in the other two.

### Why is a question's vector not in `chunks`?

Because a question is not a document. The corpus doesn't ask *"Why is Kubernetes
abbreviated as K8s?"* — it answers it.

They're near each other, never identical:

```
question vector  ←→  nearest chunk    0.2701   (close enough to retrieve)
question vector   =   any chunk        never
```

Retrieval finds the *nearest* chunk. It never looks for an equal one.

### Why does `task_type` matter?

The same model encodes text differently depending on its role:

```python
chunks:    config={"task_type": "RETRIEVAL_DOCUMENT"}
questions: config={"task_type": "RETRIEVAL_QUERY"}
```

A document is something to *be found*; a question is something *doing the
finding*. Identical text under those two settings produces two different vectors.

That is why `task_type` is part of the embedding cache key — serving a document
vector where a query vector belongs would degrade retrieval silently.

---

## Caching

### There are two caches. What's the difference?

| | `embedding_cache` | `answer_cache` |
|---|---|---|
| Lookup by | **exact text hash** | **vector similarity** |
| `k8s?` vs `k8s???` | different key → **miss** | 0.0173 apart → **hit** |
| Stores | 768 numbers | the generated answer |
| Saves | the embedding call | retrieval **and** generation |
| Share of request cost | ~1% | **~99%** |

The SQL contrast is the clearest summary:

```sql
-- embedding cache:  do I have THIS EXACT TEXT?
WHERE cache_key = 'abc123...'

-- answer cache:     do I have a question that MEANS THIS?
ORDER BY embedding <=> [vector] LIMIT 1
```

`=` versus *closest*.

### Why does `answer_cache` store the question vector when `embedding_cache` already has it?

Because they hold **different sets**.

```
embedding_cache   48 rows   every question ever embedded
answer_cache       2 rows   only questions that were ANSWERED
```

Of those 48 vectors, only 2 have an answer. The other 46 are evaluation
questions — embedded to measure retrieval, never answered, because the evaluation
is retrieval-only.

If the answer cache searched `embedding_cache` instead, it would find a close
match among those 46 and have **nothing to return**.

### If every question lands in both tables, how does it know which answer to give?

It never looks in `embedding_cache` for answers — that table has no answer
column. Demonstrated live:

```
after asking A:   embedding_cache 1    answer_cache 1
after asking B:   embedding_cache 2    answer_cache 1   ← unchanged
```

B's *text* was new, so it needed embedding (new row there). B's *meaning* wasn't
new — it matched A at 0.0959, returned A's answer, and generated nothing to
store.

They diverge by design:

| | Grows when | Ends up with |
|---|---|---|
| `embedding_cache` | a new **text** appears | one row per wording |
| `answer_cache` | an answer is **generated** | one row per meaning |

Ask the same thing ten ways: 10 rows in one, 1 row in the other.

### What is the 0.12 threshold matching on?

**Question vector against question vector.** Nothing else.

```sql
SELECT question, answer, embedding <=> %s::vector AS distance
FROM answer_cache
WHERE corpus_version = %s
ORDER BY embedding <=> %s::vector
LIMIT 1
```

Sort by distance from the new question, take the nearest, check whether it's
within 0.12. Not compared against chunks, not against answers, and no string
comparison happens at all.

### Where did 0.12 come from?

Measured on this corpus, not guessed:

```
same question, punctuation/case only     0.0093  ┐
same question, fully reworded            0.0529  ├─ must hit
same question, different words again     0.0704  ┘
                                         ------  threshold 0.12
closest two DIFFERENT questions          0.2110  ← must never hit
```

A clean gap between 0.07 and 0.21. Raising the threshold risks serving one
question's answer to another — **silently**, since a wrong cache hit is
indistinguishable from a right one. Lowering it only costs money.

**Err low.** A miss is expensive; a false hit is wrong.

### Does it actually work?

| Question | Distance | Result | Time |
|---|---:|---|---:|
| *"Why is Kubernetes abbreviated as K8s?"* | — | generated | 1947ms |
| same, repeated | 0.0000 | **HIT** | **82ms** |
| `why is kubernetes abbreviated as k8s???` | 0.0173 | **HIT** | 1079ms |
| *"Where does the K8s short name come from?"* | 0.0959 | **HIT** | 552ms |
| *"How do I configure resource limits?"* | — | generated | 8700ms |

**8700ms → 82ms.** Three different phrasings reused one answer; a genuinely
different question correctly did not.

The 82ms case had both caches hit. The 552–1079ms cases needed one embedding call
first — unavoidable, because the embedding is what the lookup *searches with*.

### Why does `answer_cache` carry a `corpus_version`?

Because re-ingesting the documents makes previous answers untrustworthy. Every
row records a fingerprint of the corpus it was generated against, and lookup
filters on it — so after an ingest, old answers stop matching rather than being
served against text that has since changed.

Without it, a cache hit after a corpus update is silently wrong.

---

## The models

### Which model finds the answer?

Neither. **Postgres finds it.**

```
question
   ├─ gemini-embedding-001    →  768 numbers      (text → numbers)
   ├─ Postgres  <=>  operator →  5 chunks         ← THE FINDING
   └─ gemini-3.1-flash-lite   →  written answer   (chunks → prose)
```

| | Input | Output | Cost |
|---|---|---|---|
| `gemini-embedding-001` | text | 768 numbers | 1 request |
| Postgres | a vector | 5 nearest chunks | free |
| `gemini-3.1-flash-lite` | question + those 5 chunks | an answer | ~98% of cost |

### What does the generation model actually see?

```
Use ONLY the information from the context below to answer the question.

CONTEXT:
[Source 1: concepts/overview/_index.md]
The name Kubernetes originates from Greek, meaning helmsman or pilot.
K8s as an abbreviation results from counting the eight letters between...

QUESTION: Why is Kubernetes abbreviated as K8s?
```

**It never sees the corpus** — only the 5 chunks retrieval handed it.

That's what "retrieval-augmented" means: the model isn't recalling Kubernetes
facts from training, it's reading text you gave it. Give it the wrong 5 chunks
and it writes a confident, wrong answer from them.

Which is why the evaluation measures retrieval and never scores the answers. If
the right chunk isn't in those 5, no model can rescue it.

---

## Choosing strategies

### How do teams decide which chunking and search strategy to use?

Two halves, and most people only know the first.

**Priors narrow the field.** Published guidance eliminates most options before
you write code — technical docs suggest structure-aware chunking and hybrid
search, code suggests AST-aware, tabular data suggests text-to-SQL rather than
RAG at all. That's why this project never considered Graph RAG.

**Then the failure list decides.** Run the baseline and read *which* questions
failed:

| Failures look like | Means | Fix |
|---|---|---|
| Exact-identifier questions miss, paraphrased ones pass | Embeddings smoothing away rare tokens | Hybrid search |
| Right document, wrong section | Chunks lost their heading context | Structure-aware chunking |
| Correct chunk at rank 6–10 | Retrieval found it, ranking buried it | Reranking |
| Retrieved a link list or table | Corpus contains non-prose | Preprocessing |
| Nothing relevant in the top 50 | Wrong corpus | Neither — fix the corpus |

You don't test all five. You measure once and one pattern usually dominates.

### So why measure at all, if best practice already tells you?

Three reasons, and this project demonstrated all of them.

**Direction isn't magnitude.** "Hybrid usually helps" doesn't say whether it's
worth the complexity.

**Your corpus has quirks the literature doesn't.**

**A broken implementation looks exactly like a working one.** The first hybrid
implementation here returned results *identical to dense*, because
`plainto_tsquery` joins every term with AND — a fifteen-word question required a
chunk containing all fifteen stems, and matched nothing. Tests passed, because
they used one-word queries.

And the result contradicted the prior: **hybrid search lost at every weighting
tested.** Dense already placed 20 of 21 exact-identifier questions in the top 5,
leaving the keyword arm nothing to add, while paraphrased questions have no rare
terms so it contributed only noise.

Without measuring, a regression would have shipped as an improvement.

---

## Related

- [`learn-search.sql`](learn-search.sql) — run these queries yourself
- [`architecture.md`](architecture.md) — the pipeline stage by stage
- [`retrieval-strategies.md`](retrieval-strategies.md) — every strategy, with costs
- [`queries.sql`](queries.sql) — database inspection queries
