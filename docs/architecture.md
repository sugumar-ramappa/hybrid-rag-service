# Architecture and build log

A document question-answering system built to be **measured** rather than
demonstrated. Every retrieval change is validated against a hand-verified
evaluation set before it counts.

| | |
|---|---|
| Corpus | 784 chunks across 56 Kubernetes documentation pages |
| Golden set | 42 hand-verified questions (21 exact-term, 21 paraphrase) |
| Embeddings | `gemini-embedding-001` at 768 dimensions, normalized |
| Storage | Postgres 16 + pgvector, HNSW and GIN indexes |
| Tests | 36, including integration tests against real Postgres |

---

## 1. What this project is for

RAG over a document corpus is the most common portfolio project there is. Almost
all of them stop at *"it returns plausible answers."* The question that separates
them — and the first one asked in any serious interview — is **how do you know it
works?**

This project exists to answer that. The architecture is deliberately ordinary;
the evaluation is not. Building a golden set by hand, measuring a baseline,
changing one thing, and measuring again is the part almost nobody does, and it is
what turns "I built a RAG system" into a conversation about trade-offs, failure
modes, and evidence.

The sentence the project is built to earn:

> Dense-only retrieval scored recall@5 of X on 42 hand-verified questions. Adding
> hybrid search with reciprocal rank fusion took it to Y — and the gain was almost
> entirely on exact-identifier queries, which is where embeddings are weakest.

---

## 2. The pipeline, in order

Each stage consumes the previous stage's output.

### 01 · Fetch the corpus — `scripts/fetch_corpus.sh`

```
GitHub → documents/
```

A blobless sparse clone of `github.com/kubernetes/website`, checking out only
`content/en/docs/concepts`. 396 markdown files in about ten seconds, without
downloading a large site repository.

```bash
git clone --depth 1 --filter=blob:none --sparse "$REPO" "$tmp"
git -C "$tmp" sparse-checkout set content/en/docs/concepts
cp -R "$tmp/content/en/docs/concepts" ./documents/
```

The corpus is **not committed** — it belongs to the Kubernetes project, and a
fetch script makes it exactly reproducible without adding hundreds of files to
the repository. `documents/` is gitignored.

### 02 · Load — `src/rag/document_loader.py`

```
documents/ → list[Document]
```

Walks the tree recursively with `rglob`, skips any path containing a hidden
component, and records `source` as the path **relative to the corpus root**.

Both details were bugs first:

- `iterdir()` reads only the top level, so a nested docs tree loaded almost
  nothing.
- Once traversal recursed, a hidden-file check on the *filename* let every file
  under `.git/` through — those files have perfectly ordinary names. The check
  has to cover every path component.

Relative paths matter because a documentation tree contains dozens of files
called `_index.md`. A bare filename makes citations useless and weakens chunk
identity, which derives partly from source.

Load failures are counted and reported in the summary. Without a count, a
systematic problem is indistinguishable from a small corpus — the per-file stack
traces scroll past and the total merely looks lower than expected.

### 03 · Split — `src/rag/text_splitter.py`

```
Document → TextChunk[]
```

`RecursiveCharacterTextSplitter`, 1000 characters with 200 overlap, trying
paragraph breaks first, then lines, sentences, words. 784 chunks from 56
documents, averaging 794 characters.

Each chunk gets a **content-derived identity**:

```python
chunk_id = sha256(f"{doc_key}:{chunk_index}:{content}").hexdigest()[:16]
```

`chunk_index` counts within its own document. It used to be a global counter
across the corpus — which meant adding one document renumbered everything after
it, and the vector store's `upsert` silently overwrote unrelated documents'
chunks. No error, no warning: answers simply started coming from the wrong pages.

**Why this one decision carries the rest.** Content-derived IDs make the pipeline
content-addressable, like git's object store. *"Have I already embedded this?"*
becomes a primary-key lookup rather than a bookkeeping problem — which is what
makes incremental ingest, resumable runs, and self-healing possible at all. Under
the positional scheme, none of those could exist.

### 04 · Diff against what is stored — `src/rag/pipeline.py`

```
TextChunk[] → only the new ones
```

Ingestion is a sync, not a rebuild:

- **insert** chunks not yet stored
- **skip** chunks already embedded by this model at this size
- **delete** chunks a source no longer produces

```sql
SELECT chunk_id FROM chunks
WHERE chunk_id = ANY(%s)
  AND embedding_model = %s
  AND embedding_dim = %s
```

Model and dimension are part of the question because vectors from different
models occupy different spaces and cannot be compared — reusing across a model
change would corrupt the index silently.

Re-running over an unchanged corpus costs **zero API calls** and about five
seconds of local work.

### 05 · Embed — `src/rag/embeddings.py`

```
text → vector(768)
```

```python
config={"task_type": "RETRIEVAL_DOCUMENT",
        "output_dimensionality": 768}
```

The model returns 3072 dimensions by default, but pgvector's HNSW index caps at
2000 — an unindexed table means a sequential scan on every query. 768 is a size
the model supports natively through Matryoshka representation learning, where the
leading dimensions are trained to work as a complete embedding on their own.

Truncation leaves the vector no longer unit-length, so it is renormalized. Cosine
distance is scale-invariant and would not care; inner-product and L2 operators
would.

**Where the system is actually bottlenecked.** The free tier allows **100 texts
per minute** and **1,000 per day** — counted per text, not per API call, so
batching reduces HTTP overhead but not quota. Ingesting the full 396-file corpus
would have taken six days. That constraint fixed the corpus at 784 chunks, and
made incremental ingest a requirement rather than an optimisation.

Rate limits are retried with 30/60/120-second backoff — deliberately not the
usual 1–2 seconds, because the limit is per *minute* and retrying sooner cannot
succeed. Daily quota exhaustion is detected separately and fails immediately,
since it will not recover by waiting.

### 06 · Store — `src/rag/vector_store.py`, `src/rag/schema.sql`

```
chunk + vector → Postgres
```

One table holds both halves of retrieval:

```sql
chunk_id        TEXT PRIMARY KEY
embedding       vector(768)
content_tsv     tsvector GENERATED ALWAYS AS
                (to_tsvector('english', content)) STORED
embedding_model TEXT      -- recorded per row so a model change
embedding_dim   INTEGER   -- can be migrated blue/green

CREATE INDEX ... USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ... USING gin  (content_tsv);
```

`content_tsv` is maintained by Postgres and can never drift from `content`.
Nothing queries it yet — hybrid search is the next change.

Each batch of 50 is written before the next is embedded. A rate-limit failure
forty minutes into a run therefore costs one batch, and re-running resumes. That
happened, and cost nine chunks out of 778.

### 07 · Build the golden set — `scripts/generate_golden_set.py`

```
chunks → 42 verified question/chunk pairs
```

Sixty candidates, sampled **by document rather than by chunk**. The corpus is
skewed — one page holds 59 of 784 chunks — so uniform sampling would have drawn
8% of questions from a single document, and the resulting number would mostly
have measured retrieval on that page.

```sql
row_number() OVER (PARTITION BY source ORDER BY random())
-- one chunk from every document before any document contributes two
```

Each candidate is generated from **one chunk in isolation** — the model never
sees the corpus, so it cannot know whether twenty other chunks answer its
question equally well. That judgement is what hand-verification supplies, and why
18 of 60 were rejected.

Two question styles are generated deliberately:

| Style | Phrasing | Weak for |
|---|---|---|
| `exact_term` | names identifiers a reader would type — `UnknownVersionInteroperabilityProxy` | dense retrieval |
| `paraphrase` | avoids the vocabulary — *"how many copies of the main management program are running?"* | keyword search |

Labelling them means the effect of hybrid search can be **attributed**, not
merely observed.

### 08 · Measure — `scripts/evaluate.py`

```
42 questions → recall@k, MRR
```

Embed each question, search, find where the expected chunk ranked. No generation,
no LLM judging — retrieval only, which is what makes it cheap enough to run after
every change.

- **recall@k** — did the right chunk appear at all, within the top k
- **MRR** — how high. recall@5 cannot distinguish rank 1 from rank 5, and a model
  attends far more to the first passage in its context than the fifth

Query embeddings are cached in Postgres, keyed on model, dimensions, **task
type** and text. Task type belongs in the key because chunks embed as
`RETRIEVAL_DOCUMENT` and queries as `RETRIEVAL_QUERY` — the same text under each
produces a different vector, and a key omitting it would serve a document vector
for a query.

---

## 3. One core, five delivery channels

```
CLI  ·  FastAPI  ·  Streamlit  ·  MCP server  ·  Google ADK agent
                        │
                   RAGPipeline
                        │
      loader → splitter → embeddings → vector store
                        │
              Postgres 16 + pgvector
```

Nothing except `RAGPipeline` talks to the vector store, embedding service, or
splitter directly.

That separation was **tested rather than assumed**. Replacing ChromaDB with
Postgres rewrote `vector_store.py` entirely — 150 lines, new dependencies, new
schema, three debugging sessions — and changed exactly one line in the pipeline's
constructor. All five interfaces were untouched.

The distinction worth naming is between a *capability* and a *delivery channel*.
"Answer questions about these documents" is one capability; a REST endpoint, a
chat UI, and an MCP tool are three ways to reach it. Most portfolio projects
conflate the two and end up with retrieval logic inside a Streamlit callback.

---

## 4. Technology choices and their costs

| Choice | Why | What it costs | Switch when |
|---|---|---|---|
| **Postgres + pgvector** over ChromaDB | Vector and full-text in the same row, so hybrid is one SQL query with metadata filtering in the same `WHERE`. Chroma has no keyword search. | HNSW caps at 2000 dimensions — a constraint Chroma does not have, which forced the embedding size down. | Beyond ~10M vectors, or when ANN performance dominates: Vertex AI Vector Search. |
| **768 dimensions** over the native 3072 | Fits under the HNSW limit, indexes normally, 4× less storage. A size the model supports directly. | Some retrieval quality, unmeasured. `halfvec(3072)` would keep all dimensions at half precision and probably retrieve better. | After the baseline exists — it becomes a measurable experiment rather than a guess. |
| **Content-derived chunk IDs** | Idempotent, resumable, self-healing ingestion. Change detection stops being a feature that needs building. | Position is part of identity, so an edit shifting chunk boundaries re-embeds chunks whose text never changed. | Never — the alternative is silent data corruption. |
| **Per-batch writes** over embed-all-then-store | A 45-minute rate-limited run *will* hit 429s. Persisting each batch means failure costs one batch. | More round trips; a partially ingested corpus becomes a valid intermediate state that must be detectable. | Never, for any long rate-limited job. |
| **Retrieval-only evaluation** | No generation calls, so a full run is seconds and free. Retrieval is upstream — wrong chunk, wrong answer regardless. | Says nothing about answer quality, faithfulness, or hallucination. | Once retrieval is good enough that answers become the bottleneck. |
| **Cache in Postgres** over Redis | Already running, already persistent, shared across instances, no extra service or cost. | ~1ms instead of ~0.1ms; TTL and eviction hand-rolled rather than native. | When volume or latency budget justifies another service. The two-method interface makes it a new class, not a refactor. |
| **Fixed-size chunking** | The baseline everything else is measured against. | Cuts inside code blocks and strips headings from the text they describe. Known to be suboptimal here. | Immediately after the baseline — structure-aware chunking is first in the queue. |

---

## 5. Scripts

| Script | Does | API cost |
|---|---|---|
| `fetch_corpus.sh` | Sparse-clones Kubernetes docs into `documents/` | none |
| `ingest.py` | Load → split → diff → embed → store. Flags: `--dir`, `--max-files`, `--dry-run` | new chunks only |
| `check_limits.py` | Probes the real per-request batch ceiling for your key, so batch size comes from evidence | ~12 calls |
| `inspect_chunking.py` | Shows blocks, chunks and actual overlap for any file — answers "why did this chunk badly?" | none |
| `generate_golden_set.py` | Samples by document, writes one question per chunk in two styles | 1 per candidate |
| `check_golden_set.py` | Flags questions restating their own chunk; can regenerate one style in place | only with `--regenerate` |
| `review_golden_set.py` | Interactive verification, saving after every decision | none |
| `evaluate.py` | recall@1/5/10 and MRR by style, plus the failure list | cached after run one |
| `query.py` | Ask a question from the terminal | 1 per query |

### The dry run is the completeness check

`ingest --dry-run` recomputes what the files *should* produce and compares
against what the database holds. `0 would be embedded` proves the two agree
exactly.

An earlier SQL check compared each document's chunk count against its highest
index — which catches gaps but is structurally blind to a missing tail, the most
likely outcome of an interrupted run. It passed while a document sat at 2 chunks
of 11.

---

## 6. What building it surfaced

### Bugs

| Bug | Why it was invisible |
|---|---|
| Chunk IDs from a global counter | Adding one document renumbered everything after it, and `upsert` overwrote unrelated chunks. No error — answers came from the wrong document. The test suite **asserted the buggy behaviour was correct**. |
| Postgres cast asymmetry | pgvector defines an *assignment* cast from `double precision[]` to `vector`, so `INSERT` worked. Operator resolution ignores assignment casts, so `<=>` failed. Every write test passed; every search test failed. |
| `register_vector` bootstrap deadlock | It looks up the `vector` type, but the code that *creates* the extension went through the same connection helper. A fresh database could never bootstrap. |
| Undeclared SDK dependency | `requirements.txt` declared the legacy SDK, which nothing imported, and omitted the current one, which everything used. It resolved only via a transitive dependency. |
| Question generator leakage | A prompt saying "use the identifiers that appear in the text" was read as "reuse the phrasing" — 78% vocabulary overlap between questions and their own answers. Those measure string matching, and would have reported near-perfect recall. |

### Measurements

- **Quota is counted per text, not per API call** — batching helps HTTP overhead
  but not throughput. 100/minute, 1,000/day, capping ingestion at 1,000 chunks a
  day whatever the batch size.
- **7% of the corpus is Hugo build markup** — shortcodes and URLs inside the
  embeddings, diluting the semantic signal.
- **29 of 56 documents end with a link list**, chunking into navigation with no
  explanatory content.
- **30% of generated questions were rejected** — link-list chunks,
  question/chunk mismatches, markup dumps, questions restating their source.
- **Overlap is a ceiling, not a target.** Configured at 200 characters, it
  delivered 27 — the splitter carries whole blocks, and the next block back was
  847 characters.

---

## 7. What building it taught me

**A green test suite means the code matches your assertions, not that it is
correct.** The chunk-ID bug survived because a test asserted indices should run
0..n across the whole corpus. That assertion *encoded the bug* — any correct fix
would have made it fail, and the instinct is to "fix the test". The same shape
appeared twice more: a fixture catching every exception and reporting real
failures as "skipped", and a completeness check blind to a missing tail.

**Make identity intrinsic and whole categories of bookkeeping disappear.**
Positional IDs needed change detection, ingestion manifests, and repair logic —
none of which existed, which is why corruption was silent. Hashing content made
the database itself the record. Git works this way. The move is always the same:
stop asking *where is it* and start asking *what is it*.

**Validate the measuring instrument before trusting the measurement.** The first
golden set had questions restating their own answers. It would have reported near
perfect recall, shown no improvement from hybrid search, and given no clue why.
Thirty seconds of checking word overlap caught it. Bad measurement is worse than
no measurement, because it looks like success.

**Constraints reshape architecture, and that is not a compromise.** A
1,000-per-day quota is why incremental ingest exists — without it a second run
would have blocked work for 24 hours. The same limit is why writes are per-batch.
Both are what production looks like anyway; the constraint just made skipping
them impossible.

**Fail where the mistake is, not three layers downstream.** A dimension mismatch
between config and schema would otherwise surface as an opaque insert error
halfway through ingestion. Checking at startup turns twenty minutes of confusion
into one sentence.

**Widen the output before theorising.** An ingest silently read the wrong
directory for hours. The path was printed on the first line of every run — and
filtered out by a `grep` narrowed to the lines that seemed interesting.

**Deferring an improvement until it can be measured is discipline, not
procrastination.** Structure-aware chunking is very likely the largest available
win here. It is still not implemented, because doing it before a baseline exists
means never knowing what it was worth.

---

## 8. Questions this project lets you answer

**How do you know your retrieval works?**
42 hand-verified question-to-chunk pairs, measured with recall@1/5/10 and MRR,
broken down by question style. 18 of 60 candidates rejected — that rejection rate
is the point, not a defect. The generator sees one chunk at a time and cannot
know whether another chunk answers better; that criterion is exactly what human
verification supplies.

**Walk me through debugging bad retrieval.**
Bisect before fixing — is the correct chunk in the top 50? **No** → recall
problem, look at chunking, then add lexical search. **Yes but not top 5** →
ranking problem, add a reranker. **Yes, top 5, bad answer** → not retrieval at
all. A reranker cannot fix a recall problem; it only reorders what retrieval
found.

**How would you migrate to a different embedding model?**
Full re-embed, always — vectors from different models occupy different spaces.
`embedding_model` and `embedding_dim` are stored per row precisely so this is
possible: build alongside, compare both against the golden set, cut over, keep
the old index until confident. Never mutate in place.

**Tell me about a hard bug.**
The chunk-ID collision. The interesting part is not the bug — it is that the test
suite was green, because a test was defending it.

**What would you do next, and why haven't you?**
Structure-aware chunking, then reranking, then stripping the Hugo markup. All
deferred for the same reason: no baseline to attribute them to yet. Changing two
things between measurements produces one number and no attribution.

**Why Postgres rather than a vector database?**
Retrieval needs two kinds of search. Embeddings are weak on exact identifiers;
keyword search is weak on paraphrase. Postgres holds both columns in the same
row, so hybrid is one query rather than two systems merged in application code.
It cost something: HNSW caps at 2000 dimensions, which forced the embedding size
down to 768.

---

## Related

- `docs/plan.md` — remaining steps and the rules that keep it on track
- `docs/retrieval-strategies.md` — chunking and search strategies considered
- `docs/walkthrough.md` — hands-on run through the pipeline with measured results
- `docs/queries.sql` — inspection queries for the database
