# Hybrid RAG Service

Document question-answering over Kubernetes documentation, built to be
**measured** rather than demonstrated.

Retrieval quality is validated against a hand-verified evaluation set. Every
change is measured before and after, so improvements are attributable rather
than assumed.

---

## Results

Measured against **43 hand-verified questions** over **784 chunks** from 56
Kubernetes documentation pages.

| Retrieval | recall@1 | recall@5 | recall@10 | MRR |
|-----------|---------:|---------:|----------:|----:|
| Dense (vector only) | 0.67 | 0.95 | 0.98 | 0.80 |
| Hybrid (vector + BM25 + RRF) | 0.74 | 0.98 | 0.98 | 0.83 |
| **Hybrid + cross-encoder rerank** | **0.72** | **0.98** | **0.98** | **0.83** |

On this 43-question set the three are within one question of each other, which is
why a **267-pair set** was built from the corpus's own structure. There, hybrid +
reranking is the clear winner and the ordering above reverses — see
[Reranking](#reranking--implemented-and-the-result-reversed-itself).

| 267 structural pairs | recall@1 | recall@5 | MRR |
|---|---:|---:|---:|
| dense | 0.509 | 0.921 | 0.677 |
| hybrid | 0.674 | 0.951 | 0.793 |
| **hybrid + rerank** | **0.730** | **0.974** | **0.828** |

Hybrid wins, and the gain lands exactly where the theory predicts — on questions
naming a specific identifier:

| | recall@1 | recall@5 |
|---|---:|---:|
| `exact_term` — names things like `UnknownVersionInteroperabilityProxy` | 0.71 → **0.86** | 0.95 → **1.00** |
| `paraphrase` — natural phrasing, no identifier | 0.64 → 0.64 | 0.95 → 0.95 |

**Every point of improvement came from identifier queries.** Paraphrased
questions were unchanged on recall and marginally worse on MRR (0.78 → 0.75) —
the residue of keyword noise.

That asymmetry is the result, not a footnote. Keyword search can only contribute
where a question carries a **rare, distinctive term**, so a question naming
`reclaimPolicy` gives the keyword arm a rare term to rank on, while *"how does
the cluster decide where to run things"* gives it nothing. Labelling every
question by style is what makes the mechanism visible in the numbers — without
the split this is "+3 points, possibly noise."

It also means a recall figure quoted without describing its question
distribution says very little. Ours is 21 identifier-based and 22 naturally
phrased, and the per-style rows are reported for exactly that reason.

Reproduce with:

```bash
python -m scripts.evaluate --mode dense  --label dense-v2
python -m scripts.evaluate --mode hybrid --label hybrid-v2
python -m scripts.evaluate --compare
```

---

## What makes this different from a RAG demo

Most RAG projects stop at *"it returns plausible answers."* This one answers the
question that follows: **how do you know?**

- **43 hand-verified question-to-chunk pairs.** 60 were generated; 17 were
  rejected for being link-list chunks, question/chunk mismatches, or questions
  that restated their own answer. That 30% rejection rate is the point of
  verification, not a defect in it.
- **Retrieval-only metrics.** recall@k and MRR need no generation calls, so a
  full evaluation is seconds and free — cheap enough to run after every change.
- **One change per measurement.** Chunking improvements are deliberately deferred
  until after the search comparison, because changing two things at once produces
  one number and no attribution.

---

## Quick start

> **This container's Postgres server is shared.** `vendor-onboarding` keeps its
> measurement cache in a `vendor_onboarding` database inside the same `ragdb`
> container, because this project claimed port 5432 first.
>
> **Do not `docker rm ragdb` and recreate it** — the `docker run` command above will
> silently destroy that project's cache, which is several days of rate-limited model
> calls. Stopping the container is safe; removing it is not.


```bash
# 1. Postgres with pgvector
docker run -d --name ragdb -p 5432:5432 \
  -e POSTGRES_PASSWORD=dev -e POSTGRES_DB=ragdb \
  pgvector/pgvector:pg16

# 2. Environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add GOOGLE_API_KEY and DATABASE_URL

# 3. Corpus (not committed - fetched reproducibly)
./scripts/fetch_corpus.sh

# 4. Ingest
python -m scripts.ingest --dry-run     # what would this cost?
python -m scripts.ingest

# 5. Ask something
python -m scripts.query "how do I keep data after a pod restarts"

# 6. Measure
python -m scripts.evaluate --mode dense
```

Free-tier note: embedding is capped at **100 texts/minute and 1,000/day**,
counted per text rather than per API call. A full ingest of this corpus is about
45 minutes. Re-running costs nothing — see incremental ingest below.

---

## Architecture

```
CLI  ·  FastAPI  ·  Streamlit  ·  MCP server  ·  Google ADK agent
                        │
                   RAGPipeline
                        │
      loader → splitter → embeddings → vector store
                        │
              Postgres 16 + pgvector
```

Nothing except `RAGPipeline` touches the vector store, embedding service or
splitter. That separation was tested rather than assumed: replacing ChromaDB with
Postgres rewrote `vector_store.py` entirely — 150 lines, new dependencies, new
schema — and changed **one line** in the pipeline constructor. All five
interfaces were untouched.

### Retrieval

One table holds both halves of the search:

```sql
embedding    vector(768)                    -- HNSW index, cosine distance
content_tsv  tsvector GENERATED ALWAYS AS   -- GIN index, keyword search
             (to_tsvector('english', content)) STORED
```

Hybrid search runs both arms in a single query and fuses them by **rank** rather
than score — `ts_rank` is unbounded and cosine distance is bounded, so the two
cannot be added meaningfully:

```
score(doc) = 1/(60 + dense_rank) + 1/(60 + keyword_rank)
```

A chunk found by both arms outranks one found by only one. No normalisation, no
per-corpus tuning.

### Incremental ingest

Ingestion is a sync, not a rebuild: **insert** new chunks, **skip** those already
embedded by this model at this size, **delete** chunks a source no longer
produces.

This works because chunk identity is content-derived:

```python
chunk_id = sha256(f"{source}:{chunk_index}:{content}")[:16]
```

Re-running over an unchanged corpus costs **zero API calls**. Editing one
document re-embeds only the chunks whose text actually changed.

---

## Key decisions

| Choice | Why | What it cost |
|--------|-----|--------------|
| Postgres + pgvector over ChromaDB | Vector and full-text in one row, so hybrid is one query with metadata filtering in the same `WHERE` | HNSW caps at 2000 dimensions — a limit Chroma doesn't have |
| 768 dimensions over the native 3072 | Fits under that cap, indexes normally, 4× less storage | Some retrieval quality, unmeasured |
| Content-derived chunk IDs | Idempotent, resumable, self-healing ingestion | An edit shifting chunk boundaries re-embeds unchanged text |
| Per-batch writes | A 45-minute rate-limited run *will* be interrupted; failure costs one batch | More round trips; partial corpora become a valid state |
| Retrieval-only evaluation | No generation calls, so a run is seconds and free | Says nothing about answer quality or hallucination |

Full reasoning, including rejected alternatives, in
[`docs/architecture.md`](docs/architecture.md).

---

## What building it surfaced

**A green test suite defended a data-corruption bug.** Chunk IDs came from a
global counter, so adding one document renumbered everything after it and the
upsert silently overwrote unrelated chunks. A test asserted indices should run
`0..n` across the whole corpus — encoding the bug, so any correct fix would have
failed it.

**Postgres cast contexts are asymmetric.** pgvector defines an *assignment* cast
from `double precision[]` to `vector`, so `INSERT` worked. Operator resolution
ignores assignment casts, so `<=>` failed. Every write test passed; every search
test failed.

**The first golden set measured string matching, not retrieval.** A prompt saying
"use the identifiers that appear in the text" was read as "reuse the phrasing" —
78% vocabulary overlap between questions and their own answers. It would have
reported near-perfect recall on the first run and shown no improvement from
hybrid search.

**Quota is counted per text, not per API call.** Batching reduces HTTP overhead
but not throughput. That capped ingestion at 1,000 chunks/day and made
incremental ingest a requirement rather than an optimisation.

---

## Reranking — implemented, and the result reversed itself

The baseline pointed here. Of the 8 questions dense retrieval missed, **5 had the
correct chunk at rank 6-10** — found, but buried — and 3 were absent entirely. A
reranker only reorders what retrieval returned, so it targets the five.

`retrieve 50 → cross-encoder → keep 5`, using `ms-marco-MiniLM-L-6-v2` (90 MB,
local, no API key).

### Measured on 267 structural pairs

| config | recall@1 | recall@5 | MRR |
|---|---:|---:|---:|
| dense | 0.509 | 0.921 | 0.677 |
| dense + rerank | 0.678 | 0.944 | 0.788 |
| hybrid | 0.674 | 0.951 | 0.793 |
| **hybrid + rerank** | **0.730** | **0.974** | **0.828** |

**Latency:** dense 51 ms · hybrid 30 ms · +rerank 195 ms.

**Reranking stacks with hybrid rather than replacing it.** It lifts dense by
0.169 and hybrid by 0.056 on recall@1. Keyword fusion and semantic reordering fix
*different* failures, so they are additive — which was not the expectation going
in.

### The 43-question set gave the wrong answer

| config | 43 questions | 267 pairs |
|---|---:|---:|
| hybrid | 0.744 | 0.674 |
| hybrid + rerank | **0.721** *(worst)* | **0.730** *(best)* |

On 43 questions, adding reranking to hybrid looked like a regression and was
written up as one. On 267 pairs it is the clear winner.

**The difference was one question.** On 43, a single flipped answer moves recall@1
by 2.33 points; on 267 it moves 0.33. The small set could not distinguish a real
effect from a coin flip, and two confident conclusions were drawn from it anyway:
"reranking does not help here" and "hybrid and reranking are alternatives, pick
one". Both wrong.

### Fusing beats replacing

The first implementation replaced the retrieval order with the cross-encoder's
score outright. Measured, that is not neutral:

```
exact_term   recall@1  0.857 -> 0.810     lexical signal discarded
paraphrase   recall@5  0.955 -> 1.000     semantic judgement gained
```

Identifier questions were already correct *because* the literal token matched. A
cross-encoder scores semantic plausibility, so it will promote a passage that
reads like a better answer over one that actually contains `reclaimPolicy`.

Fusing by rank — the same RRF the hybrid query already uses, applied one level up
— keeps both signals. By rank, not score, for the same reason: cosine distance is
bounded, RRF scores are small positives, and a cross-encoder logit is unbounded
and here ranges about -11 to -2. Those three cannot be added.

---

## A 267-pair evaluation set, built from the corpus's own structure

`recall@5 = 1.000` on 43 questions is 43 of 43. One awkward new question drops it
to 0.977, so a two-point difference between configurations is one question. Every
conclusion above was drawn on that basis, and one of them was wrong.

The usual fix is a public benchmark with human relevance judgements. This does
something different, because **the corpus already contains the labels**:

```
## Reclaiming
When a user is done with their volume, they can delete the PVC...
```

A heading names a topic; the prose beneath it covers that topic. So
`(heading, the chunk holding that prose)` is a labelled retrieval pair — 267 of
them here, with no annotation, no LLM and no quota.

```bash
python -m scripts.build_structural_set
python -m scripts.evaluate --golden-set eval/structural_set.json --mode hybrid --rerank
```

### What it measures, and what it does not

A heading is a **topic label**, not a user question. Nobody asks "Configuring the
kubelet"; they ask "how do I change kubelet settings". So this set measures
*topical* retrieval, which is related to but not the same as question answering —
and the pairs are marked `provenance: structural`, never `verified`, so a
267-pair set cannot inherit the credibility of the 43 that a person actually read.

The two sets are complementary rather than competing:

| | phrasing | intent | size |
|---|---|---|---|
| 43 hand-verified | real | real | too small to trust a 2-point difference |
| 267 structural | thin | none | large enough that 15 pairs is not noise |

### And they agree

| hybrid + rerank | 43 hand-verified | 267 structural |
|---|---:|---:|
| recall@5 | 0.977 | 0.974 |
| MRR | 0.826 | 0.828 |

Two sets built by completely different methods, agreeing to within a percentage
point. That convergence is stronger evidence than either alone: 0.97 recall@5 is
a property of the system, not of how one person worded 43 questions.

### The bug that cost 7 points

32 of the first 301 pairs — **11% of the set** — had the Hugo shortcode
`{{% heading "whatsnext" %}}` as the query. The noise filter stripped shortcodes
from the body and never from the heading, so they passed straight through: 32
identical meaningless queries pointing at 32 different answers, which can only
score as misses.

Removing them lifted **every** configuration by roughly 7 points. A corrupted
evaluation set does not announce itself — it just makes everything look uniformly
worse, which reads as a hard problem rather than a broken measurement.

---

## Running on Kubernetes

Deployed to a local k3s cluster — Deployment, headless Service, Ingress via
Traefik, Postgres as a StatefulSet with a PersistentVolumeClaim, the API key as a
Secret and tuning as a ConfigMap. Setup and reasoning in
[`k8s/README.md`](k8s/README.md).

```
rag-postgres-0     1/1 Running    StatefulSet + 2Gi PVC, pgvector/pgvector:pg16
rag-service        1/1 Running    liveness /healthz · readiness /stats
ingress            rag.localhost via Traefik
```

**The point is not that it runs.** It is that containerising and scheduling it
asked two questions local development never had to answer, and both had been
wrong since the API was written:

**`api.py` was never in the image.** The Dockerfile copied `src/`, `documents/`
and `scripts/` — not `api.py`. `fastapi` and `uvicorn` were absent from
`requirements.txt` too; the file's own docstring said `pip install fastapi
uvicorn`. **The HTTP API had never once run in a container.** Developing it with
`uvicorn api:app` at a shell hid that completely, because the shell had the
dependencies the image did not.

**There was no liveness endpoint.** The only candidate was `/stats`, which builds
the pipeline and queries the corpus. Kubernetes answers a liveness failure by
*restarting* the container, so pointing liveness at the database means a database
blip restarts every healthy pod — a short outage becomes a crash loop that
outlives its cause. `/healthz` was added to check nothing but the process;
readiness is the probe allowed to touch dependencies, because failing it removes
the pod from the Service rather than killing it.

**On the corpus.** The cluster database starts empty and deliberately does not
point at any Postgres on the host. Rather than re-embed, the existing 784 chunks
and 337 cached embeddings were copied with `pg_dump | psql` — read-only on the
source, a few seconds, and ~784 embedding calls not spent.

**And a query worth keeping.** The first question asked was about reclaiming
persistent volumes, which this 56-document corpus does not cover. The service
answered *"the provided context does not contain information on how to reclaim a
persistent volume"* — it declined instead of fabricating from the wrong chunks.
That is a better demonstration than the query that worked.

## What's next

**Not chunking.** This README previously recommended structure-aware chunking to
rescue "the 3 that were never found". At recall@5 of 0.974 they *are* found —
that recommendation is retracted, and acting on it would have spent a full day of
embedding quota on a solved problem.

**Not more retrieval tuning either.** 0.974 recall@5, corroborated by two
independent sets, is above a normal production bar for RAG — the number that
matters is whether the answer reaches the context the model sees, and it does.

The remaining weakness is not retrieval at all: **nothing here measures what the
system does when retrieval fails.** Seven of 267 queries have no correct chunk in
the top 5. Whether the answer is then "I don't know" or a confident fabrication
from the wrong chunk matters far more than the 2.6%, and this harness stops at
retrieval by design — generation quality was explicitly out of scope, which is
what made a full evaluation free and repeatable.

That is the honest next project, and it is a different one.

---

## Project layout

```
src/
  config.py                 configuration from environment
  rag/
    document_loader.py      recursive walk, path-relative sources
    text_splitter.py        chunking + content-derived identity
    embeddings.py           Gemini, 768-dim, retry with backoff
    embedding_cache.py      query embeddings cached in Postgres
    reranker.py             cross-encoder reranking, and RRF fusion with retrieval
    vector_store.py         dense and hybrid search
    schema.sql              tables, HNSW and GIN indexes
    pipeline.py             orchestration; the only public entry point
  mcp_server/               MCP tools
  agent/                    Google ADK agent
scripts/                    fetch, ingest, evaluate, inspect, golden set tooling
eval/golden_set.json        43 hand-verified questions
eval/structural_set.json    267 pairs derived from headings - no annotation
tests/                      55 tests, integration against real Postgres
docs/                       architecture, plan, retrieval strategies, walkthrough
```

## Documentation

**Start here:** [`docs/interview/`](docs/interview/) — three documents covering
the architecture flow with method names, a method-by-method code walkthrough, and
how the system was tested and measured.

- [`docs/interview/architecture-flow.md`](docs/interview/architecture-flow.md) —
  ingestion, query, and hybrid-search diagrams; models and strategies chosen
- [`docs/interview/code-walkthrough.md`](docs/interview/code-walkthrough.md) —
  every class and method in data-flow order
- [`docs/interview/testing.md`](docs/interview/testing.md) — the test suite, the
  golden set, the eval harness, and what measuring found
- [`docs/architecture.md`](docs/architecture.md) — the pipeline stage by stage,
  technology trade-offs, findings, and interview answers
- [`docs/plan.md`](docs/plan.md) — remaining steps and the rules that keep them honest
- [`docs/retrieval-strategies.md`](docs/retrieval-strategies.md) — chunking and
  search strategies considered
- [`docs/walkthrough.md`](docs/walkthrough.md) — hands-on run with measured results
- [`docs/queries.sql`](docs/queries.sql) — database inspection queries

## Tests

```bash
pytest tests/ -v
```

55 tests. The vector store suite runs against a real Postgres rather than mocks —
two of the three hardest bugs here were Postgres behaviour that no mock would
have caught.
