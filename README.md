# Hybrid RAG Service

Document question-answering over Kubernetes documentation, built to be
**measured** rather than demonstrated.

Retrieval quality is validated against a hand-verified evaluation set. Every
change is measured before and after, so improvements are attributable rather
than assumed.

---

## Results

Measured on **42 hand-verified questions** over **784 chunks** from 56
Kubernetes documentation pages.

| Retrieval | recall@1 | recall@5 | recall@10 | MRR |
|-----------|---------:|---------:|----------:|----:|
| **Dense (vector only)** | **0.52** | **0.81** | **0.93** | **0.66** |
| Hybrid, keyword weight 0.3 | 0.45 | 0.81 | 0.88 | 0.61 |
| Hybrid, equal weight (standard RRF) | 0.50 | 0.74 | 0.81 | 0.60 |

**Dense-only won.** Hybrid search — the standard recommendation for technical
documentation — lost at every keyword weighting tested (1.0, 0.5, 0.3, 0.2).

The breakdown shows why, in questions out of 21 per style:

| | exact_term in top 5 | paraphrase in top 5 | paraphrase at rank 1 |
|---|---:|---:|---:|
| Dense | 20 / 21 | **14 / 21** | **7 / 21** |
| Hybrid w=0.3 | **21 / 21** | 13 / 21 | 3 / 21 |
| Hybrid w=1.0 | **21 / 21** | 10 / 21 | 3 / 21 |

Hybrid reliably fixed **one** exact-identifier question and broke **four**
paraphrased ones. Dense already placed 20 of 21 exact-term questions in the top
5, so the keyword arm had almost nothing to add — while paraphrased questions
contain no rare terms by construction, so it contributed only noise. Even
weighted at 0.2, that noise was enough to displace the correct top-ranked chunk.

Reproduce with:

```bash
python -m scripts.evaluate --mode dense
python -m scripts.evaluate --mode hybrid --keyword-weight 0.3
python -m scripts.evaluate --compare
```

---

## What makes this different from a RAG demo

Most RAG projects stop at *"it returns plausible answers."* This one answers the
question that follows: **how do you know?**

- **42 hand-verified question-to-chunk pairs.** 60 were generated; 18 were
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

## What's next, and why

The failure list points at ranking, not recall. Of the 8 questions dense missed:

- **5** had the correct chunk at rank 6–10 — retrieval found it, ranking buried it
- **3** were absent from the top 10 entirely

That split says **reranking** is the highest-value next change, not chunking —
a reranker only reorders what retrieval already found, so it addresses the
majority case here.

In order:

1. **Cross-encoder reranking** — retrieve 50, rerank to 5. Directly targets the
   5 near-misses.
2. **Structure-aware chunking** — split on markdown headings and prepend the
   heading path. Targets the 3 that were never found: 7% of this corpus is Hugo
   build markup and 29 of 56 documents end with a link list.
3. **768 vs `halfvec(3072)`** — keep all dimensions at half precision instead.

Neither of the first two is implemented. Both require either a re-embed of the
corpus (a full day of free-tier quota) or a reranking model, and the evidence for
ordering them came from the baseline — which is the point of having measured it
first.

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
    vector_store.py         dense and hybrid search
    schema.sql              tables, HNSW and GIN indexes
    pipeline.py             orchestration; the only public entry point
  mcp_server/               MCP tools
  agent/                    Google ADK agent
scripts/                    fetch, ingest, evaluate, inspect, golden set tooling
eval/golden_set.json        42 hand-verified questions
tests/                      39 tests, integration against real Postgres
docs/                       architecture, plan, retrieval strategies, walkthrough
```

## Documentation

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

39 tests. The vector store suite runs against a real Postgres rather than mocks —
two of the three hardest bugs here were Postgres behaviour that no mock would
have caught.
