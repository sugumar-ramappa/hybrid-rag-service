# Interview reference

Three documents, in the order you would walk someone through the project.

| # | Document | Answers |
|---|---|---|
| 1 | [**architecture-flow.md**](architecture-flow.md) | *"Walk me through how it works."* Diagrams for ingestion, query, and hybrid search, plus every model and strategy chosen and why |
| 2 | [**code-walkthrough.md**](code-walkthrough.md) | *"Show me the code."* Every class and method, in data-flow order, with the non-obvious decision inside each |
| 3 | [**testing.md**](testing.md) | *"How do you know it works?"* The 55 tests, the 43-question golden set, the harness, and what measuring actually found |

## The 60-second version

> A document QA system over 784 chunks of Kubernetes docs, on Postgres with
> pgvector. Retrieval is hybrid — vector search and Postgres full-text fused with
> reciprocal rank fusion — and it is **measured**, not asserted: hybrid took
> recall@5 from 0.95 to 0.98 and recall@1 from 0.67 to 0.74 across 43
> hand-verified questions.
>
> The number that matters is the breakdown, not the headline. **The entire gain
> is on identifier-based queries** — those went 0.95 → 1.00 at recall@5 and
> 0.71 → 0.86 at recall@1 — while paraphrased questions were completely
> unchanged. That is exactly what the technique predicts: the keyword arm can
> only contribute where the question and the answer share vocabulary. Labelling
> every golden-set question by style is what made that visible, rather than
> reporting "+3 points" and hoping it wasn't noise.

## Three things to have ready

**The bug worth telling.** Chunk IDs were a global counter, so adding one
document renumbered everything after it and the upsert silently overwrote
unrelated rows. No error — answers just came from the wrong pages. The existing
test *asserted the buggy behaviour*. Fixed with content-derived IDs, which is
also what made incremental ingest possible.
→ [code-walkthrough.md](code-walkthrough.md#_make_chunk_iddoc_key-chunk_index-content---str)

**The measurement worth showing.** `--compare` prints every labelled run side by
side, with per-question outcomes stored on disk. That turns "hybrid improved
recall" from an assertion into evidence.
→ [testing.md](testing.md#compare--the-audit-trail)

**The claim worth correcting yourself on.** The HNSW index is **not used** at 784
rows — sequential scan 2.3 ms, forced index scan 73 ms. Postgres is right to
ignore it. What the 768-dimension decision bought was keeping the index
*available* as the corpus grows, since the model's native 3072 exceeds pgvector's
2000-dim cap. Volunteering that reads far better than claiming the index makes
queries fast.

## Other project docs

| | |
|---|---|
| [`../architecture.md`](../architecture.md) | Full build log, stage by stage, with trade-offs |
| [`../concepts.md`](../concepts.md) | Q&A reference — vectors, search, caching, measured numbers |
| [`../retrieval-strategies.md`](../retrieval-strategies.md) | 9 chunking and 11 search strategies, and how the choice was made |
| [`../learn-search.sql`](../learn-search.sql) | Guided SQL walkthrough — run it against the real corpus |
