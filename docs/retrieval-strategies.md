# Retrieval strategies

Design rationale for how this service chunks documents and searches them, plus
the alternatives that were considered.

**Full reference:** https://claude.ai/code/artifact/aa382e11-55b8-4cb1-be20-0e93d7dcdb77
(nine chunking strategies, eleven search strategies, evaluation metrics)

---

## What this service currently does

| Stage | Implementation | Where |
|-------|---------------|-------|
| Chunking | Recursive character split, 1000 chars, 200 overlap | `src/rag/text_splitter.py` |
| Chunk identity | SHA-256 of source, position, content | `src/rag/text_splitter.py` |
| Embedding | `gemini-embedding-001` truncated to 768 dimensions, normalized | `src/rag/embeddings.py` |
| Storage | Postgres with pgvector, HNSW index | `src/rag/vector_store.py` |
| Keyword index | Generated `tsvector` column, GIN index | `src/rag/schema.sql` |
| Search | Dense only (cosine distance) | `src/rag/vector_store.py` |

The `tsvector` column exists but is not yet queried. Hybrid search is the next
planned change, and dense-only is deliberately kept as the measured baseline.

## Why these choices

**Postgres over ChromaDB.** Chroma has no keyword search, so hybrid retrieval
would have meant a separate in-memory BM25 index merged in application code.
Postgres holds the vector and the `tsvector` in the same row, making hybrid a
single query with metadata filtering available in the same `WHERE` clause.

**768 dimensions.** pgvector's HNSW index supports at most 2000 dimensions and
the model returns 3072 by default, which would have left the table unindexed.
768 is a size the model supports natively via Matryoshka truncation. Kept
configurable via `EMBEDDING_DIM` so the eval harness can measure the cost.

**Content-derived chunk IDs.** The original implementation numbered chunks with
a corpus-wide counter, so adding one document shifted every later chunk's ID and
the upsert silently overwrote unrelated documents. Identity is now derived from
the chunk itself, which also makes re-ingestion idempotent.

**No graph, no late interaction, no semantic chunking.** All were considered and
rejected as premature. There is no measurement yet to justify their cost.

## Planned, in order

Each change is measured independently against the golden set. Two changes at
once produce one number and no attribution.

1. **Golden set and eval harness** — recall@5 and MRR. No quality gain; makes
   every later gain measurable.
2. **Hybrid search with RRF** — fuse `tsvector` and vector rankings by position.
   Expected to help most on exact technical terms (`PersistentVolumeClaim`,
   `kubectl`) where dense retrieval is weakest.
3. **Structure-aware chunking** — split on markdown headings and prepend the
   heading path before embedding. Likely the larger win on this corpus, but
   measured second so its effect is separable from hybrid.

## Known weaknesses of the current chunking

On Kubernetes documentation specifically:

- **Code blocks get sliced.** A 1000-character cut lands mid-YAML, retrieving
  half a manifest.
- **Chunks lose their heading.** A chunk about `spec.replicas` carries nothing
  identifying it as Deployments rather than StatefulSets.
- **The `. ` separator misfires** on `v1.2.3` and `metadata.name`.

These are expected to show up in the eval as recall failures, and are what
motivates the structure-aware change above.

## Debugging retrieval

Bisect before fixing. Is the correct chunk in the top 50?

- **No** — recall problem. Fix chunking; add lexical search.
- **Yes, but not top 5** — ranking problem. Add reranking.
- **Yes, top 5, bad answer** — not retrieval. It is the prompt or generation.

A reranker cannot fix a recall problem: it only reorders what retrieval found.
