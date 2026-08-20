# Retrieval strategies

Every chunking and search strategy worth knowing, what each costs, which this
service uses, and — most importantly — **how the choice was made** and how we
would know it was wrong.

---

## What this service uses

| Stage | Choice | Status |
|-------|--------|--------|
| Chunking | Recursive character, 1000 chars / 200 overlap | baseline |
| Embedding | `gemini-embedding-001`, 768 dimensions, normalized | measured |
| Storage | Postgres + pgvector, HNSW + GIN indexes | measured |
| Search | **Hybrid** — vector + keyword fused with RRF | measured, and the default |
| Reranking | none | deferred |
| Query rewriting | none | not needed (single-turn) |

784 chunks, 56 Kubernetes documentation pages, measured against 43 hand-verified
questions.

---

## Part 1 — Chunking strategies

Chunking decides what can be retrieved at all. No search strategy recovers from a
chunk that split a YAML manifest in half or dropped the heading that gave it
meaning. It is the highest-leverage and most-neglected part of the pipeline.

### Fixed-size with overlap

**How** — cut every N characters, carrying the last M into the next chunk.

**Use when** — establishing a baseline, or the corpus has no reliable structure.
Always start here: you need something to measure improvements against.

**Fails on** — anything structured. Code blocks, tables and lists get sliced
mid-element; chunks lose the heading that disambiguated them.

**Typical** — 512–1024 tokens, 10–20% overlap. **Cost:** free.

### Recursive character splitting — *what this service uses*

**How** — fixed-size, but tries a hierarchy of separators in order: paragraph
break, line break, sentence, word, character. Drops to the next only when a piece
is still too large.

**Use when** — prose with clean paragraph structure. This is what most frameworks
default to and it is a reasonable default.

**Fails on** — sentence separators misfire on `v1.2.3`, `metadata.name`, file
extensions. Still structure-blind: a heading is just another line break.

**Cost:** free.

### Document-structure aware

**How** — split on the document's own boundaries (markdown headings, HTML
sections, PDF outline) and **prepend the heading path** to each chunk before
embedding, so `Workloads > Deployments > Scaling` travels with the text.

**Use when** — documentation, wikis, manuals, papers. Usually the biggest single
win available on structured corpora, and it costs nothing at query time.

**Fails on** — unstructured sources. Sections vary wildly in length, so you still
need a size cap with a fallback splitter underneath. **Measured on this corpus:
naive heading-splitting turned a 70-chunk document into 13 chunks averaging 4,000
characters — far too large.** The real implementation is headings *plus* a size
cap *plus* recursive fallback.

**Cost:** free.

### Parent–child (small-to-big)

**How** — embed small chunks for precise matching, return the larger parent
section to the model. Matching and reading are different jobs with different
optimal sizes; this stops pretending they are the same.

**Use when** — answers need surrounding context: contracts, code, papers, anything
where a fragment is technically relevant but practically useless.

**Fails on** — context budget. Parents are large, so fewer results fit. Needs a
parent/child link in the store and deduplication when several children share a
parent.

**Typical** — ~200-token children, ~2000-token parents. **Cost:** free at ingest,
more tokens per query.

### Sentence window

**How** — embed individual sentences; on retrieval, expand to the k sentences
either side. A continuous variant of parent–child.

**Use when** — flowing prose with no section structure: books, transcripts,
articles.

**Fails on** — technical content, where a lone sentence rarely carries enough
signal to embed usefully. **Cost:** free.

### Semantic chunking

**How** — embed sentence by sentence and cut where consecutive similarity drops
below a threshold. Topic shift decides the boundary, not character count.

**Use when** — unstructured documents covering several distinct topics with no
headings to signal the change.

**Fails on** — cost and consistency. Requires embedding **every sentence** —
5–10× the calls of the chunks themselves. Results are mixed: it often loses to
structure-aware chunking on structured corpora while costing far more. Threshold
tuning is fiddly and corpus-specific.

**Cost: high.** On this corpus it would be 4,000–8,000 embedding calls to build
the index once.

### Contextual retrieval

**How** — before embedding, have an LLM write one or two sentences situating the
chunk inside its document, and prepend that. Pronouns and bare references gain
their referent.

**Use when** — retrieval quality genuinely justifies the cost and the corpus is
stable enough that you are not re-paying constantly.

**Fails on** — cost and ingestion latency: one generation call per chunk. Prompt
caching over the source document makes it cheaper but never free.

**Cost: very high.**

### Code / AST-aware

**How** — parse to a syntax tree and split on real boundaries — function, class,
module — carrying imports and the signature into each chunk.

**Use when** — any code corpus. Character-based splitting on code is close to
useless: half a function retrieves as noise.

**Fails on** — needs a parser per language; very long functions still need a
fallback. **Cost:** free.

### Late chunking

**How** — run the whole document through a long-context embedding model, then pool
token embeddings into chunk vectors. Each chunk's vector is computed with
full-document context already baked in.

**Use when** — you have a long-context embedding model and documents whose meaning
depends heavily on earlier context.

**Fails on** — requires a model exposing token-level embeddings. Newer and less
battle-tested. **Cost:** moderate.

---

## Part 2 — Search strategies

Retrieval fails in two distinct ways needing different fixes. Dense search misses
exact terms; lexical search misses paraphrase. Almost every technique below
patches one of those two holes.

### Dense / vector search — *this service's baseline*

**How** — embed the query, find nearest neighbours by cosine distance over an ANN
index.

**Strong at** — paraphrase and intent. *"How do I keep data after a pod restarts"*
finds the storage docs despite sharing no words with them.

**Weak at** — exact identifiers, rare terms, error codes, acronyms. Embeddings
compress away precisely the tokens that make a technical query specific.

**Cost:** one embedding per query.

### Sparse / keyword search

**How** — BM25 or Postgres `tsvector`: rank by term overlap, weighted by how rare
each term is across the corpus.

**Strong at** — exact matches. `PersistentVolumeClaim` or `ERR_CONN_REFUSED` land
precisely, and rare terms are treated as the strong signal they are.

**Weak at** — vocabulary mismatch. A question sharing no words with the answer
scores zero, however obviously related.

**Cost:** free — no model, no API call.

### Hybrid with rank fusion — *this service uses this*

**How** — run both, then merge. **Reciprocal Rank Fusion** scores each document as
the sum of `1/(k + rank)` across result lists, combining by *position* so the two
systems' incompatible score scales never need normalising.

```
score(doc) = 1/(60 + dense_rank) + 1/(60 + keyword_rank)
```

**Use when** — nearly always. The highest value-per-effort change in retrieval,
needing no model and no training.

**Watch for** — weighted score fusion can beat RRF when tuned, but needs
per-corpus normalisation and retuning as data drifts. RRF's parameter-free
robustness is usually worth more.

**Cost:** free — one extra SQL CTE.

### Cross-encoder reranking

**How** — retrieve broadly (say 50 candidates), then score each by running query
and document *together* through a model, keeping the top 5. Bi-encoders embed the
two separately and can never see their interaction; a cross-encoder can.

**Use when** — precision matters more than latency, and the context window is
tight enough that the top 5 need to genuinely be the best 5.

**Watch for** — real latency: a model pass per candidate. Usually the
second-largest quality gain after hybrid, and the largest latency cost.

**Cost:** a model download or a paid API, plus latency.

### Query rewriting

**How** — an LLM rewrites the query before retrieval: resolving pronouns from chat
history, expanding acronyms, stripping conversational padding.

**Use when** — multi-turn chat, where *"what about the second one?"* is meaningless
as a standalone search. Effectively mandatory for conversational RAG.

**Watch for** — a rewrite that drifts from user intent fails invisibly. Log both
versions or you will never diagnose it.

**Cost:** one generation call per query.

### HyDE

**How** — have the LLM write a *hypothetical answer* to the question, then embed
that instead of the question. Answers resemble documents; questions do not.

**Use when** — short questions over verbose corpora, and domains where question
and answer phrasing diverge sharply.

**Watch for** — the model hallucinates the hypothetical answer, so it can pull
retrieval toward plausible-but-absent content. Weakest where the corpus is
unusual — exactly where you needed help.

**Cost:** one generation call per query.

### Multi-query and decomposition

**How** — generate several phrasings and union the results, or split a compound
question into sub-questions and retrieve for each.

**Use when** — multi-hop questions. *"How does X compare to Y"* needs facts about
both, and a single query retrieves neither well.

**Watch for** — N× the retrieval cost, and results need deduplication.

**Cost:** one generation call plus N searches.

### Metadata filtering

**How** — constrain the candidate set by structured attributes (tenant,
permission, date, document type) before or during the similarity search.

**Use when** — multi-tenant or permissioned corpora, where this is a *correctness*
requirement rather than a quality improvement.

**Watch for** — **pre-filtering can starve an ANN index of candidates and quietly
degrade recall**; post-filtering is safer for recall but wastes work and can
return fewer results than requested. Knowing this distinction is a strong signal
in an interview.

**Cost:** free.

### MMR (diversity)

**How** — Maximal Marginal Relevance re-scores candidates to trade relevance
against redundancy, so results are not five near-identical passages.

**Use when** — corpora with heavy duplication: versioned docs, boilerplate-laden
contracts, overlapping chunks.

**Watch for** — push diversity too hard and you evict the genuinely best result.

**Cost:** free.

### Late interaction (ColBERT-style)

**How** — store a vector per *token* rather than per chunk, scoring by matching
query tokens against document tokens. Much of a cross-encoder's precision at
closer to bi-encoder speed.

**Use when** — precision matters, reranking latency does not fit the budget, and
you can afford the index.

**Watch for** — storage grows by an order of magnitude; tooling support is thinner.

**Cost: high** (storage).

### Graph RAG

**How** — extract entities and relationships into a graph, then traverse it to
assemble context rather than relying on similarity alone.

**Use when** — questions genuinely require multi-hop reasoning over connected
entities, or corpus-wide synthesis no single chunk contains.

**Watch for** — expensive to build and maintain, and frequently reached for where
hybrid plus reranking would have sufficed. **Knowing when *not* to use it is the
better signal.**

**Cost: very high.**

---

## Part 3 — How the choice is actually made

This is the part interviews probe, and the honest answer has two halves.

### Half one: start from priors

Nobody starts from scratch. Published guidance narrows a dozen options to two or
three before you write any code:

| Corpus | Chunking | Search |
|--------|----------|--------|
| Technical docs, API references | Structure-aware + heading path | Hybrid + RRF |
| Legal, contracts, policy | Parent–child | Hybrid + reranking |
| Support knowledge base, tickets | Fixed-size + overlap | Hybrid + rerank + query rewriting |
| Source code | AST-aware, one function per chunk | Hybrid, heavily lexical |
| Research papers | Structure-aware + parent–child | Hybrid + reranking |
| Long-form narrative | Semantic or sentence-window | Dense + MMR |
| Multi-tenant enterprise | Any, plus tenant metadata | **Filter first**, then hybrid |
| Tables, metrics, records | — | Text-to-SQL, not RAG — embeddings cannot count |

That table is why this project never considered Graph RAG or late interaction.
Priors did most of the elimination.

### Half two: read the failures

Priors tell you where to start. **The failure list tells you what to fix next** —
and it is evidence rather than opinion.

Run the baseline, then look at *which* questions failed and *why*:

| What the failures look like | What it means | Fix |
|---|---|---|
| Exact-identifier questions miss; paraphrased ones pass | Embeddings smoothing away rare tokens | **Hybrid search** |
| Right document retrieved, wrong section | Chunks lost their heading context | **Structure-aware chunking** |
| Retrieved chunk is half the answer, rest is next door | Boundaries cutting through explanations | **Parent–child** |
| Retrieved chunk is a link list or table | Corpus contains non-prose | **Preprocessing** |
| Correct chunk at rank 6–10, never higher | Retrieval finds it, ranking buries it | **Reranking** |
| Nothing relevant anywhere in the top 50 | Wrong corpus, or the answer is not in it | Neither — fix the corpus |

You do not test all six. You run once, read the misses, and usually one pattern
dominates.

`scripts/evaluate.py` prints exactly this list for that reason.

### Diagnose before you treat

When retrieval is bad, find out which half is broken:

- Correct chunk **not in the top 50** → a **recall** problem. Fix chunking, add
  lexical search.
- **In the top 50 but not top 5** → a **ranking** problem. Add reranking.
- **In the top 5, answer still bad** → not retrieval at all. Prompt or generation.

A reranker cannot fix a recall problem — it only reorders what retrieval already
found.

---

## Part 4 — What we chose, and how we would know it was wrong

### The decisions

**Recursive character splitting** — chosen as the *baseline*, not as the best
option. Structure-aware is very likely better on this corpus, and it is
deliberately deferred: changing chunking before a baseline exists makes the
improvement unattributable.

**Dense + hybrid, measured separately** — hybrid was chosen from priors
(technical documentation, dense with exact identifiers), then measured to confirm
the prior applied *here*.

**No reranking** — real cost in latency and either a model download or a paid API.
Deferred until the failure list shows a ranking problem rather than a recall
problem.

**No semantic chunking** — 4,000–8,000 embedding calls to build the index once,
to *infer* boundaries that markdown headings already *declare*. Wrong tool for a
structured corpus.

**No query rewriting** — single-turn interface. It solves a problem this service
does not have.

### How we would know we were wrong

- **Hybrid does not beat dense** → the prior did not apply here. Likely because
  Postgres's English stemmer handles camelCase identifiers like `reclaimPolicy`
  poorly, weakening the keyword arm. Worth checking the `tsvector` output before
  concluding hybrid does not help.
- **Failures are mostly "right document, wrong section"** → chunking is the
  problem, and structure-aware jumps ahead of everything else.
- **Failures are mostly rank 6–10** → retrieval is fine, ranking is not, and
  reranking moves to the front of the queue.
- **Failures are mostly link-list or markup chunks** → the corpus needs
  preprocessing before any retrieval strategy will help. 7% of this corpus is Hugo
  build markup and 29 of 56 documents end with a link list, so this is a live
  possibility.

Each of those would change the plan. That is the point of measuring rather than
ranking by intuition.

---

## Part 5 — Measuring

| Metric | Measures | Use for |
|--------|----------|---------|
| **recall@k** | Was the correct chunk retrieved at all, within the top k | The headline number. Free — no generation calls |
| **MRR** | How high the first correct result ranked | Ranking quality. recall@5 cannot tell rank 1 from rank 5, and models attend more to what comes first |
| **nDCG@k** | Ranking quality with graded relevance | When results are partially relevant rather than right or wrong |
| Faithfulness | Whether the answer is grounded in retrieved context | Catching hallucination. Needs an LLM judge |
| Answer relevance | Whether the answer addresses the question | End-to-end quality. Needs an LLM judge |

Retrieval metrics need only a query embedding and a search, so they are
effectively free and fast to iterate on. Generation metrics need an LLM per
question per configuration. **Build the retrieval harness first** — it is where
the engineering leverage is and where the numbers are cheapest.

### On golden sets

30–50 hand-verified question-and-source pairs is enough to steer decisions. Have
an LLM propose candidates from sampled chunks, then verify every one yourself —
the verification is what makes it trustworthy.

Report confidence intervals rather than bare percentages. *"94.1% (95% CI
92.3–95.6, n=800)"* shows you understand that a small sample carries uncertainty.

**Sample documents, not chunks.** A skewed corpus — one page holding 59 of 784
chunks — means uniform chunk sampling concentrates questions on a single document,
and the number measures retrieval on that page rather than on the corpus.

---

## Related

- [`architecture.md`](architecture.md) — the pipeline stage by stage, and the
  technology trade-offs
- [`plan.md`](plan.md) — what is deferred and why
- [`walkthrough.md`](walkthrough.md) — measured chunking behaviour on a real
  document
