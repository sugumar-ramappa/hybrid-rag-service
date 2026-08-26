# Testing and evaluation

Two independent things are tested, and conflating them is the most common mistake
in RAG projects:

| | Question it answers | Tool | Cost |
|---|---|---|---|
| **Correctness** | Does the code do what it says? | `pytest`, 55 tests | free |
| **Quality** | Does retrieval find the right chunk? | `scripts/evaluate.py`, 43 questions | ~43 cached embeddings |

A green test suite says nothing about whether search works. A recall@5 of 0.98
says nothing about whether the ingest handles a deleted document. **Both are
needed, and neither substitutes for the other.**

Companion documents: [`architecture-flow.md`](architecture-flow.md) ·
[`code-walkthrough.md`](code-walkthrough.md)

---

# Part 1 — Correctness: the test suite

```bash
export TEST_DATABASE_URL="postgresql://rag:rag@localhost:5432/ragdb_test"
pytest tests/ -v
```

**55 tests across 5 files.** No mocked database — the store tests run against a
real Postgres with pgvector, because every serious bug found in this project was
in the SQL or the type adaptation, which a mock would have faked correctly.

| File | Tests | Covers |
|---|---:|---|
| `test_document_loader.py` | 14 | recursive walk, hidden paths, `max_files` determinism, failure isolation |
| `test_text_splitter.py` | 8 | **chunk identity** — the data-corruption regression |
| `test_vector_store.py` | 18 | upsert, incremental diff, orphan deletion, **both searches** |
| `test_embedding_cache.py` | 6 | key composition, the `Vector` read-path bug |
| `test_answer_cache.py` | 9 | threshold behaviour, corpus invalidation |

## The safety guard, and why it exists

```python
TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", "")

@pytest.fixture
def store():
    if not TEST_DATABASE_URL:
        pytest.skip("TEST_DATABASE_URL not set")

    db_name = TEST_DATABASE_URL.rsplit("/", 1)[-1].split("?")[0]
    if "test" not in db_name.lower():
        pytest.skip(f"refusing to write to database '{db_name}'")
```

**Two independent conditions, both required.** A separate variable *and* the word
"test" in the database name.

This is here because it went wrong. The fixture originally read `DATABASE_URL`
and guarded only against non-local hosts — which is backwards. Running
`pytest tests/` truncated the **real** `chunks` table. 784 chunks, a full day of
free-tier quota, gone.

The lesson is worth stating in an interview: **a guard that checks the wrong
property is worse than no guard**, because it produces confidence. "Is it local?"
was never the question. "Is it the throwaway one?" was.

Tests **skip** rather than fail when the variable is unset, so `pytest` on a
laptop with no database still runs the 22 pure-logic tests.

## The tests that exist because something broke

### `test_chunk_ids_are_stable_when_another_document_is_added`

The regression test for the worst bug in the project.

```python
before = splitter.split_documents([doc_a, doc_b])
after  = splitter.split_documents([doc_a, doc_new, doc_b])

ids_before = {c.chunk_id for c in before if c.metadata["source"] == "b.md"}
ids_after  = {c.chunk_id for c in after  if c.metadata["source"] == "b.md"}
assert ids_before == ids_after
```

Under the old global counter, inserting a document renumbered every chunk after
it. `add_chunks` upserts on `chunk_id`, so the renumbered chunks **silently
overwrote** unrelated documents' rows. No exception, no log line — retrieval just
started returning content from the wrong pages.

**The pre-existing test asserted the buggy behaviour.** It was called
`test_chunk_indices_are_sequential` and it checked that indices ran 0, 1, 2, 3
across the whole corpus — which is exactly the property that caused the
corruption. It passed for as long as the bug existed.

That is the more interesting half of the story: the test suite was green, and the
data was being destroyed. A test that asserts an implementation detail will
defend a bug as loyally as it defends a feature.

Three tests replaced it, each asserting a property rather than a mechanism:
stability under insertion, uniqueness across documents, and
`test_identical_pdf_pages_do_not_collide`.

### `test_hybrid_keyword_arm_works_on_a_realistic_question`

The regression test for hybrid search silently doing nothing.

```python
# A full-length question, as the eval harness actually sends. Only SOME of
# its terms appear in the target chunk.
question = ("Which Pod quality classifications are blocked from accessing "
            "swap memory when LimitedSwap is active")

results = store.hybrid_search(vec(1.0, 0.0, 0.0), question, top_k=2)

assert any("LimitedSwap" in r.content for r in results), \
    "the keyword arm found nothing - check the tsquery operator"
```

The dense vector is deliberately pointed at the **wrong** chunk
(`vec(1.0, 0.0, 0.0)` matches the noise document), so the target can only surface
via the keyword arm. If the tsquery operator regresses to AND, this test is the
one that fails.

The original hybrid tests used **one-word queries**, where AND and OR are
identical, so they passed against a keyword arm that could never match anything
real. Hybrid returned byte-identical results to dense in production and the suite
was green.

The other hybrid tests cover the fusion logic itself:

- `test_hybrid_finds_what_dense_alone_misses` — the reason hybrid exists
- `test_hybrid_still_works_when_no_keyword_matches` — the `FULL OUTER JOIN` path
- `test_hybrid_ranks_agreement_above_single_list_hits` — the core RRF property

### `test_round_trip_returns_a_usable_list`

`register_vector` makes psycopg return a pgvector `Vector` object on read, not a
list — `TypeError: 'Vector' object is not iterable`. The cache had been shipped
with only its **write** path tested. Six tests now cover the read path and key
composition.

### `test_existing_chunk_ids_ignores_a_different_model`

Vectors from different models are not comparable. If the diff matched on
`chunk_id` alone, changing `EMBEDDING_MODEL` would skip re-embedding and leave
two incompatible vector spaces in one index — silently, with degraded retrieval
as the only symptom.

## CI

`.github/workflows/ci.yml` runs `ruff`, then `pytest` against a **pgvector
service container**, then validates the golden set itself:

```python
if len(verified) < 20:
    sys.exit(f"FAIL: only {len(verified)} verified questions - "
             "below 20 the metrics are too noisy to act on")
if len(styles) < 2:
    sys.exit("FAIL: both question styles are needed for the "
             "per-style breakdown to mean anything")
```

CI treats the **evaluation set as a build artifact**, not as data. It is JSON
edited by hand during review, so it is exactly the kind of file that gets a
trailing comma at 11pm — and a pull request that silently dropped verified
questions would otherwise only be discovered when the metrics moved for no
apparent reason.

---

# Part 2 — Quality: the evaluation harness

This is the part almost nobody builds, and the part every interview asks about.

## The golden set

**43 hand-verified question → chunk pairs** in `eval/golden_set.json`. Each entry
names a question, the exact chunk that should answer it, and a style label.

```json
{
  "id": "q003",
  "style": "exact_term",
  "question": "What is the current beta status for structured logging features...",
  "source": "concepts/cluster-administration/system-logs.md",
  "chunk_index": 4,
  "verified": true
}
```

### `scripts/generate_golden_set.py` — draft candidates

```bash
python -m scripts.generate_golden_set          # 60 candidates
```

Every candidate is written with `"verified": false`, and **the harness ignores
unverified entries** — so skipping the review produces an empty result rather
than a flattering one.

**Sampling is by document, not by chunk.** The corpus is skewed: one page
produces 59 of 784 chunks while several produce one. Uniform chunk sampling would
have drawn ~8% of questions from a single document, and the headline number would
mostly have measured retrieval quality on that one page.

**Two styles are generated deliberately:**

| Style | Phrasing | Weak for |
|---|---|---|
| `exact_term` | names identifiers a reader would type — `PersistentVolumeClaim`, `reclaimPolicy` | dense search |
| `paraphrase` | describes the need in natural language | keyword search |

Labelling each question is what makes it possible to report recall **per style**,
which shows *where* a change helps rather than just that a number moved. Without
the split, hybrid search looks like "+3 points" instead of "identifier queries
went from 0.95 to 1.00 and paraphrases were untouched."

Real examples from the set:

```
exact_term  q003  "What is the current beta status for structured logging
                   features starting in version v1.23?"
            q013  "How does the aggregation layer differ from the functionality
                   provided by Custom Resource Definitions?"

paraphrase  q010  "What should I do if I need to update a Node's state
                   significantly?"
            q018  "How do I update app configuration without needing to rebuild
                   my container image?"
```

### What actually separates the two styles

The intuitive answer — *"paraphrased questions share fewer words with their
answer"* — is **wrong on this corpus**, and measuring it is what showed that:

| Style | Word overlap with its answer chunk | Questions with ≥1 rare term | Rare terms per question |
|---|---:|---:|---:|
| `exact_term` | 45% | 20 / 21 | **2.6** |
| `paraphrase` | 52% | 18 / 22 | **1.5** |

*Rare* = a word appearing in ≤1% of the 784 chunks.

Paraphrased questions share slightly **more** vocabulary, because natural
phrasing still reuses common words like *update*, *configuration*, *node*. What
they lack is **distinctive** vocabulary.

That matters because `ts_rank` weights matches by term rarity. Matching on
*"update"* — which appears in hundreds of chunks — tells you almost nothing;
matching on `FlowSchema` narrows 784 chunks to two. So the keyword arm's value is
not driven by *how many* words are shared, but by **how rare the shared words
are**.

This is the sharper version of the claim, and the one worth making in an
interview: hybrid search helps when queries carry **rare terms**, not merely when
they carry shared terms.

### `scripts/check_golden_set.py` — catch leakage mechanically

```bash
python -m scripts.check_golden_set
```

Measures vocabulary overlap between each question and its chunk, and flags the
longest verbatim shared run. **A question that reuses its chunk's wording tests
string matching, not retrieval** — both search modes find it trivially, so it
inflates recall while proving nothing.

Worth catching by script rather than by eye, because it is a property of how the
question was generated, not a judgement call. No API calls.

### `scripts/review_golden_set.py` — the hand-verification

```bash
python -m scripts.review_golden_set
python -m scripts.review_golden_set --style paraphrase
```

Shows each question beside the chunk it should retrieve, plus any warnings
already flagged, and asks whether it is a fair test. Saves after every decision,
so quitting loses nothing.

**17 of 60 candidates were rejected** — a 30% rejection rate:

- link-list chunks ("What's next: ...") with no explanatory content
- questions another chunk answers better
- questions that restate their own source
- Hugo shortcode and HTML markup dumps

The generator sees **one chunk in isolation** and cannot know whether twenty
other chunks answer its question equally well. That judgement is precisely what
hand-verification supplies, and precisely what an automated pipeline cannot.

**The 30% rejection rate is the point, not a defect.** An automated eval would
have measured happily against navigation sections and HTML tables and produced a
confident, meaningless number.

## `scripts/evaluate.py` — the measurement

```bash
python -m scripts.evaluate --mode dense  --label dense-v2
python -m scripts.evaluate --mode hybrid --label hybrid-v2
python -m scripts.evaluate --compare
```

For each verified question: embed it, search, find where the expected chunk
ranked.

```python
vector = embeddings.embed_query(q["question"])      # cached after the first run

if mode == "hybrid":
    results = store.hybrid_search(vector, q["question"], top_k=MAX_K, ...)
else:
    results = store.search(query_embedding=vector, top_k=MAX_K)

rank = None
for position, result in enumerate(results, start=1):
    if (result.metadata.get("source") == q["source"]
            and result.metadata.get("chunk_index") == q["chunk_index"]):
        rank = position
        break
```

**Matching is on `(source, chunk_index)`, not `chunk_id`.** That is deliberate
and it is what let the golden set survive a full re-ingest: chunk IDs are derived
from content, so re-chunking regenerates them, but a chunk's *position within its
document* is stable. The golden set was built before a re-ingest and still valid
after it.

**Same cached embedding for both modes.** Switching mode costs zero API calls,
which is what makes the two runs directly comparable — the only variable that
changed is the retrieval strategy.

### The metrics

```python
def recall_at(outcomes, k):
    return sum(1 for o in outcomes if o.rank and o.rank <= k) / len(outcomes)

def mrr(outcomes):
    return sum(1 / o.rank if o.rank else 0 for o in outcomes) / len(outcomes)
```

**recall@k** — fraction of questions whose expected chunk appeared in the top k.
The headline. **recall@5 matters most because 5 chunks is what the prompt
carries** (`TOP_K_RESULTS=5`); anything at rank 6 is never seen by the model.

**MRR** — mean reciprocal rank: `1/rank`, averaged, 0 for a miss. Rewards ranking
the right chunk **first** rather than fifth. recall@5 cannot tell those apart, and
generation quality depends on it, because models attend more to what comes first.

**recall@10 is fetched but not optimised for.** It exists to tell you *how badly*
a miss missed. Rank 6 means ranking needs a nudge; absent from the top 10 means
retrieval never found it at all. Different problems, different fixes.

### What it prints

```
  ============================================================================
    hybrid-v2   ·   43 questions   ·   784 chunks
  ============================================================================

                 recall@1  recall@5  recall@10     MRR
    overall          0.74      0.98       0.98    0.83
    exact_term       0.86      1.00       1.00    0.91
    paraphrase       0.64      0.95       0.95    0.75

    1 questions where the expected chunk was not in the top 5:
      q020  [paraphrase ] not in top 10
            What configuration fields are mandatory when defining a new
            wanted concepts/overview/working-with-objects/_index.md#6, got ...
```

That is the **real and only miss** in the current hybrid run — one question out
of 43. Worth knowing why it misses: *"what configuration fields are mandatory
when defining a new Kubernetes resource?"* is answered acceptably by several
chunks about object spec and metadata, so the single chunk the golden set names
is a defensible answer rather than the only one. It is close to the limit of what
a single-chunk ground truth can express.

**Every miss is printed with what it got instead**, followed by:

> Read these before blaming retrieval. A question whose expected chunk ranks
> badly is sometimes a bad question — one that another chunk answers better —
> which hand-verification cannot fully rule out.

That line is in the output because it was true more than once.

## `--compare` — the audit trail

Every labelled run writes `eval/results/<label>.json` with the full per-question
outcome, not just the summary. `--compare` prints every saved run side by side.

This is what makes a claim defensible. "Hybrid improved recall" is an assertion;
a stored run showing which of 43 questions changed rank, and in which direction,
is evidence.

---

# Part 3 — The method, and what it found

## One change per measurement

Hybrid search went in as **one change**, with nothing else touched between the
baseline and the re-measurement. Two improvements between measurements give one
number and no attribution — and "I improved retrieval" without knowing which
change earned it is exactly the claim that falls apart under questioning.

## The result

| Mode | recall@1 | recall@5 | recall@10 | MRR |
|---|---:|---:|---:|---:|
| `dense-v2` | 0.67 | 0.95 | 0.98 | 0.80 |
| **`hybrid-v2`** | **0.74** | **0.98** | 0.98 | **0.83** |

| Style | recall@1 | recall@5 |
|---|---|---|
| `exact_term` | 0.71 → **0.86** | 0.95 → **1.00** |
| `paraphrase` | 0.64 → 0.64 | 0.95 → 0.95 |

## The finding worth leading with

Not the headline number — **where the gain landed.**

Hybrid took overall recall@5 from 0.95 to 0.98, which on its own is a small
number that could be noise. The per-style breakdown shows it is not:

| Style | recall@1 | recall@5 | What changed |
|---|---|---|---|
| `exact_term` | 0.71 → **0.86** | 0.95 → **1.00** | every question now retrieved |
| `paraphrase` | 0.64 → 0.64 | 0.95 → 0.95 | **nothing** |

**The entire gain is on identifier questions, and paraphrased questions are
untouched — not slightly better, not slightly worse. Unchanged.**

That is exactly what the technique predicts. The keyword arm can only contribute
where the question carries a rare, distinctive term. A question naming
`UnknownVersionInteroperabilityProxy` gives it a rare term to match on, and it
ranks the right chunk first. A question phrased as *"how does the cluster decide
where to run things"* gives it nothing, so RRF gets a vote from only one arm and
the dense ranking survives intact.

Two things follow, and both are worth saying out loud in an interview:

1. **A retrieval score quoted without describing its question distribution is
   close to meaningless.** "recall@5 of 0.98" is only interpretable alongside
   what the questions look like — this same system scores differently on
   differently-phrased sets, because that is what hybrid retrieval *is*.
2. **Labelling the questions by style is what made the result legible.** Without
   the split, this is "+3 points, probably noise." With it, the mechanism is
   visible in the numbers, and the result matches the theory for a reason you
   can point at rather than assert.

## What is deliberately not tested

**LLM-as-judge.** Scoring generated answers with another model measures the
judge as much as the system, costs an API call per question, and is not
reproducible run to run. Retrieval metrics are deterministic, free after the
first run, and localise the failure: if the right chunk was never retrieved, no
prompt engineering will fix the answer.

**Answer text quality.** The prompt constrains the model to answer from context
only. If retrieval puts the right chunk in front of it, the answer follows —
so retrieval is the variable worth measuring.

**Chunking strategies beyond the baseline.** Structure-aware chunking, semantic
chunking, and parent-child retrieval are all listed in
[`retrieval-strategies.md`](../retrieval-strategies.md) with expected trade-offs,
and none were measured. Each requires a **full re-embed** of 784 chunks against a
1,000-per-day free-tier quota, and the golden set is anchored to
`(source, chunk_index)` — re-chunking would invalidate all 43 questions. Saying
"unmeasured, and here is what it would cost to measure" is a stronger answer than
guessing.

---

# The scripts, in order of use

| Script | What it does | API cost |
|---|---|---|
| `fetch_corpus.sh` | Sparse-clones the Kubernetes docs. 396 files in ~10s | none |
| `check_limits.py` | Probes what your key actually allows — finds the largest working batch. Sets `EMBED_BATCH_SIZE` from evidence, not guesswork | ~12 calls |
| `inspect_chunking.py` | Shows exactly how one document chunks: block sizes, packing, real overlap, resulting IDs. No DB, no API | none |
| `ingest.py --dry-run` | Answers "what would this cost?" — full load and split, then stops | none |
| `ingest.py` | The real ingest. `--dir` and `--max-files` scope one run | 1 per 50 chunks |
| `generate_golden_set.py` | 60 candidates, all `verified: false` | ~60 calls |
| `check_golden_set.py` | Flags vocabulary leakage before you spend time reviewing | none |
| `review_golden_set.py` | Hand-verification, one question at a time | none |
| `evaluate.py` | The measurement. `--mode`, `--keyword-weight`, `--label`, `--compare` | 43, then cached |
| `query.py` | Interactive CLI — the thing you demo | 1 embed + 1 generate |

**Six of ten cost nothing**, which is not an accident. On a 1,000-per-day quota,
anything that can be checked locally has to be checked locally, or you spend the
day's budget discovering that `CHUNK_SIZE` was wrong.

`--dir` on `ingest.py` is a flag rather than `export DOCUMENTS_DIR=...` for the
same reason: an exported variable lingers for the rest of the shell session, so
setting it once for a test means every later ingest silently reads the wrong
directory. A flag applies to one invocation and shows up in shell history.

---

# Reproducing the whole thing

```bash
# 1. Postgres with pgvector
docker run -d --name ragdb -p 5432:5432 \
  -e POSTGRES_USER=rag -e POSTGRES_PASSWORD=rag -e POSTGRES_DB=ragdb \
  pgvector/pgvector:pg16
docker exec ragdb psql -U rag -d ragdb -c "CREATE DATABASE ragdb_test;"

# 2. Correctness — 55 tests, free, ~20 seconds
export TEST_DATABASE_URL="postgresql://rag:rag@localhost:5432/ragdb_test"
pytest tests/ -v

# 3. Corpus — 396 files, ~10 seconds
./scripts/fetch_corpus.sh

# 4. Cost check before spending quota
python -m scripts.ingest --dry-run

# 5. Ingest — 784 chunks, ~25 min at free-tier rate limits
python -m scripts.ingest

# 6. Quality — the two runs the README reports
python -m scripts.evaluate --mode dense  --label dense-v2
python -m scripts.evaluate --mode hybrid --label hybrid-v2
python -m scripts.evaluate --compare
```

Step 2 needs no API key. Steps 4–6 need one, and step 5 is the only one that
costs meaningful quota.
