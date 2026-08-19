# Walkthrough: one document, end to end

A guided run over a single document — ingest it, look at the rows in Postgres,
search it, then re-ingest it unchanged and changed to see exactly what the
incremental logic does.

Everything here uses `walkthrough/storage-guide.md`, which is about 4 KB and
produces four or five chunks. Small enough to read every row by hand.

> **Using a GUI client?** Run the queries from `docs/queries.sql` instead of
> copying them from here. The versions below are wrapped for the terminal
> (`psqlrag -c "..."`), so pasting one into DBeaver brings a stray `"` with it.
> `queries.sql` is plain SQL, same queries, no wrapping.

---

## Setup

```bash
cd ~/"Learning Project/hybrid-rag-service"
source .venv/bin/activate
docker ps | grep ragdb            # Postgres must be running
```

One shell shortcut, so the SQL commands below stay short:

```bash
alias psqlrag='docker exec -i ragdb psql -U postgres -d ragdb'
```

Every ingest below passes `--dir walkthrough`, so it reads the one test document
rather than your real corpus. Use the flag rather than `export DOCUMENTS_DIR=...`:
an exported variable stays set for the rest of the shell session, so it is easy
to run a later ingest against the wrong directory without noticing. A flag
applies to one command and is visible in shell history.

Worth knowing before you start:

```bash
python -m scripts.ingest --dir walkthrough --dry-run
```

reports how many chunks would be embedded, and in how many requests, without
calling the API or writing anything.

Start from an empty table:

```bash
psqlrag -c "TRUNCATE chunks;"
psqlrag -c "SELECT count(*) FROM chunks;"     # expect 0
```

---

## Step 1 — first ingest

```bash
python -m scripts.ingest --dir walkthrough
```

**Expect:** `4 chunks from 1 documents: 4 new, 0 already embedded`, then
`✓ 4 chunks in N seconds — 4 embedded, 0 reused, 0 removed`.

The exact chunk count depends on `CHUNK_SIZE`; four or five is normal.

---

## Step 2 — look at what was stored

**The rows:**

```bash
psqlrag -c "
SELECT chunk_index,
       chunk_id,
       source,
       length(content) AS chars,
       left(content, 45) AS preview
FROM chunks
ORDER BY chunk_index;"
```

Note `source` is `storage-guide.md` — the path relative to the corpus root, not
an absolute path.

**The vectors are real, and normalized:**

```bash
psqlrag -c "
SELECT chunk_index,
       vector_dims(embedding) AS dims,
       round(sqrt(sum(v*v))::numeric, 4) AS magnitude
FROM chunks, unnest(embedding::real[]) AS v
GROUP BY chunk_index, embedding
ORDER BY chunk_index;"
```

`dims` should be **768**, and `magnitude` should be **1.0000** — that is
`_normalize()` in `embeddings.py` doing its job after Matryoshka truncation.

**The keyword column Postgres maintains for you:**

```bash
psqlrag -c "
SELECT chunk_index, left(content_tsv::text, 70) AS tsvector_preview
FROM chunks ORDER BY chunk_index;"
```

You never wrote this. `content_tsv` is `GENERATED ALWAYS AS`, so Postgres derives
it from `content` and keeps it in sync. Nothing queries it yet — hybrid search at
step 6 of the plan is what turns it on.

**Which model produced these:**

```bash
psqlrag -c "SELECT DISTINCT embedding_model, embedding_dim FROM chunks;"
```

Recorded per row so a future model change can be migrated blue/green.

**The indexes:**

```bash
psqlrag -c "\d chunks"
```

`chunks_embedding_idx` is HNSW for vector search, `chunks_content_tsv_idx` is GIN
for keyword search, `chunks_source_idx` supports per-document deletes.

---

## Step 3 — search it

```bash
python -m scripts.query "how does a pod ask for storage"
```

Note the question shares almost no words with the answer — no "PersistentVolumeClaim",
no "request for storage by a user". Dense retrieval finds it anyway, because the
embedding captures meaning rather than tokens.

Now try the opposite kind of question:

```bash
python -m scripts.query "what is reclaimPolicy"
```

This one hinges on an exact identifier. Watch whether the right chunk comes back
first — this is precisely where dense-only retrieval is weakest, and precisely
what hybrid search fixes later. Worth noting the result now so you can compare.

---

## Step 4 — re-ingest, nothing changed

```bash
python -m scripts.ingest --dir walkthrough
```

**Expect:** `4 chunks from 1 documents: 0 new, 4 already embedded`, and
`0 embedded, 4 reused, 0 removed`, plus *"Nothing changed since the last run"*.

**Zero API calls.** Confirm nothing moved:

```bash
psqlrag -c "SELECT count(*), min(created_at), max(created_at) FROM chunks;"
```

Same count, and `created_at` unchanged — these are the original rows, not
rewritten ones.

---

## Step 5 — change the END of the document

Edit `walkthrough/storage-guide.md` and change one sentence in the **Volume
Lifecycle** section — the last one. For example, replace "Retain keeps the volume
and its data for manual cleanup" with "Retain preserves the volume and its data
for an administrator to clean up manually".

Record the current IDs first so you can compare:

```bash
psqlrag -c "SELECT chunk_index, chunk_id FROM chunks ORDER BY chunk_index;"
```

Then:

```bash
python -m scripts.ingest --dir walkthrough
```

**Expect:** `1 new, 3 already embedded` and `1 embedded, 3 reused, 1 removed`.

One chunk re-embedded, and the superseded version deleted. Confirm the count did
not grow:

```bash
psqlrag -c "SELECT count(*) FROM chunks;"                       -- still 4
psqlrag -c "SELECT chunk_index, chunk_id FROM chunks ORDER BY chunk_index;"
```

Only the last chunk's ID changed. The others are byte-identical, so their hashes
are identical, so they were never touched.

---

## Step 6 — make an edit that crosses a chunk boundary

An edit only forces *extra* re-embedding when it changes **which blocks land in
which chunk**. Editing inside a block re-embeds that block's chunk and nothing
else - true wherever in the document you edit.

Find this line in the Persistent Volumes section:

```
configure dynamic provisioning so that storage is created on demand.
```

and append to it:

```
 Each volume also records a phase such as Available, Bound, Released, or Failed, which controllers watch.
```

That grows the first paragraph past `CHUNK_SIZE`, so it can no longer share a
chunk with the headings above it.

Check the cost before spending anything:

```bash
python -m scripts.ingest --dir walkthrough --dry-run
```

then:

```bash
python -m scripts.ingest --dir walkthrough
```

**Expect 5 chunks, and all 5 re-embedded** — `5 embedded, 0 reused, 4 removed`.
Sizes go from `[927, 871, 769, 692]` to `[49, 975, 871, 769, 692]`: chunk 0
becomes a 49-character chunk holding nothing but the two headings.

Three of those chunks have **byte-identical content** to before and still
re-embedded, because `chunk_id = sha256(source : chunk_index : content)` and
every index below the change shifted by one. Position is part of identity, so
renumbering alone is enough.

That is the real rule, and it is worth stating precisely:

> Re-embedding cost depends on whether an edit **shifts a block across a chunk
> boundary**, not on where in the document the edit sits.

In this document blocks are whole paragraphs and chunk boundaries align with
sections, so boundary crossings are rare. Real Kubernetes docs have sections far
longer than `CHUNK_SIZE`, which forces the splitter down to sentence-level
pieces - and once chunks are packed from many small pieces, an edit near the top
shifts sentences across every boundary below it. That is where the cascade
actually bites.

Structure-aware chunking contains it: split on headings and a section's
boundaries stop depending on how long the preceding sections are.

---

## Step 7 — prove the orphan sweep

Deliberately break it: write a chunk that the corpus no longer produces.

```bash
psqlrag -c "
INSERT INTO chunks (chunk_id, source, chunk_index, content, metadata,
                    embedding, embedding_model, embedding_dim)
SELECT 'fake_orphan_001', source, 99, 'stale content that no longer exists',
       '{}'::jsonb, embedding, embedding_model, embedding_dim
FROM chunks LIMIT 1;"

psqlrag -c "SELECT count(*) FROM chunks;"     -- one more than before
```

```bash
python -m scripts.ingest --dir walkthrough
```

**Expect** `1 removed`, and the count back to normal. `delete_orphans` removed a
chunk whose source was re-ingested but which the corpus no longer produces —
exactly what happens to superseded versions of edited text.

---

## Step 8 — scale up

```bash
./scripts/fetch_corpus.sh
```

Ingest with `--max-files 20`, then 50, then 150. Watch the counts:

```
MAX_FILES=20   →  140 new,   0 reused
MAX_FILES=50   →  210 new, 140 reused     ← only the difference is embedded
MAX_FILES=150  →  700 new, 350 reused
```

Climbing the ladder costs exactly the same total tokens as jumping straight to
150 — the cap only controls when you spend them.

---

## Cleanup

```bash
psqlrag -c "TRUNCATE chunks;"
git checkout walkthrough/storage-guide.md    # undo the test edits
```

---

# Observed results

Measured on `walkthrough/storage-guide.md` (3,196 characters) with
`CHUNK_SIZE=1000`, `CHUNK_OVERLAP=200`, `gemini-embedding-001` at 768
dimensions. Reproduce any of it with:

```bash
python -m scripts.inspect_chunking walkthrough/storage-guide.md
```

## Baseline chunking

| Chunk | Chars | Starts with |
|-------|-------|-------------|
| 0 | 927 | `# Kubernetes Storage Guide` |
| 1 | 871 | `## Persistent Volume Claims` |
| 2 | 769 | `## Storage Classes` |
| 3 | 692 | `## Volume Lifecycle` |

Four chunks, not the three that 3196 / 1000 suggests. Each section is under
`CHUNK_SIZE`, so the splitter separated on blank lines and left each section
whole. **The document's structure decided the chunking; `CHUNK_SIZE` was only a
ceiling.**

## Overlap: asked for 200, got 27

Chunk 0 ends with `## Persistent Volume Claims` and chunk 1 begins with it —
27 characters shared, not 200.

The splitter carries back **whole blocks**. Working backwards from the boundary,
the heading (27 chars) fits the budget; the block before it is an 847-character
paragraph, which does not. It cannot take part of a block, so it stops at 27.

Total duplication across all three boundaries: **63 characters**, against a
budget that would have allowed 600.

The side effect is a chunk ending in a heading with no content beneath it, which
mildly pollutes chunk 0's embedding with a topic it never discusses.

## Cost of an edit

| Edit | Chunks | Re-embedded |
|------|--------|-------------|
| One phrase, last section | 4 | **1** |
| One phrase, first paragraph | 4 | **1** |
| +79 characters, first paragraph | 4 | **1** |
| +120 characters, first paragraph | **5** | **5** |

The first three change one chunk regardless of position. Blocks are atomic, so
editing inside one changes only the chunk holding it.

The fourth is different. The paragraph grew past `CHUNK_SIZE`, so it could no
longer share a chunk with the headings above it — the splitter closed chunk 0
early, leaving a 49-character chunk of nothing but headings, and everything
below shifted down by one.

**Every chunk re-embedded, including three whose text was byte-identical.**
Because `chunk_id = sha256(source : chunk_index : content)`, renumbering alone
changes the ID. That is the cost of putting position in the identity — the
benefit being that two identical paragraphs in one document stay distinct.

> Re-embedding cost depends on whether an edit **shifts a block across a chunk
> boundary**, not on where in the document it sits.

## Incremental ingest

| Run | Result |
|-----|--------|
| First ingest | 4 new, 0 reused |
| Re-ingest, no change | **0 embedded**, 4 reused, `created_at` unchanged |
| After a one-phrase edit | 1 embedded, 3 reused, **1 removed** |

Re-running over an unchanged corpus costs zero API calls, and superseded chunks
are swept rather than accumulating.
