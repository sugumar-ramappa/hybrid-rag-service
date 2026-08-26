"""
Measure retrieval quality against the hand-verified golden set.

    python -m scripts.evaluate                        # run and print
    python -m scripts.evaluate --label dense-baseline # name it, save results
    python -m scripts.evaluate --compare              # show all saved runs

For every verified question: embed it, search, and find where the expected
chunk ranked. No generation calls, no LLM judging - retrieval only, which is
what makes this cheap enough to run after every change.

METRICS
  recall@k  fraction of questions whose expected chunk appeared in the top k.
            The headline number. recall@5 matters most because 5 chunks is what
            a typical RAG prompt can carry.

  MRR       mean reciprocal rank: 1/rank of the expected chunk, averaged, with
            0 for a miss. Rewards ranking the right chunk FIRST rather than
            fifth - recall@5 cannot tell those apart, and generation quality
            depends on it, because models attend more to what comes first.
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from src.config import get_config
from src.rag.embedding_cache import EmbeddingCache
from src.rag.embeddings import EmbeddingService
from src.rag.reranker import CrossEncoderReranker, fuse, movement
from src.rag.vector_store import VectorStore

RESULTS_DIR = Path("eval/results")
MAX_K = 10

# How many candidates the reranker is given before it picks the top MAX_K.
#
# 50 rather than 10, deliberately. Reranking the same 10 the metric reads could
# only reorder within a window that already contains the answer - it could never
# rescue a chunk sitting at rank 23. Widening the net first is the entire point
# of retrieve-then-rerank: cheap search casts wide, the expensive model sorts.
#
# The failure analysis said 5 of 8 misses were at rank 6-10 and 3 were outside
# the top 10 entirely. A pool of 50 gives the reranker a chance at both groups,
# so the result distinguishes "ranking was the problem" from "retrieval was".
RERANK_POOL = 50


@dataclass
class Outcome:
    question_id: str
    style: str
    question: str
    expected: str
    rank: int | None  # 1-based position of the expected chunk; None if absent
    top_source: str


def recall_at(outcomes: list[Outcome], k: int) -> float:
    if not outcomes:
        return 0.0
    return sum(1 for o in outcomes if o.rank is not None and o.rank <= k) / len(outcomes)


def mrr(outcomes: list[Outcome]) -> float:
    if not outcomes:
        return 0.0
    return sum(1 / o.rank for o in outcomes if o.rank is not None) / len(outcomes)


def evaluate(questions: list[dict], embeddings: EmbeddingService,
             store: VectorStore, mode: str = "dense",
             keyword_weight: float = 1.0,
             reranker: CrossEncoderReranker | None = None,
             rerank_mode: str = "fuse") -> list[Outcome]:
    outcomes = []
    # Mutable cells so the loop can accumulate without a class. How much the
    # reranker moved things is reported alongside the score: a gain with almost
    # no movement came from one or two questions, which on 43 is noise.
    promoted_total = [0]
    biggest_jump = [0]

    for i, q in enumerate(questions, start=1):
        # Same cached embedding either way, so switching modes costs nothing -
        # which is what makes the two runs directly comparable.
        vector = embeddings.embed_query(q["question"])

        # A wider pool when reranking, because the reranker can only reorder
        # what retrieval handed it.
        pool = RERANK_POOL if reranker else MAX_K

        if mode == "hybrid":
            results = store.hybrid_search(vector, q["question"], top_k=pool,
                                          keyword_weight=keyword_weight)
        else:
            results = store.search(query_embedding=vector, top_k=pool)

        if reranker and results:
            scored = reranker.rerank(q["question"], [r.content for r in results])
            # Fuse by default. Replacing outright discards the lexical signal
            # that identifier questions depend on - measured, not assumed.
            scored = (fuse(scored, top_k=MAX_K) if rerank_mode == "fuse"
                      else scored[:MAX_K])
            results = [results[c.index] for c in scored]
            moved = movement(scored)
            promoted_total[0] += moved["promoted"]
            biggest_jump[0] = max(biggest_jump[0], moved["max_promotion"])

        # search() returns content and metadata, not chunk_id, so match on the
        # (source, chunk_index) pair the metadata carries.
        rank = None
        for position, result in enumerate(results, start=1):
            if (result.metadata.get("source") == q["source"]
                    and result.metadata.get("chunk_index") == q["chunk_index"]):
                rank = position
                break

        outcomes.append(Outcome(
            question_id=q["id"],
            style=q["style"],
            question=q["question"],
            expected=f"{q['source']}#{q['chunk_index']}",
            rank=rank,
            top_source=results[0].metadata.get("source", "?") if results else "?",
        ))

        marker = f"#{rank}" if rank else "MISS"
        print(f"  [{i:>2}/{len(questions)}] {marker:<5} {q['id']} {q['question'][:60]}")

    if reranker:
        print(f"\n  reranker moved {promoted_total[0]} chunk(s) up, "
              f"biggest promotion {biggest_jump[0]} places")

    return outcomes


def report(outcomes: list[Outcome], label: str, corpus_size: int) -> dict:
    print(f"\n{'=' * 78}")
    print(f"  {label}   ·   {len(outcomes)} questions   ·   {corpus_size} chunks")
    print("=" * 78)

    print(f"\n  {'':<12} {'recall@1':>9} {'recall@5':>9} {'recall@10':>10} {'MRR':>7}")
    rows = [("overall", outcomes)]
    for style in sorted({o.style for o in outcomes}):
        rows.append((style, [o for o in outcomes if o.style == style]))

    summary = {}
    for name, subset in rows:
        stats = {
            "n": len(subset),
            "recall@1": recall_at(subset, 1),
            "recall@5": recall_at(subset, 5),
            "recall@10": recall_at(subset, 10),
            "mrr": mrr(subset),
        }
        summary[name] = stats
        print(f"  {name:<12} {stats['recall@1']:>9.2f} {stats['recall@5']:>9.2f} "
              f"{stats['recall@10']:>10.2f} {stats['mrr']:>7.2f}")

    missed = [o for o in outcomes if o.rank is None or o.rank > 5]
    if missed:
        print(f"\n  {len(missed)} questions where the expected chunk was not in the top 5:\n")
        for o in missed:
            where = f"rank {o.rank}" if o.rank else "not in top 10"
            print(f"    {o.question_id}  [{o.style:<10}] {where}")
            print(f"          {o.question[:66]}")
            print(f"          wanted {o.expected}, got {o.top_source}")

        print("\n  Read these before blaming retrieval. A question whose expected chunk")
        print("  ranks badly is sometimes a bad question - one that another chunk")
        print("  answers better - which hand-verification cannot fully rule out.")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    parser.add_argument("--golden-set", default="eval/golden_set.json")
    parser.add_argument("--rerank", action="store_true",
                        help="retrieve %d candidates and reorder them with a "
                             "cross-encoder before scoring. Targets the 5 of 8 "
                             "misses that were at rank 6-10." % RERANK_POOL)
    parser.add_argument("--rerank-mode", choices=["fuse", "replace"], default="fuse",
                        help="fuse: combine retrieval rank with reranker rank "
                             "(default). replace: use the reranker's order alone")
    parser.add_argument("--rerank-model", default=None,
                        help="override the cross-encoder")
    parser.add_argument("--mode", choices=["dense", "hybrid"], default="dense",
                        help="dense = vector only; hybrid = vector + keyword fused with RRF")
    parser.add_argument("--keyword-weight", type=float, default=1.0,
                        help="hybrid only: how much the keyword arm counts in fusion. "
                             "1.0 = equal vote with dense; lower reduces its influence")
    parser.add_argument("--label", default=None,
                        help="name for this run (defaults to the mode)")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--compare", action="store_true",
                        help="print every saved run side by side and exit")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    if args.compare:
        runs = sorted(RESULTS_DIR.glob("*.json"))
        if not runs:
            print("No saved runs yet.")
            return
        # Group by question count. Runs measured against different golden sets
        # are NOT comparable - the question distribution is the instrument, and
        # this project measured a 14-point swing from wording alone.
        loaded = [json.loads(p.read_text()) for p in runs]
        by_set: dict[int, list] = {}
        for d in loaded:
            by_set.setdefault(d["summary"]["overall"]["n"], []).append(d)

        for n in sorted(by_set):
            group = sorted(by_set[n], key=lambda d: -d["summary"]["overall"]["recall@5"])
            gset = group[0].get("golden_set", "(not recorded)")
            print(f"\n  question set: {gset}  ({n} questions)")
            print(f"  {'run':<24} {'recall@1':>9} {'recall@5':>9} {'recall@10':>10} {'MRR':>7}")
            print("  " + "-" * 62)
            for d in group:
                o = d["summary"]["overall"]
                print(f"  {d['label']:<24} {o['recall@1']:>9.2f} {o['recall@5']:>9.2f} "
                      f"{o['recall@10']:>10.2f} {o['mrr']:>7.2f}")

        if len(by_set) > 1:
            print("\n  Runs under different question sets are not comparable -")
            print("  the question distribution is part of the instrument.")
        return

    config = get_config()

    data = json.loads(Path(args.golden_set).read_text())
    # Two admissible provenances, kept distinguishable on purpose.
    #
    #   verified            a person read the question and confirmed the chunk
    #   structural          derived from a heading and the prose beneath it
    #
    # The structural set is 7x larger and thinner: a heading is a topic label,
    # not how anyone asks a question. It is large enough that a two-point
    # difference is not one question, which the 43-question set cannot say. Both
    # are useful; conflating them is not, so the header prints which is in use.
    questions = [q for q in data["questions"]
                 if q.get("verified") or q.get("provenance") == "structural"]
    provenance = ("hand-verified" if all(q.get("verified") for q in questions)
                  else "structurally derived" if questions else "none")
    if not questions:
        print("No verified questions. Review the golden set first.")
        return

    cache = EmbeddingCache(config.database.url)
    embeddings = EmbeddingService(
        api_key=config.gemini.api_key,
        model_name=config.gemini.embedding_model,
        dimensions=config.gemini.embedding_dim,
        batch_size=config.gemini.embed_batch_size,
        cache=cache,
    )
    store = VectorStore(
        database_url=config.database.url,
        embedding_dim=config.gemini.embedding_dim,
        embedding_model=config.gemini.embedding_model,
    )

    label = args.label or (
        args.mode if args.mode == "dense" or args.keyword_weight == 1.0
        else f"{args.mode}-w{args.keyword_weight:g}"
    )
    how = args.mode + (f" + rerank/{args.rerank_mode}({RERANK_POOL}->{MAX_K})" if args.rerank else "")
    print(f"\nEvaluating {len(questions)} {provenance} questions in {how} mode "
          f"({len(data['questions']) - len(questions)} rejected)\n")

    reranker = None
    if args.rerank:
        kwargs = {"model_name": args.rerank_model} if args.rerank_model else {}
        reranker = CrossEncoderReranker(**kwargs)

    outcomes = evaluate(questions, embeddings, store, mode=args.mode,
                        keyword_weight=args.keyword_weight, reranker=reranker,
                        rerank_mode=args.rerank_mode)
    summary = report(outcomes, label, store.get_document_count())

    stats = cache.stats
    print(f"\n  query cache: {stats['hits']} hits, {stats['misses']} misses "
          f"({stats['hit_rate']:.0%} hit rate), {cache.size()} entries stored")
    if stats["misses"]:
        print(f"  {stats['misses']} embedding requests used. The next run costs none.")

    if not args.no_save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULTS_DIR / f"{label}.json"
        out.write_text(json.dumps({
            "label": label,
            "golden_set": args.golden_set,
            "mode": args.mode,
            "keyword_weight": args.keyword_weight,
            "corpus_chunks": store.get_document_count(),
            "summary": summary,
            "outcomes": [
                {"id": o.question_id, "style": o.style, "rank": o.rank,
                 "expected": o.expected, "question": o.question}
                for o in outcomes
            ],
        }, indent=2) + "\n")
        print(f"\n  saved to {out}   ·   compare with: python -m scripts.evaluate --compare")


if __name__ == "__main__":
    main()
