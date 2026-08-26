"""Reorder retrieved chunks with a cross-encoder.

WHY THIS IS THE NEXT CHANGE, AND NOT CHUNKING
The baseline said so. Of the 8 questions dense retrieval missed at k=5:

    5  had the correct chunk at rank 6-10   retrieval found it, ranking buried it
    3  were absent from the top 10 entirely

A reranker only reorders what retrieval already returned, so it can fix the five
and cannot touch the three. Chunking would address the three and is a re-embed of
the whole corpus - a full day of free-tier quota. Doing the cheaper change that
targets the majority case first is what the baseline was measured for.

WHAT A CROSS-ENCODER ACTUALLY IS
An embedding model encodes the query and the document *separately*, then compares
two vectors. It never sees them together, which is what makes it fast: every
chunk is embedded once, at ingest, and a query is one more embedding plus a
nearest-neighbour lookup.

A cross-encoder reads the query and one document *in the same forward pass* and
outputs a single relevance score. It can weigh how the question relates to the
passage rather than whether two summaries happen to point the same way - which is
exactly the judgement that puts the right chunk at rank 1 instead of rank 7.

The cost is that it cannot be precomputed. Scoring N candidates is N forward
passes at query time, so it is only affordable over a shortlist. Hence
retrieve-then-rerank: cast a wide cheap net, then spend the expensive model on
the handful that came back.

    retrieve 50 by vector/hybrid   ~30ms, precomputed index
    rerank to 5 by cross-encoder   ~50 forward passes, at query time
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# ~90 MB, trained on MS MARCO passage ranking - the standard baseline
# cross-encoder. Small enough to run on CPU in a test, good enough that any
# improvement it fails to produce is a fact about the task rather than about
# model capacity.
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _ensure_ca_bundle() -> None:
    """Trust locally installed roots as well as the public ones.

    Networks that inspect TLS re-sign HTTPS with their own certificate authority.
    Browsers accept it because it is in the system keychain; Python verifies
    against certifi, which by design carries only public authorities, so the model
    download fails with

        CERTIFICATE_VERIFY_FAILED: self-signed certificate in certificate chain

    which reads as a network fault and is not one.

    setdefault, never assignment - an explicitly configured bundle is already
    correct and must not be overridden.
    """
    bundle = Path.home() / ".certs" / "combined-ca.pem"
    if bundle.exists():
        for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
            os.environ.setdefault(var, str(bundle))


def pick_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass(frozen=True)
class Scored:
    """One candidate and what the cross-encoder thought of it."""

    index: int          # position in the list handed to rerank()
    score: float
    original_rank: int  # 1-based rank before reranking, for the movement report


class CrossEncoderReranker:
    """Scores (query, passage) pairs and reorders by that score.

    Loaded lazily. Constructing this must not pull 90 MB into memory, because
    the evaluation harness builds one per run and the dense-only mode never
    calls it.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None,
                 batch_size: int = 16):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None

    def _load(self):
        if self._model is None:
            _ensure_ca_bundle()
            from sentence_transformers import CrossEncoder

            device = self.device or pick_device()
            log.info("loading reranker %s on %s", self.model_name, device)
            self._model = CrossEncoder(self.model_name, device=device,
                                       max_length=512)
        return self._model

    def rerank(self, query: str, passages: list[str],
               top_k: int | None = None) -> list[Scored]:
        """Return candidates ordered by relevance, best first.

        `top_k` truncates the *output*, not the scoring: every candidate is
        scored and then the best k are kept. Truncating before scoring would
        reintroduce the ranking the reranker exists to correct.
        """
        if not passages:
            return []

        model = self._load()
        pairs = [(query, p) for p in passages]
        scores = model.predict(pairs, batch_size=self.batch_size,
                               show_progress_bar=False)

        ordered = sorted(
            (Scored(index=i, score=float(s), original_rank=i + 1)
             for i, s in enumerate(scores)),
            key=lambda c: c.score,
            reverse=True,
        )
        return ordered if top_k is None else ordered[:top_k]


def movement(scored: list[Scored]) -> dict:
    """How much the reranker actually changed the order.

    WHY THIS IS REPORTED
    A reranker that improves recall@1 while barely moving anything is suspicious:
    it suggests the gain came from one or two questions rather than from a
    systematic reordering, and one or two questions on a 43-question set is
    noise. Conversely a reranker that churns the order violently and does not
    improve recall has actively made things worse in a way an aggregate hides.

    Both need seeing, so both are measured.
    """
    if not scored:
        return {"promoted": 0, "demoted": 0, "unchanged": 0, "max_promotion": 0}

    promoted = demoted = unchanged = 0
    best_jump = 0
    for new_rank, candidate in enumerate(scored, start=1):
        delta = candidate.original_rank - new_rank
        if delta > 0:
            promoted += 1
            best_jump = max(best_jump, delta)
        elif delta < 0:
            demoted += 1
        else:
            unchanged += 1
    return {"promoted": promoted, "demoted": demoted,
            "unchanged": unchanged, "max_promotion": best_jump}

# Reciprocal Rank Fusion constant, the same 60 hybrid_search uses. Kept
# identical deliberately: two fusions in one system with different constants
# invites the question "why", and there is no answer here beyond convention.
RRF_K = 60


def fuse(scored: list[Scored], top_k: int | None = None) -> list[Scored]:
    """Combine the retrieval order with the reranker's order, by rank.

    WHY FUSING BEATS REPLACING
    Replacing the ranking with the cross-encoder's score discards everything
    retrieval knew. Measured on this corpus, that is not a neutral trade:

        exact_term   recall@1  0.857 -> 0.810     lexical signal thrown away
        paraphrase   recall@5  0.955 -> 1.000     semantic judgement gained

    Identifier questions were already being answered correctly *because* the
    literal token matched. A cross-encoder scores semantic plausibility, so it
    will promote a passage that reads like a better answer over one that
    actually contains `reclaimPolicy` - and on those questions the literal match
    was the right signal.

    Fusing keeps both. A chunk ranked highly by retrieval AND by the reranker
    rises; one favoured by only a single method is moderated by the other rather
    than being allowed to win outright.

    By rank, not by score, for the same reason hybrid_search fuses by rank:
    cosine distance is bounded, RRF scores are small positives, and a
    cross-encoder logit is an unbounded real number that here ranges from about
    -11 to -2. Those three cannot be added meaningfully. Ranks can.
    """
    if not scored:
        return []

    rerank_position = {c.index: new for new, c in enumerate(scored, start=1)}

    fused = sorted(
        scored,
        key=lambda c: (1.0 / (RRF_K + c.original_rank))
                      + (1.0 / (RRF_K + rerank_position[c.index])),
        reverse=True,
    )
    return fused if top_k is None else fused[:top_k]
