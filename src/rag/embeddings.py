"""
Embeddings - converts text into numerical vectors using Google AI Studio (Gemini).

WHAT ARE EMBEDDINGS?
Think of them as converting text to coordinates in a high-dimensional space.
Similar texts end up near each other. This is how we find relevant documents.

Example: "Python programming" and "coding in Python" would have similar vectors,
while "cooking recipes" would be far away.

PYTHON CONCEPTS FOR JAVA DEVS:
- list[list[float]] = like Java's List<List<Double>>
- google.genai = the Google AI Studio SDK (free, API key based)
"""

import logging
import math
import time

from google import genai
from google.genai import errors as genai_errors

logger = logging.getLogger(__name__)

# 429 (rate limited) and 5xx (transient server trouble) are worth retrying.
# Anything else - a bad key, a wrong model name - will fail identically no
# matter how long we wait.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _normalize(vector: list[float]) -> list[float]:
    """
    Scale a vector to unit length.

    Truncating a Matryoshka embedding to fewer dimensions leaves it no longer
    unit-length. Cosine distance is scale-invariant so ranking is unaffected
    either way, but normalizing keeps the vectors correct for inner-product and
    L2 operators too, which costs nothing and avoids a subtle trap later.
    """
    magnitude = math.sqrt(sum(v * v for v in vector))
    if magnitude == 0.0:
        return vector
    return [v / magnitude for v in vector]


class EmbeddingService:
    """
    Generates embeddings using Google AI Studio - like a Spring @Service.

    Uses Google's gemini-embedding-001 model (free tier).
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-embedding-001",
        dimensions: int = 768,
        batch_size: int = 50,
        max_retries: int = 4,
        cache=None,
    ) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name
        self._dimensions = dimensions
        self._max_retries = max_retries
        # Optional EmbeddingCache. Applied to queries only - document
        # embeddings are already protected from repeat work by incremental
        # ingest, which skips chunks whose content is unchanged.
        self._cache = cache
        # Public: callers slice their work by this so they can persist each
        # batch as it completes rather than losing everything on a late failure.
        self.batch_size = batch_size
        logger.info(
            "Initialized embedding model: %s (%d dimensions, batches of %d)",
            model_name, dimensions, batch_size,
        )

    def _embed_with_retry(self, contents, task_type: str):
        """
        Call the embedding API, retrying on rate limits and transient errors.

        Backoff starts at 30 seconds rather than the usual 1-2. The free-tier
        limit is tokens *per minute*, so once it is hit, retrying inside the
        same minute is guaranteed to fail again - the window has to roll over.
        """
        for attempt in range(self._max_retries):
            try:
                return self._client.models.embed_content(
                    model=self._model_name,
                    contents=contents,
                    config={
                        "task_type": task_type,
                        "output_dimensionality": self._dimensions,
                    },
                )
            except (genai_errors.ClientError, genai_errors.ServerError) as exc:
                status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
                message = str(exc)

                # A daily quota is not a transient condition. Both arrive as
                # 429 RESOURCE_EXHAUSTED, but one clears when the minute rolls
                # over and the other clears at midnight Pacific - so retrying
                # the second just burns minutes to fail anyway.
                if "PerDay" in message or "per day" in message.lower():
                    raise RuntimeError(
                        "Daily embedding quota exhausted - this will not recover by "
                        "waiting. Everything embedded so far is already stored, so "
                        "re-running tomorrow resumes from where this stopped. "
                        f"Original error: {message[:200]}"
                    ) from exc

                retryable = status in _RETRYABLE_STATUS or "RESOURCE_EXHAUSTED" in message

                if not retryable or attempt == self._max_retries - 1:
                    raise

                wait = 30 * (2 ** attempt)  # 30s, 60s, 120s
                logger.warning(
                    "Embedding rate limited (attempt %d/%d), waiting %ds: %s",
                    attempt + 1, self._max_retries, wait, message[:120],
                )
                time.sleep(wait)

        raise RuntimeError("unreachable: retry loop exhausted without returning")

    def _check_dim(self, vector: list[float]) -> list[float]:
        """Fail fast if the API returns a different size than the schema expects."""
        if len(vector) != self._dimensions:
            raise ValueError(
                f"Expected {self._dimensions}-dimension embeddings but got "
                f"{len(vector)}. The vector(N) column in schema.sql and "
                f"EMBEDDING_DIM must agree with what the model returns."
            )
        return _normalize(vector)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of strings to embed

        Returns:
            List of embedding vectors (each is a list of floats)
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            result = self._embed_with_retry(batch, "RETRIEVAL_DOCUMENT")
            all_embeddings.extend([self._check_dim(e.values) for e in result.embeddings])
            logger.debug("Embedded batch %d-%d of %d", i, i + len(batch), len(texts))

        logger.info("Generated %d embeddings", len(all_embeddings))
        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """
        Generate embedding for a single search query.
        Uses RETRIEVAL_QUERY task type for better search results.
        """
        if self._cache is not None:
            cached = self._cache.get(
                query, self._model_name, self._dimensions, "RETRIEVAL_QUERY"
            )
            if cached is not None:
                return cached

        result = self._embed_with_retry(query, "RETRIEVAL_QUERY")
        vector = self._check_dim(result.embeddings[0].values)

        if self._cache is not None:
            self._cache.put(
                query, self._model_name, self._dimensions, "RETRIEVAL_QUERY", vector
            )

        return vector
