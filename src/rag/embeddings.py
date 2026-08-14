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

from google import genai

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Generates embeddings using Google AI Studio - like a Spring @Service.

    Uses Google's gemini-embedding-001 model (free tier).
    """

    _MAX_BATCH_SIZE = 100

    def __init__(self, api_key: str, model_name: str = "gemini-embedding-001") -> None:
        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name
        logger.info("Initialized embedding model: %s", model_name)

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

        for i in range(0, len(texts), self._MAX_BATCH_SIZE):
            batch = texts[i : i + self._MAX_BATCH_SIZE]

            result = self._client.models.embed_content(
                model=self._model_name,
                contents=batch,
                config={"task_type": "RETRIEVAL_DOCUMENT"},
            )
            all_embeddings.extend([e.values for e in result.embeddings])

            logger.debug("Embedded batch %d-%d of %d", i, i + len(batch), len(texts))

        logger.info("Generated %d embeddings", len(all_embeddings))
        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """
        Generate embedding for a single search query.
        Uses RETRIEVAL_QUERY task type for better search results.
        """
        result = self._client.models.embed_content(
            model=self._model_name,
            contents=query,
            config={"task_type": "RETRIEVAL_QUERY"},
        )
        return result.embeddings[0].values
