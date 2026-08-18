"""
Vector Store - stores and searches document embeddings using ChromaDB.

WHAT IS A VECTOR STORE?
Like a database, but instead of SQL queries, you search by similarity.
You give it a query vector, and it returns the closest document vectors.
Think of it like a KNN (K-Nearest Neighbors) search.

ChromaDB runs locally (no cloud cost!) and persists to disk.

PYTHON CONCEPTS FOR JAVA DEVS:
- dict = like Java's Map<String, Object>
- **kwargs = like Java's varargs but for named parameters
- None = like Java's null
- Optional[X] = like Java's @Nullable X
"""

import logging
from dataclasses import dataclass
from typing import Optional

import chromadb
from chromadb.config import Settings

from src.rag.text_splitter import TextChunk

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A search result from the vector store."""
    content: str
    metadata: dict
    score: float  # Lower = more similar (distance metric)


class VectorStore:
    """
    ChromaDB-based vector store - like a Spring @Repository.

    Stores document chunks as vectors and retrieves the most similar ones
    for a given query.
    """

    def __init__(self, persist_dir: str, collection_name: str = "rag_documents") -> None:
        """
        Initialize ChromaDB with persistent storage.

        Args:
            persist_dir: Directory to store the database (like a JDBC URL)
            collection_name: Name of the collection (like a table name)
        """
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # Cosine distance: lower score = more similar
        )
        logger.info(
            "Vector store ready: collection='%s', documents=%d",
            collection_name, self._collection.count()
        )

    def add_chunks(
        self,
        chunks: list[TextChunk],
        embeddings: list[list[float]],
    ) -> int:
        """
        Add document chunks with their embeddings to the store.

        Args:
            chunks: Text chunks to store
            embeddings: Corresponding embedding vectors

        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings"
            )

        # Stable, content-derived IDs. Using a positional counter here meant that
        # adding one document shifted every later chunk's ID, and the upsert below
        # silently overwrote unrelated chunks.
        ids = [chunk.chunk_id for chunk in chunks]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate chunk IDs in batch - refusing to upsert")

        documents = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        # Upsert = insert or update (like SQL MERGE)
        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info("Added %d chunks to vector store", len(chunks))
        return len(chunks)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: Optional[dict] = None,
    ) -> list[SearchResult]:
        """
        Search for the most similar documents to the query.

        Args:
            query_embedding: The query vector to search with
            top_k: Number of results to return (like SQL LIMIT)
            where: Optional metadata filter (like SQL WHERE)

        Returns:
            List of SearchResult, sorted by relevance
        """
        query_params: dict = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, self._collection.count() or top_k),
        }
        if where:
            query_params["where"] = where

        if self._collection.count() == 0:
            logger.warning("Vector store is empty. Ingest documents first.")
            return []

        results = self._collection.query(**query_params)

        search_results: list[SearchResult] = []
        # ChromaDB returns lists of lists (batch queries), we take first batch
        for i in range(len(results["ids"][0])):
            search_results.append(SearchResult(
                content=results["documents"][0][i],
                metadata=results["metadatas"][0][i],
                score=results["distances"][0][i],
            ))

        logger.info("Found %d results for query", len(search_results))
        return search_results

    def get_document_count(self) -> int:
        """Get the total number of documents in the store."""
        return self._collection.count()

    def clear(self) -> None:
        """Delete all documents from the collection."""
        self._client.delete_collection(self._collection.name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection.name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Vector store cleared")
