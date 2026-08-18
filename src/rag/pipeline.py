"""
RAG Pipeline - orchestrates the full Retrieval Augmented Generation flow.

This is the main entry point that ties everything together.
Think of it like a Spring @Service that coordinates other services.

FLOW:
1. INGEST: Load docs → Split into chunks → Generate embeddings → Store in ChromaDB
2. QUERY:  User question → Embed query → Search ChromaDB → Build context → Ask Gemini
"""

import logging
import time
from dataclasses import dataclass

from google import genai
from google.genai import errors as genai_errors

from src.config import AppConfig
from src.rag.document_loader import DocumentLoader
from src.rag.embeddings import EmbeddingService
from src.rag.text_splitter import TextSplitter
from src.rag.vector_store import SearchResult, VectorStore

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """The result of a RAG query."""
    answer: str
    sources: list[SearchResult]
    query: str


class RAGPipeline:
    """
    Orchestrates the full RAG pipeline.

    Like a Spring @Service with constructor injection:
        @Service
        public class RAGPipeline {
            private final DocumentLoader loader;
            private final TextSplitter splitter;
            ...
            @Autowired
            public RAGPipeline(DocumentLoader loader, ...) { ... }
        }
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config

        # Initialize Google AI client
        self._genai_client = genai.Client(api_key=config.gemini.api_key)

        # Initialize components (like dependency injection)
        self._loader = DocumentLoader(config.rag.documents_dir)
        self._splitter = TextSplitter(
            chunk_size=config.rag.chunk_size,
            chunk_overlap=config.rag.chunk_overlap,
        )
        self._embeddings = EmbeddingService(
            api_key=config.gemini.api_key,
            model_name=config.gemini.embedding_model,
            dimensions=config.gemini.embedding_dim,
        )
        self._vector_store = VectorStore(
            database_url=config.database.url,
            embedding_dim=config.gemini.embedding_dim,
            embedding_model=config.gemini.embedding_model,
        )
        self._model_name = config.gemini.model_name

        logger.info("RAG pipeline initialized")

    def ingest_documents(self) -> int:
        """
        Ingest all documents from the documents directory.

        Returns: Number of chunks ingested
        """
        # Step 1: Load documents
        documents = self._loader.load_all()
        if not documents:
            logger.warning("No documents found in %s", self._config.rag.documents_dir)
            return 0

        # Step 2: Split into chunks
        chunks = self._splitter.split_documents(documents)
        if not chunks:
            logger.warning("No chunks generated from documents")
            return 0

        # Step 3: Generate embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = self._embeddings.embed_texts(texts)

        # Step 4: Store in vector store
        count = self._vector_store.add_chunks(chunks, embeddings)
        logger.info("Ingested %d chunks from %d documents", count, len(documents))
        return count

    def query(self, question: str) -> QueryResult:
        """
        Answer a question using RAG.

        Flow:
        1. Embed the question
        2. Search for relevant chunks
        3. Build a prompt with context
        4. Ask Gemini to answer
        """
        logger.info("Query: %s", question[:100])

        start = time.time()

        # Step 1: Embed the query
        query_embedding = self._embeddings.embed_query(question)

        # Step 2: Retrieve relevant chunks
        results = self._vector_store.search(
            query_embedding=query_embedding,
            top_k=self._config.rag.top_k_results,
        )

        if not results:
            return QueryResult(
                answer="No relevant documents found. Please ingest documents first.",
                sources=[],
                query=question,
            )

        # Step 3: Build context from retrieved chunks
        context = self._build_context(results)

        # Step 4: Generate answer using Gemini (with retry for rate limits)
        prompt = self._build_prompt(question, context)
        answer = self._generate_with_retry(prompt)

        logger.info("Query answered in %dms, %d sources used",
            int((time.time() - start) * 1000), len(results))
        return QueryResult(answer=answer, sources=results, query=question)

    def _generate_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        """Call Gemini with retry on rate limit (503/429) errors."""
        for attempt in range(max_retries):
            try:
                response = self._genai_client.models.generate_content(
                    model=self._model_name,
                    contents=prompt,
                )
                return response.text
            except genai_errors.ServerError as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt * 5  # 5s, 10s, 20s
                    logger.warning("Rate limited (attempt %d/%d), retrying in %ds", attempt + 1, max_retries, wait)
                    time.sleep(wait)
                else:
                    raise

    def _build_context(self, results: list[SearchResult]) -> str:
        """Build context string from search results."""
        context_parts: list[str] = []
        for i, result in enumerate(results, start=1):
            source = result.metadata.get("source", "unknown")
            page = result.metadata.get("page", "")
            page_info = f" (page {page})" if page else ""
            context_parts.append(
                f"[Source {i}: {source}{page_info}]\n{result.content}"
            )
        return "\n\n---\n\n".join(context_parts)

    def _build_prompt(self, question: str, context: str) -> str:
        """Build the prompt for Gemini with RAG context."""
        return f"""You are a helpful assistant that answers questions based on the provided context.
Use ONLY the information from the context below to answer the question.
If the context doesn't contain enough information to answer, say so clearly.
Always cite which source(s) you used.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    def get_stats(self) -> dict:
        """Get pipeline statistics."""
        return {
            "document_count": self._vector_store.get_document_count(),
            "model": self._config.gemini.model_name,
            "embedding_model": self._config.gemini.embedding_model,
        }
