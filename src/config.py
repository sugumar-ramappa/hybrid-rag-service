"""
Configuration module - similar to Spring's @Configuration + application.properties

PYTHON CONCEPTS FOR JAVA DEVS:
- dataclass = like a Java record or Lombok @Data
- os.getenv() = like System.getenv()
- Optional[str] = like Java's Optional<String>
- Path = like Java's java.nio.file.Path
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env file (like Spring reading application.properties)
load_dotenv()


@dataclass(frozen=True)  # frozen=True makes it immutable (like Java's record)
class GeminiConfig:
    """Google Gemini configuration (via Google AI Studio)."""
    api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    model_name: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"))
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
    )
    # pgvector's HNSW index supports at most 2000 dimensions, and this model
    # returns 3072 by default - which would leave the table unindexed and turn
    # every query into a sequential scan. 768 is a size the model supports
    # directly (Matryoshka truncation), and is a common production dimension.
    #
    # Kept configurable so the eval harness can measure what the truncation
    # costs. Must match the vector(N) column in src/rag/schema.sql - changing it
    # requires a schema migration and a full re-ingest.
    embedding_dim: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "768"))
    )
    # Texts per embedding request.
    #
    # The free-tier ceiling is tokens per minute, not requests, so batch size
    # does not change total throughput - it changes exposure. At ~250 tokens a
    # chunk, a batch of 100 is ~25k tokens: 83% of a 30k/min budget in one
    # call, close enough that a batch of longer-than-average chunks is rejected
    # outright rather than throttled. It also decides how much work is lost
    # when a request fails.
    embed_batch_size: int = field(
        default_factory=lambda: int(os.getenv("EMBED_BATCH_SIZE", "50"))
    )


@dataclass(frozen=True)
class DatabaseConfig:
    """Postgres + pgvector connection settings."""
    url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", ""))


@dataclass(frozen=True)
class RAGConfig:
    """RAG pipeline configuration."""
    chroma_persist_dir: str = field(
        default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
    )
    documents_dir: str = field(
        default_factory=lambda: os.getenv("DOCUMENTS_DIR", "./documents")
    )
    chunk_size: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE", "1000"))
    )
    chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "200"))
    )
    top_k_results: int = field(
        default_factory=lambda: int(os.getenv("TOP_K_RESULTS", "5"))
    )
    # Cap on how many files a single ingest reads. 0 means no limit.
    #
    # Embedding is the expensive step, so this bounds both API spend and the
    # wall-clock cost of a rate-limited run. Because the loader walks files in
    # sorted order, the same cap always selects the same files - so a golden
    # set built against a capped corpus stays valid.
    #
    # Raising it later is cheap: incremental ingest embeds only the files the
    # higher cap newly includes.
    max_files: int = field(
        default_factory=lambda: int(os.getenv("MAX_FILES", "0"))
    )
    collection_name: str = "rag_documents"


@dataclass(frozen=True)
class AppConfig:
    """
    Root configuration - like Spring's main @Configuration class.
    Aggregates all sub-configs.
    """
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)

    def validate(self) -> None:
        """Validate required configuration is present."""
        if not self.gemini.api_key:
            raise ValueError(
                "GOOGLE_API_KEY is required. "
                "Get a free API key from https://aistudio.google.com/apikey "
                "and set it in .env"
            )
        if not self.database.url:
            raise ValueError(
                "DATABASE_URL is required, e.g. "
                "postgresql://postgres:dev@localhost:5432/ragdb - "
                "start one with: docker run -d --name ragdb -p 5432:5432 "
                "-e POSTGRES_PASSWORD=dev -e POSTGRES_DB=ragdb pgvector/pgvector:pg16"
            )


def get_config() -> AppConfig:
    """
    Factory function to create config - like Spring's @Bean method.
    This is the single entry point for configuration.
    """
    config = AppConfig()
    config.validate()
    return config
