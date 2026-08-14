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
    collection_name: str = "rag_documents"


@dataclass(frozen=True)
class AppConfig:
    """
    Root configuration - like Spring's main @Configuration class.
    Aggregates all sub-configs.
    """
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)

    def validate(self) -> None:
        """Validate required configuration is present."""
        if not self.gemini.api_key:
            raise ValueError(
                "GOOGLE_API_KEY is required. "
                "Get a free API key from https://aistudio.google.com/apikey "
                "and set it in .env"
            )


def get_config() -> AppConfig:
    """
    Factory function to create config - like Spring's @Bean method.
    This is the single entry point for configuration.
    """
    config = AppConfig()
    config.validate()
    return config
