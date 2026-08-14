"""
CLI script to ingest documents into the RAG pipeline.

Usage:
    python -m scripts.ingest

PYTHON CONCEPTS FOR JAVA DEVS:
- if __name__ == "__main__" = like Java's public static void main(String[] args)
- argparse = like Java's Apache Commons CLI or picocli
"""

import logging
import sys
import time

from src.config import get_config
from src.rag.pipeline import RAGPipeline


def main() -> None:
    """Ingest documents into the vector store."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    try:
        config = get_config()
    except ValueError as e:
        logger.error("Configuration error: %s", e)
        logger.error("Copy .env.example to .env and fill in your values.")
        sys.exit(1)

    logger.info("Starting document ingestion...")
    logger.info("Documents directory: %s", config.rag.documents_dir)

    start = time.time()
    pipeline = RAGPipeline(config)
    count = pipeline.ingest_documents()
    elapsed = time.time() - start

    if count > 0:
        logger.info("✓ Ingested %d chunks in %.1f seconds", count, elapsed)
    else:
        logger.warning("No documents found in %s", config.rag.documents_dir)
        logger.info("Add .txt, .md, or .pdf files to that directory and try again.")


if __name__ == "__main__":
    main()
