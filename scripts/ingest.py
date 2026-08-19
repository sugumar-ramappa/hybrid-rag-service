"""
CLI script to ingest documents into the RAG pipeline.

    python -m scripts.ingest                          # corpus from .env
    python -m scripts.ingest --dir walkthrough        # the small test document
    python -m scripts.ingest --max-files 20           # cap this run
    python -m scripts.ingest --dry-run                # what would it cost?

The --dir flag exists rather than `export DOCUMENTS_DIR=...` because an
exported variable lingers for the rest of the shell session: set it once for a
test and every later ingest silently reads the wrong directory. A flag applies
to one invocation and shows up in shell history.

PYTHON CONCEPTS FOR JAVA DEVS:
- if __name__ == "__main__" = like Java's public static void main(String[] args)
- argparse = like picocli or Apache Commons CLI
- dataclasses.replace = copy-with-changes on a frozen record, since AppConfig
  is immutable
"""

import argparse
import logging
import sys
import time
from dataclasses import replace

from src.config import AppConfig, get_config
from src.rag.pipeline import RAGPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest documents into the vector store.")
    parser.add_argument(
        "--dir",
        metavar="PATH",
        help="corpus directory for this run only (overrides DOCUMENTS_DIR)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        metavar="N",
        help="read at most N files this run (overrides MAX_FILES; 0 means no limit)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be embedded without calling the API or writing",
    )
    return parser.parse_args()


def apply_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    """Return a copy of config with any command-line overrides applied."""
    if args.dir is None and args.max_files is None:
        return config

    return replace(
        config,
        rag=replace(
            config.rag,
            documents_dir=args.dir if args.dir is not None else config.rag.documents_dir,
            max_files=args.max_files if args.max_files is not None else config.rag.max_files,
        ),
    )


def main() -> None:
    """Ingest documents into the vector store."""
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    try:
        config = apply_overrides(get_config(), args)
    except ValueError as e:
        logger.error("Configuration error: %s", e)
        logger.error("Copy .env.example to .env and fill in your values.")
        sys.exit(1)

    logger.info("Documents directory: %s", config.rag.documents_dir)
    if config.rag.max_files:
        logger.info("File cap: %d", config.rag.max_files)
    if args.dry_run:
        logger.info("Dry run - no API calls, nothing written")

    start = time.time()
    pipeline = RAGPipeline(config)
    result = pipeline.ingest_documents(dry_run=args.dry_run)
    elapsed = time.time() - start

    if result.total_chunks == 0:
        logger.warning("No documents found in %s", config.rag.documents_dir)
        logger.info("Run ./scripts/fetch_corpus.sh, or add .txt/.md/.pdf files there.")
        return

    if args.dry_run:
        would_embed = result.total_chunks - result.skipped
        logger.info(
            "Dry run: %d chunks total, %d already embedded, %d would be embedded",
            result.total_chunks, result.skipped, would_embed,
        )
        return

    logger.info(
        "✓ %d chunks in %.1f seconds — %d embedded, %d reused, %d removed",
        result.total_chunks, elapsed, result.embedded, result.skipped, result.deleted,
    )
    if result.embedded == 0:
        logger.info("Nothing changed since the last run, so no embedding calls were made.")


if __name__ == "__main__":
    main()
