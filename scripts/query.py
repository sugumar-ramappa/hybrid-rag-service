"""
CLI script to query the RAG pipeline interactively.

Usage:
    python -m scripts.query
    python -m scripts.query "What is Python?"

PYTHON CONCEPTS FOR JAVA DEVS:
- input() = like Java's Scanner.nextLine()
- sys.argv = like Java's String[] args
"""

import logging
import sys

from src.config import get_config
from src.rag.pipeline import RAGPipeline


def main() -> None:
    """Interactive RAG query CLI."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        config = get_config()
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Copy .env.example to .env and fill in your values.")
        sys.exit(1)

    pipeline = RAGPipeline(config)
    stats = pipeline.get_stats()
    print(f"\n📚 RAG Pipeline Ready | {stats['document_count']} chunks indexed")
    print(f"   Model: {stats['model']} | Embeddings: {stats['embedding_model']}")
    print("   Type 'quit' to exit\n")

    # If a question was passed as argument, answer it and exit
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        _answer_question(pipeline, question)
        return

    # Interactive loop
    while True:
        try:
            question = input("❓ Ask: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        _answer_question(pipeline, question)


def _answer_question(pipeline: RAGPipeline, question: str) -> None:
    """Process a single question."""
    print("\n🔍 Searching...\n")
    result = pipeline.query(question)

    print(f"📝 Answer:\n{result.answer}\n")

    if result.sources:
        print("📎 Sources:")
        for s in result.sources:
            source = s.metadata.get("source", "unknown")
            page = s.metadata.get("page", "")
            page_info = f" (page {page})" if page else ""
            print(f"   - {source}{page_info} [score: {s.score:.4f}]")
    print()


if __name__ == "__main__":
    main()
