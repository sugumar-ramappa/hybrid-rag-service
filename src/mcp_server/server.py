"""
MCP Server - exposes RAG pipeline as MCP tools.

WHAT IS MCP (Model Context Protocol)?
MCP is a standard protocol that lets AI assistants (like Claude) call your code.
Think of it like a REST API, but specifically designed for LLMs.

Key MCP concepts:
- Tool = like an API endpoint. LLM can call it with parameters.
- Resource = like a GET endpoint that returns data.
- Server = your application that exposes tools/resources.

PYTHON CONCEPTS FOR JAVA DEVS:
- @mcp.tool() = like Spring's @GetMapping / @PostMapping
- async def = like CompletableFuture methods
- Annotated[str, ...] = like Java's @Param annotations
"""

import json
import logging
from typing import Annotated

from mcp.server import MCPServer

from src.config import AppConfig, get_config
from src.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

# Create the MCP server (like creating a Spring Boot application)
mcp = MCPServer(
    name="rag-mcp-server",
    version="0.1.0",
    description="A RAG pipeline that searches your documents using Vertex AI",
)

# Lazy initialization (pipeline is created on first use)
_pipeline: RAGPipeline | None = None


def _get_pipeline() -> RAGPipeline:
    """Lazy singleton - like Spring's @Lazy @Bean."""
    global _pipeline
    if _pipeline is None:
        config = get_config()
        _pipeline = RAGPipeline(config)
    return _pipeline


# ──────────────────────────────────────────────
# MCP TOOLS (like REST endpoints for AI)
# ──────────────────────────────────────────────


@mcp.tool()
def search_documents(
    query: Annotated[str, "The question to search for in the documents"],
    top_k: Annotated[int, "Number of results to return (default 5)"] = 5,
) -> str:
    """
    Search ingested documents and get an AI-generated answer.

    Use this tool when you need to find information from the document knowledge base.
    The tool searches through all ingested documents and returns a comprehensive answer
    with source citations.
    """
    try:
        pipeline = _get_pipeline()
        result = pipeline.query(query)

        # Format response for the LLM
        sources_info = []
        for s in result.sources:
            source = s.metadata.get("source", "unknown")
            page = s.metadata.get("page", "")
            score = f"{s.score:.4f}"
            sources_info.append(
                f"  - {source}" + (f" (page {page})" if page else "") + f" [relevance: {score}]"
            )

        return (
            f"**Answer:**\n{result.answer}\n\n"
            f"**Sources used:**\n" + "\n".join(sources_info)
        )
    except Exception as e:
        logger.exception("Search failed")
        return f"Error searching documents: {e}"


@mcp.tool()
def ingest_documents() -> str:
    """
    Ingest all documents from the documents directory into the vector store.

    Call this tool when new documents have been added and need to be indexed
    for searching. This will load, split, embed, and store all documents.
    """
    try:
        pipeline = _get_pipeline()
        count = pipeline.ingest_documents()
        return f"Successfully ingested {count} document chunks into the vector store."
    except Exception as e:
        logger.exception("Ingestion failed")
        return f"Error ingesting documents: {e}"


@mcp.tool()
def get_pipeline_stats() -> str:
    """
    Get statistics about the RAG pipeline.

    Returns information about the number of indexed documents,
    the models being used, and the current configuration.
    """
    try:
        pipeline = _get_pipeline()
        stats = pipeline.get_stats()
        return json.dumps(stats, indent=2)
    except Exception as e:
        logger.exception("Failed to get stats")
        return f"Error getting stats: {e}"


# ──────────────────────────────────────────────
# MCP RESOURCES (like GET endpoints)
# ──────────────────────────────────────────────


@mcp.resource("rag://config")
def get_config_resource() -> str:
    """Returns the current RAG configuration."""
    try:
        config = get_config()
        return json.dumps({
            "model": config.gemini.model_name,
            "embedding_model": config.gemini.embedding_model,
            "chunk_size": config.rag.chunk_size,
            "chunk_overlap": config.rag.chunk_overlap,
            "top_k": config.rag.top_k_results,
        }, indent=2)
    except Exception as e:
        return f"Error: {e}"


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────


def main() -> None:
    """Start the MCP server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("Starting RAG MCP Server...")
    mcp.run()


if __name__ == "__main__":
    main()
