"""
Google ADK (Agent Development Kit) Agent - an AI agent that uses RAG.

WHAT IS GOOGLE ADK?
Google ADK is a framework for building AI agents. An agent is an LLM that can:
1. Understand user intent
2. Decide which tools to use
3. Call tools and use the results
4. Provide a final answer

Think of it like a smart controller that orchestrates tool calls.

WHAT IS AN AGENT vs RAG?
- RAG = "search docs, build context, ask LLM" (a pipeline)
- Agent = LLM that DECIDES when to search, what to search, and how to combine results
  The agent might search multiple times, refine queries, or combine tools.

PYTHON CONCEPTS FOR JAVA DEVS:
- async def = like Java's CompletableFuture/Mono
- await = like .get() or .block() but non-blocking
- Callable = like Java's Function<T, R>
"""

import logging
import os

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from src.config import AppConfig, get_config
from src.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# TOOL FUNCTIONS (what the agent can call)
# ──────────────────────────────────────────────

# Module-level pipeline (initialized lazily)
_pipeline: RAGPipeline | None = None


def _ensure_pipeline(config: AppConfig) -> RAGPipeline:
    """Initialize pipeline if needed."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(config)
    return _pipeline


def search_knowledge_base(query: str) -> dict:
    """
    Search the document knowledge base for information.

    Args:
        query: The question or topic to search for.

    Returns:
        A dictionary with the answer and source information.
    """
    config = get_config()
    pipeline = _ensure_pipeline(config)
    try:
        result = pipeline.query(query)
    except Exception as e:
        logger.exception("Search failed for query: %s", query)
        return {"error": str(e), "query": query}
    return {
        "answer": result.answer,
        "sources": [
            {
                "content_preview": s.content[:200],
                "source": s.metadata.get("source", "unknown"),
                "relevance_score": s.score,
            }
            for s in result.sources
        ],
        "query": result.query,
    }


def ingest_new_documents() -> dict:
    """
    Ingest new documents from the documents directory.

    Returns:
        A dictionary with the number of chunks ingested.
    """
    config = get_config()
    pipeline = _ensure_pipeline(config)
    try:
        count = pipeline.ingest_documents()
    except Exception as e:
        logger.exception("Ingestion failed")
        return {"error": str(e), "status": "failed"}
    return {"chunks_ingested": count, "status": "success"}


def get_knowledge_base_stats() -> dict:
    """
    Get statistics about the knowledge base.

    Returns:
        A dictionary with document count and model information.
    """
    config = get_config()
    pipeline = _ensure_pipeline(config)
    try:
        return pipeline.get_stats()
    except Exception as e:
        logger.exception("Failed to get stats")
        return {"error": str(e)}


# ──────────────────────────────────────────────
# AGENT CREATION
# ──────────────────────────────────────────────


def create_rag_agent(config: AppConfig) -> Agent:
    """
    Create a Google ADK agent with RAG tools.

    This is like creating a Spring Bean with all its dependencies wired up.
    """
    # Set the API key for Google ADK to use
    os.environ["GOOGLE_API_KEY"] = config.gemini.api_key

    agent = Agent(
        name="rag_assistant",
        model=config.gemini.model_name,
        description="An AI assistant that answers questions from a document knowledge base.",
        instruction="""You are a helpful document assistant. You help users find information
from their documents.

RULES:
1. Always use the search_knowledge_base tool to find information before answering.
2. If the user asks to ingest/add documents, use the ingest_new_documents tool.
3. If the user asks about the knowledge base status, use get_knowledge_base_stats.
4. Always cite your sources when answering questions.
5. If no relevant information is found, say so honestly.
6. Be concise but thorough in your answers.""",
        tools=[
            search_knowledge_base,
            ingest_new_documents,
            get_knowledge_base_stats,
        ],
    )

    return agent


# ──────────────────────────────────────────────
# RUNNER (executes the agent)
# ──────────────────────────────────────────────

# Shared state for conversation memory across turns
_session_service: InMemorySessionService | None = None
_runner: Runner | None = None
_session_id: str | None = None


async def run_agent_query(question: str) -> str:
    """
    Run a query through the ADK agent.

    Reuses session across calls so the agent remembers previous turns.
    """
    global _session_service, _runner, _session_id

    config = get_config()

    if _runner is None:
        agent = create_rag_agent(config)
        _session_service = InMemorySessionService()
        _runner = Runner(
            agent=agent,
            app_name="rag_app",
            session_service=_session_service,
        )

    if _session_id is None:
        session = await _session_service.create_session(
            app_name="rag_app",
            user_id="user",
        )
        _session_id = session.id

    from google.genai.types import Content, Part

    user_message = Content(
        role="user",
        parts=[Part.from_text(text=question)],
    )

    response_parts: list[str] = []
    async for event in _runner.run_async(
        user_id="user",
        session_id=_session_id,
        new_message=user_message,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    response_parts.append(part.text)

    return "\n".join(response_parts) if response_parts else "No response generated."
