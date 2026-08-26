"""
FastAPI REST API for the RAG pipeline.

Usage:
    pip install fastapi uvicorn
    uvicorn api:app --reload
    Open http://localhost:8000/docs for Swagger UI
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import get_config
from src.rag.pipeline import RAGPipeline

app = FastAPI(
    title="RAG Document Q&A API",
    description="Search documents using RAG pipeline",
    version="0.1.0",
)

_pipeline: RAGPipeline | None = None


def _get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(get_config())
    return _pipeline


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    query: str


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """Search documents and get an answer."""
    pipeline = _get_pipeline()
    result = pipeline.query(request.question)
    return QueryResponse(
        answer=result.answer,
        sources=[{"source": s.metadata.get("source"), "score": s.score} for s in result.sources],
        query=result.query,
    )


@app.get("/healthz")
def healthz():
    """Is the process alive? Deliberately checks NOTHING else.

    This is the liveness probe, and a liveness probe that touches a dependency is
    a mistake: Kubernetes responds to liveness failure by RESTARTING the
    container. Point it at the database and a database blip restarts every
    healthy pod at once, turning a brief outage into a crash loop that outlives
    it.

    Readiness is the probe that may check dependencies, because its failure
    removes the pod from the Service instead of killing it. That is /stats below.
    """
    return {"status": "ok"}


@app.get("/stats")
def stats():
    """Get pipeline statistics.

    Doubles as the READINESS probe. It constructs the pipeline and queries the
    corpus, so a pod that cannot reach Postgres or has an empty index fails it
    and is taken out of the Service - which is correct, because such a pod would
    answer every question with nothing found.
    """
    return _get_pipeline().get_stats()


@app.post("/ingest")
def ingest():
    """Ingest documents from the documents directory."""
    result = _get_pipeline().ingest_documents()
    return {
        "total_chunks": result.total_chunks,
        "embedded": result.embedded,
        "skipped": result.skipped,
        "deleted": result.deleted,
    }
