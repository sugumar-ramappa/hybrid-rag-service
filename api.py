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


@app.get("/stats")
def stats():
    """Get pipeline statistics."""
    return _get_pipeline().get_stats()


@app.post("/ingest")
def ingest():
    """Ingest documents from the documents directory."""
    count = _get_pipeline().ingest_documents()
    return {"chunks_ingested": count}
