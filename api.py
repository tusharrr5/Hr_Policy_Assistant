#!/usr/bin/env python3
"""
api.py — FastAPI microservice for the HR Policy Assistant.

Endpoints:
  GET  /health  → Health check
  POST /ask     → Ask a question to the HR Policy RAG engine
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from rag_engine import HRPolicyRAG

# ── Global RAG instance ────────────────────────────────────────────────
rag_engine: HRPolicyRAG | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG engine on startup."""
    global rag_engine
    print("Initializing RAG engine...")
    try:
        rag_engine = HRPolicyRAG()
    except RuntimeError as e:
        print(f"Notice: {e}")
        print("Falling back to offline evaluation mode (mock LLM + real retrieval & guardrails).")
        rag_engine = HRPolicyRAG(mock_llm=True)
    print("✓ RAG engine ready.")
    yield
    print("Shutting down.")


app = FastAPI(
    title="HR Policy Assistant API",
    description="RAG-powered Q&A over the Enso Group HR Policy Manual.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Schemas ─────────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    question: str
    formatted_response: str
    is_supported: bool
    page_number: int | None
    section_title: str | None
    latency_ms: float


# ── Endpoints ───────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "engine_loaded": rag_engine is not None,
    }


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """Ask a question to the HR Policy Assistant."""
    result = rag_engine.query(request.question, stateless=True)
    return AskResponse(
        question=request.question,
        formatted_response=result["formatted_response"],
        is_supported=result["is_supported"],
        page_number=result["page_number"],
        section_title=result["section_title"],
        latency_ms=result["latency_ms"],
    )


# ── CLI entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    print("Starting HR Policy Assistant API on http://localhost:8000")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
