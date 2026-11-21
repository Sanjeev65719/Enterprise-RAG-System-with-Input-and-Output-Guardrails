import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import uvicorn

# Ensure the project root is on sys.path so `src` package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.orchestrator.graph import RAGOrchestrator

# Load environment variables
load_dotenv()

app = FastAPI(
    title="SentinelRAG API",
    description="Enterprise RAG with Multiagent Guardrails",
    version="1.0.0"
)

from fastapi.responses import JSONResponse

_orchestrator = None

def get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = RAGOrchestrator()
    return _orchestrator

class QueryRequest(BaseModel):
    query: str = Field(..., example="What is the security policy for PII?", description="The user's query text")
    org_id: Optional[str] = Field(default="default-org-123", description="Organization identifier for data isolation")
    user_id: Optional[str] = Field(default="user-456", description="User identifier for access control")
    acl_tags: Optional[List[str]] = Field(default=None, example=["confidential", "hr"], description="Optional access control tags")

class QueryResponse(BaseModel):
    request_id: Optional[str] = None
    verdict: Optional[str] = None
    response: Optional[str] = None
    iterations: int = 0
    audit_trail: Optional[List[Dict[str, Any]]] = None
    status: str
    message: Optional[str] = None
    errors: List[str] = []
    context: Optional[str] = None

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "online", "service": "SentinelRAG API"}

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Submit a query to the SentinelRAG pipeline.
    The request is processed through the Sentry, Librarian, Generator, and Auditor.
    """
    try:
        # The orchestrator.run method is synchronous.
        # FastAPI runs sync endpoints in a separate threadpool to avoid blocking the event loop.
        orchestrator = get_orchestrator()
        result = orchestrator.run(
            query=request.query,
            org_id=request.org_id or "default-org-123",
            user_id=request.user_id or "user-456",
            acl_tags=request.acl_tags
        )
        return result
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "response": None,
                "context": None,
                "iterations": 0,
                "message": f"Internal Server Error: {str(e)}",
                "errors": [str(e)],
                "request_id": None,
                "verdict": "ERROR",
                "audit_trail": []
            }
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
