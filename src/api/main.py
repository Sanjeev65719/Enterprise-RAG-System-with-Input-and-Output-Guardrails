import os
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import uvicorn

from src.orchestrator.graph import RAGOrchestrator

# Load environment variables
load_dotenv()

app = FastAPI(
    title="SentinelRAG API",
    description="Enterprise RAG with Multiagent Guardrails",
    version="1.0.0"
)

# singleton orchestrator to avoid reloading configs on every request
orchestrator = RAGOrchestrator()

class QueryRequest(BaseModel):
    query: str = Field(..., example="What is the security policy for PII?", description="The user's query text")
    org_id: str = Field(..., example="org_12345", description="Organization identifier for data isolation")
    user_id: str = Field(..., example="user_67890", description="User identifier for access control")
    acl_tags: Optional[List[str]] = Field(default=None, example=["confidential", "hr"], description="Optional access control tags")

class QueryResponse(BaseModel):
    request_id: str
    verdict: str
    response: str
    iterations: int
    audit_trail: List[Dict[str, Any]]
    status: str

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
        result = orchestrator.run(
            query=request.query,
            org_id=request.org_id,
            user_id=request.user_id,
            acl_tags=request.acl_tags
        )
        return result
    except Exception as e:
        # Log the exception in a real system
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
