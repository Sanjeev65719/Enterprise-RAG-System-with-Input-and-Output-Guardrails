import os
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from src.orchestrator.graph import RAGOrchestrator
from src.core.logger import logger
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Enterprise RAG Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_orchestrator = None

def get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = RAGOrchestrator()
    return _orchestrator


class QueryRequest(BaseModel):
    query: str
    org_id: Optional[str] = "default-org-123"
    user_id: Optional[str] = "user-456"
    acl_tags: Optional[List[str]] = None


class QueryResponse(BaseModel):
    status: str
    response: Optional[str] = None
    context: Optional[str] = None
    iterations: int = 0
    message: Optional[str] = None
    errors: List[str] = []
    request_id: Optional[str] = None
    verdict: Optional[str] = None
    audit_trail: Optional[List[Dict[str, Any]]] = None


@app.get("/api/health")
async def health_check():
    """Render health-check endpoint."""
    has_key = bool(os.getenv("OPENAI_API_KEY"))
    return {
        "message": "Enterprise RAG Agent is Online",
        "status": "healthy",
        "openai_configured": has_key
    }


@app.post("/query", response_model=QueryResponse)
async def ask_agent(request: QueryRequest):
    try:
        logger.info(f"API Request received: {request.query}")
        orchestrator = get_orchestrator()
        result = orchestrator.run(
            query=request.query,
            org_id=request.org_id or "default-org-123",
            user_id=request.user_id or "user-456",
            acl_tags=request.acl_tags
        )
        return result
    except Exception as e:
        logger.error(f"API Error: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "response": None,
                "context": None,
                "iterations": 0,
                "message": f"Orchestrator error: {str(e)}",
                "errors": [str(e)],
                "request_id": None,
                "verdict": "ERROR",
                "audit_trail": []
            }
        )


# ── Static frontend — MUST be mounted LAST so API routes take priority ──
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
