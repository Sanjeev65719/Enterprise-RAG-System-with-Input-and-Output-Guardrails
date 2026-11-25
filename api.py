import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from src.orchestrator.graph import RAGOrchestrator
from src.ingestion.pipeline import IngestionPipeline
from src.core.logger import logger
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Enterprise RAG Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_orchestrator = None
_ingestion_pipeline = None

def get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = RAGOrchestrator()
    return _orchestrator

def get_ingestion_pipeline():
    global _ingestion_pipeline
    if _ingestion_pipeline is None:
        _ingestion_pipeline = IngestionPipeline()
    return _ingestion_pipeline


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
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_groq = bool(os.getenv("GROQ_API_KEY"))
    provider = os.getenv("LLM_PROVIDER", "groq" if (has_groq and not has_openai) else ("openai" if has_openai else "none"))
    return {
        "message": "Enterprise RAG Agent is Online",
        "status": "healthy",
        "provider": provider,
        "llm_configured": has_openai or has_groq,
        "openai_configured": has_openai,
        "groq_configured": has_groq
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


@app.post("/api/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    org_id: Optional[str] = Form("default-org-123"),
    user_id: Optional[str] = Form("user-456"),
    acl_tags: Optional[str] = Form("INTERNAL")
):
    """Uploads and ingests a document directly into the vector store."""
    try:
        import tempfile
        import shutil

        # Save uploaded file temporarily to disk so parser can process it
        suffix = Path(file.filename or "upload.txt").suffix or ".txt"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name

        pipeline = get_ingestion_pipeline()
        tags_list = [t.strip() for t in acl_tags.split(",")] if acl_tags else ["INTERNAL"]

        chunks = pipeline.process_document(
            file_path=tmp_path,
            org_id=org_id or "default-org-123",
            user_id=user_id or "user-456",
            acl_tags=tags_list
        )

        try:
            os.remove(tmp_path)
        except Exception:
            pass

        return {
            "status": "success",
            "filename": file.filename,
            "chunks_ingested": len(chunks),
            "message": f"Successfully ingested {len(chunks)} chunks into vector store."
        }
    except Exception as e:
        logger.error(f"Ingestion API Error: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "filename": file.filename,
                "chunks_ingested": 0,
                "message": f"Failed to ingest document: {str(e)}"
            }
        )


# ── Static frontend — MUST be mounted LAST so API routes take priority ──
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
