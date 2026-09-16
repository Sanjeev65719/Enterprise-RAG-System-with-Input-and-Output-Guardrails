from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.orchestrator.graph import RAGOrchestrator
from src.core.logger import logger
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(title="Enterprise RAG Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
orchestrator = RAGOrchestrator()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    status: str
    response: str = None
    context: str = None
    iterations: int = 0
    message: str = None
    errors: list = []

@app.get("/api/health")
async def root():
    return {"message": "Enterprise RAG Agent is Online", "status": "healthy"}

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

@app.post("/query", response_model=QueryResponse)
async def ask_agent(request: QueryRequest):
    try:
        logger.info(f"API Request received: {request.query}")
        result = orchestrator.run(request.query)
        return result
    except Exception as e:
        logger.error(f"API Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run on localhost:8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
