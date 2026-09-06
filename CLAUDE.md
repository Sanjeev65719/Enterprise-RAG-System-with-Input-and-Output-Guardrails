# SentinelRAG: Enterprise RAG with Multiagent Guardrails

## 🎯 Project Overview
SentinelRAG is a high-assurance Enterprise RAG system designed to eliminate hallucinations and secure data boundaries. Unlike linear RAG chains, it implements a **Verified Loop** where an Auditor agent mathematically verifies the faithfulness of the response before it ever reaches the user.

## 🏗️ Architecture: The Verified Loop
The system transforms the traditional `Query -> Retrieve -> Generate` flow into a closed-loop pipeline:

1.  **The Sentry (Ingress Guard)**: `src/guardrails/input_guards.py`
2.  **The Librarian (Retrieval Guard)**: `src/core/rag_engine.py`
3.  **The Generator (Core Engine)**: `src/core/generator.py`
4.  **The Auditor (Egress Guard)**: `src/guardrails/output_guards.py`

## 🛠️ Tech Stack
- **Language**: Python 3.10+
- **Vector DB**: Qdrant (via `qdrant-client`)
- **LLM/Embeddings**: OpenAI (`gpt-4o`, `text-embedding-3-large`)
- **API Layer**: FastAPI / Uvicorn
- **Infrastructure**: Docker / Docker-Compose

## 📂 Key Components
- `src/ingestion/`: High-assurance parsing and semantic chunking.
- `src/core/`: The `RAGEngine` and `Generator`.
- `src/orchestrator/`: The `RAGOrchestrator` state machine.
- `src/guardrails/`: Logic for the Sentry and Auditor.
- `src/api/`: FastAPI implementation for production access.
- `tests/eval/`: LLM-as-a-Judge evaluation framework.

## 🚀 Development Guide

### Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Set environment variables in `.env`: `OPENAI_API_KEY`, `QDRANT_HOST`, `QDRANT_PORT`.
3. Start the stack: `docker-compose up --build`

### Running the Pipeline
- **API**: `uvicorn src.api.main:app --reload`
- **Benchmark**: `python tests/eval/benchmark.py`

### Testing Security Boundaries
Run the multi-tenancy leak tests: `python -m unittest tests/security/boundary_tests.py`

## 📋 Project Status & Roadmap

### Completed ✅
- [x] High-Assurance Ingestion Pipeline
- [x] Multi-tenant Data Boundary Schema in Qdrant
- [x] Verified Loop State Machine implementation
- [x] Operationalization of Guardrail agents (Sentry & Auditor)
- [x] **API Layer**: FastAPI implementation with `POST /query` endpoint
- [x] **Evaluation Framework**: Golden Dataset and LLM-as-a-Judge benchmark script
- [x] **Infrastructure**: Dockerfile and Docker-Compose orchestration

### Pending ⏳
- [ ] **Production Deployment**: Move from local Docker to Vercel + Qdrant Cloud.
- [ ] **Dataset Expansion**: Grow the "Golden Dataset" to cover more complex edge cases.
- [ ] **Advanced Guardrails**: Implement domain-specific PII filters beyond general sanitization.
