# Project Status: SentinelRAG (Enterprise RAG with Multiagent Guardrails)

## 🎯 Project Goal
Build a high-assurance Enterprise RAG system that eliminates hallucinations and secures data boundaries using a closed-loop verification pipeline (multiagent guardrails) instead of a linear retrieval chain.

## 🏗️ Architecture Overview
The system transforms the linear `Query -> Retrieve -> Generate` flow into a `Verified Loop`:
1. **Ingress Guard (The Sentry):** Filters PII, prompt injections, and out-of-scope queries.
2. **Retrieval Guard (The Librarian):** Audits retrieved chunks for semantic relevance and eliminates noise.
3. **Core Engine (The Generator):** Synthesizes a grounded answer using ONLY the audited context.
4. **Egress Guard (The Auditor):** Performs a "Faithfulness" check (Answer ⊆ Context). If it fails, it triggers a retry loop with specific correction notes.

## 📋 Technical Specifications

### 📦 Pipeline State Schema
The `PipelineState` object tracks the lifecycle of a request:
- `userId`, `rawQuery`, `sanitizedQuery`
- `retrievedDocs` (id, text, score, metadata)
- `filteredDocs` (final set used for generation)
- `generationAttempts` (count of retries)
- `currentAnswer`
- `auditLog` (Step, Status, Reason, Timestamp)
- `isFaithful` (boolean)
- `correctionNotes` (feedback from Auditor to Generator)
- `finalVerdict` (APPROVED | REJECTED | ERROR)

### 🤖 Agent Personas
- **The Sentry:** Security perimeter agent. Focuses on safety and scope.
- **The Librarian:** Context auditor. Focuses on precision and relevance.
- **The Generator:** Precise technical writer. Focuses on grounding and conciseness.
- **The Auditor:** Fact-checker. Focuses on mathematical faithfulness.

## 🛠️ Implementation Roadmap (Task List)
- [x] Project Architecture Blueprint (Completed)
- [x] Pipeline State Schema & Agent Personas (Completed)
- [ ] Design Data Ingestion Pipeline (Pending)
- [ ] Architect Cyclic Workflow Logic (Pending)
- [ ] Establish Evaluation Framework (Pending)

## 📂 Artifacts Created
- **SentinelRAG: Enterprise Blueprint**: High-level architecture and visual flow.
- **SentinelRAG: Pipeline Specs**: Technical state schema and persona prompts.

## 🚀 Next Steps
1. Define the document ingestion strategy (parsing -> semantic chunking -> embedding).
2. Map the graph transitions in LangGraph/CrewAI to implement the Auditor -> Generator retry loop.
3. Setup the evaluation dataset (Ground Truth Q&A pairs) to measure Faithfulness and Relevance.
