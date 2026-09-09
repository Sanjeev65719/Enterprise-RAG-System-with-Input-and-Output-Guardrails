# 🛡️ Enterprise RAG System with Guardrails

SentinelRAG is a high-assurance Enterprise RAG system designed to eliminate hallucinations and secure data boundaries. It implements a **Verified Loop** where a response is mathematically verified for faithfulness before being displayed.

## 🚀 Deployment

### Local Run
1. Install dependencies: `pip install -r requirements.txt`
2. Launch app: `streamlit run app.py`

### Cloud Deployment
1. Push this repo to GitHub.
2. Connect your repo to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Add your API keys in the **Secrets** menu.

## 🏗️ Architecture
The system follows a high-assurance pipeline:
- **The Sentry**: Ingress guard to block prompt injections and PII.
- **The Librarian**: Retrieval guard to ensure boundary-respecting context.
- **The Generator**: Core LLM synthesis engine.
- **The Auditor**: Egress guard to detect and block hallucinations.

## 🛠️ Tech Stack
- **UI**: Streamlit
- **Framework**: LangChain / Python 3.10
- **Vector DB**: Qdrant
- **LLM**: OpenAI (GPT-4o)
