import logging
import yaml
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

# Resolve config path relative to this file, not the CWD
_DEFAULT_CONFIG_PATH = str(Path(__file__).resolve().parent.parent.parent / "config" / "config.yaml")

from src.core.logger import logger
from src.ingestion.indexing.store import SentinelVectorStore

@dataclass
class RetrievedDocument:
    content: str
    score: float
    metadata: Dict[str, Any]

class RAGEngine:
    """Core RAG logic: Query optimization, Retrieval, and Re-ranking."""

    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        # Using the real SentinelVectorStore implemented in the ingestion phase
        self.vector_db = SentinelVectorStore()
        self.top_k = self.config['system']['rag']['top_k']

    def _load_config(self, path: str = None) -> Dict:
        if path is None:
            path = _DEFAULT_CONFIG_PATH
        if os.path.exists(path):
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        logger.warning(f"Config file not found at '{path}'. Falling back to default configuration.")
        return {
            'system': {
                'logging': {
                    'level': 'INFO',
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                },
                'rag': {
                    'chunk_size': 512,
                    'chunk_overlap': 50,
                    'top_k': 5,
                    'rerank_top_n': 3
                },
                'guardrails': {
                    'input': {
                        'pii_masking': 'enabled',
                        'safety_filter': 'enabled',
                        'domain_check': 'enabled'
                    },
                    'output': {
                        'hallucination_check': 'enabled',
                        'relevance_check': 'enabled',
                        'compliance_check': 'enabled',
                        'max_hallucination_threshold': 0.2
                    }
                }
            }
        }

    def retrieve(self, query: str, org_id: str, user_id: str, tags: List[str] = None) -> List[RetrievedDocument]:
        """Performs hybrid retrieval with strict multi-tenant boundaries."""
        logger.info(f"Retrieving context for query: {query} (Org: {org_id})")

        # 1. Convert query to vector using OpenAI
        # For simplicity, we'll use the same logic as the Ingestor but as a helper
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        response = client.embeddings.create(
            input=[query],
            model="text-embedding-3-large"
        )
        query_vector = response.data[0].embedding

        # 2. Perform Filtered Search in Qdrant
        results = self.vector_db.filtered_search(
            query_vector=query_vector,
            org_id=org_id,
            user_id=user_id,
            tags=tags,
            limit=self.top_k
        )

        # 3. Convert Qdrant results to RetrievedDocument format
        retrieved_docs = []
        for res in results:
            retrieved_docs.append(RetrievedDocument(
                content=res.payload['text'],
                score=res.score,
                metadata=res.payload['metadata']
            ))

        return retrieved_docs

    def rerank(self, query: str, documents: List[RetrievedDocument]) -> List[RetrievedDocument]:
        """Re-ranks documents using a Cross-Encoder for higher precision."""
        logger.info("Re-ranking documents...")
        # Mocking re-ranking logic - in production we would use Cohere Rerank
        sorted_docs = sorted(documents, key=lambda x: x.score, reverse=True)
        n = self.config['system']['rag']['rerank_top_n']
        return sorted_docs[:n]

    def get_structured_context(self, query: str, org_id: str, user_id: str, tags: List[str] = None) -> List[RetrievedDocument]:
        """Full pipeline: Retrieve -> Re-rank -> Return structured documents."""
        docs = self.retrieve(query, org_id, user_id, tags)
        return self.rerank(query, docs)

    def get_context(self, query: str, org_id: str, user_id: str, tags: List[str] = None) -> str:
        """Full pipeline: Retrieve -> Re-rank -> Format as context string."""
        reranked_docs = self.get_structured_context(query, org_id, user_id, tags)
        context = "\n\n".join([f"Source: {d.metadata['source']}\nContent: {d.content}" for d in reranked_docs])
        return context
