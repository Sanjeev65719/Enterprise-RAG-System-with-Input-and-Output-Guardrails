import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from openai import OpenAI
from src.ingestion.indexing.schema import VectorPayload, DocumentMetadata

from src.core.logger import logger

class SentinelVectorStore:
    """
    High-assurance wrapper for Qdrant vector database.
    Enforces strict multi-tenant boundaries via payload filtering.
    """

    def __init__(self, host: str = "localhost", port: int = 6333, collection_name: str = "sentinel_rag"):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self._ensure_collection()

    def _ensure_collection(self):
        """Initializes the collection with the correct vector configuration."""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if not exists:
            logger.info(f"Creating collection {self.collection_name}...")
            # Using 3072 dimensions for text-embedding-3-large
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=3072, distance=models.Distance.COSINE),
            )

    def upsert(self, vector: List[float], payload: VectorPayload):
        """Inserts or updates a vector with its associated payload."""
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=str(payload.metadata.doc_id), # Simplified; usually a unique chunk ID
                    vector=vector,
                    payload=payload.model_dump()
                )
            ]
        )

    def filtered_search(self, query_vector: List[float], org_id: str, user_id: str, tags: List[str] = None, limit: int = 5):
        """
        Performs a semantic search with strict multi-tenant filters.
        """
        # Build the boundary filter
        # Must match org_id AND (be owned by user_id OR match one of the acl_tags)
        filter_condition = models.Filter(
            must=[
                models.FieldCondition(key="metadata.org_id", match=models.MatchValue(value=org_id)),
                models.Filter(
                    should=[
                        models.FieldCondition(key="metadata.user_id", match=models.MatchValue(value=user_id)),
                        models.FieldCondition(key="metadata.acl_tags", match=models.MatchAny(any=tags or [])),
                    ]
                )
            ]
        )

        return self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=filter_condition,
            limit=limit
        )

class BatchIngestor:
    """Handles efficient batch embedding and indexing."""

    def __init__(self, vector_store: SentinelVectorStore, openai_api_key: str):
        self.store = vector_store
        self.openai_client = OpenAI(api_key=openai_api_key)

    def ingest_chunks(self, chunks: List[str], metadata: DocumentMetadata):
        """
        Embeds and indexes a list of chunks in batches.
        """
        logger.info(f"Indexing {len(chunks)} chunks...")

        # Batch embed using OpenAI
        response = self.openai_client.embeddings.create(
            input=chunks,
            model="text-embedding-3-large"
        )
        embeddings = [data.embedding for data in response.data]

        # Upsert to Vector Store
        for i, vector in enumerate(embeddings):
            payload = VectorPayload(
                text=chunks[i],
                metadata=metadata
            )
            self.store.upsert(vector, payload)

        logger.info("Batch ingestion complete.")
