import os
import logging
from typing import List, Dict, Any, Optional
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
from qdrant_client import QdrantClient
from qdrant_client.http import models
from src.ingestion.indexing.schema import VectorPayload, DocumentMetadata
from src.core.embedder import get_embedder
from src.core.logger import logger

class SentinelVectorStore:
    """
    High-assurance wrapper for Qdrant vector database.
    Enforces strict multi-tenant boundaries via payload filtering.
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        vector_size: Optional[int] = None
    ):
        url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")

        if url and api_key:
            logger.info(f"Connecting to Qdrant Cloud at {url}...")
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            q_host = host or os.getenv("QDRANT_HOST", "localhost")
            q_port = port or int(os.getenv("QDRANT_PORT", 6333))
            logger.info(f"Connecting to Qdrant Local at {q_host}:{q_port}...")
            self.client = QdrantClient(host=q_host, port=q_port)

        embedder = get_embedder()
        self.vector_size = vector_size or embedder.dimension
        self.collection_name = collection_name or os.getenv("QDRANT_COLLECTION", "sentinel_rag")
        self._ensure_collection()

    def _ensure_collection(self):
        """Initializes the collection with the correct vector configuration."""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)

            if exists:
                try:
                    info = self.client.get_collection(self.collection_name)
                    existing_size = getattr(info.config.params.vectors, 'size', None)
                    if existing_size and existing_size != self.vector_size:
                        logger.warning(
                            f"Collection '{self.collection_name}' has vector dimension {existing_size}, "
                            f"but current embedder produces {self.vector_size}. Switching collection name to '{self.collection_name}_{self.vector_size}'."
                        )
                        self.collection_name = f"{self.collection_name}_{self.vector_size}"
                        self._ensure_collection()
                        return
                except Exception as e:
                    logger.warning(f"Could not inspect existing collection config: {e}")

            if not exists:
                logger.info(f"Creating collection '{self.collection_name}' with vector size {self.vector_size}...")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(size=self.vector_size, distance=models.Distance.COSINE),
                )
        except Exception as e:
            logger.warning(
                f"Could not connect to Qdrant or initialize collection '{self.collection_name}': {e}. "
                "System will start, but vector searches will be unavailable until Qdrant is reachable."
            )

    def upsert(self, vector: List[float], payload: VectorPayload):
        """Inserts or updates a vector with its associated payload."""
        try:
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
        except Exception as e:
            logger.warning(f"Could not upsert vector into collection '{self.collection_name}': {e}")

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

    def __init__(self, vector_store: SentinelVectorStore, openai_api_key: Optional[str] = None):
        self.store = vector_store
        self.embedder = get_embedder()

    def ingest_chunks(self, chunks: List[str], metadata: DocumentMetadata):
        """
        Embeds and indexes a list of chunks in batches using the configured embedder.
        """
        if not chunks:
            return

        logger.info(f"Indexing {len(chunks)} chunks using {self.embedder.provider} ({self.embedder.model_name})...")

        # Batch embed using local FastEmbed (or OpenAI if configured)
        embeddings = self.embedder.embed_documents(chunks)

        # Upsert to Vector Store
        for i, vector in enumerate(embeddings):
            payload = VectorPayload(
                text=chunks[i],
                metadata=metadata
            )
            self.store.upsert(vector, payload)

        logger.info("Batch ingestion complete.")
