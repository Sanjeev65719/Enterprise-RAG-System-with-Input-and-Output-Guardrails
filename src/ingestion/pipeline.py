import logging
import hashlib
from typing import List, Dict, Any
from pathlib import Path

from src.core.logger import logger
from src.ingestion.parsers.impl import MimeDetector, UnstructuredParser
from src.ingestion.normalizer import DocumentNormalizer
from src.ingestion.chunking.semantic import SemanticChunker, SemanticChunk
from src.ingestion.indexing.store import SentinelVectorStore, BatchIngestor
from src.ingestion.indexing.schema import DocumentMetadata
from uuid import uuid4

class IngestionPipeline:
    """
    The high-assurance orchestrator for the SentinelRAG ingestion process.
    Implements the flow: Parse -> Normalize -> Chunk -> Embed -> Index.
    """

    def __init__(self, openai_api_key: str, qdrant_host: str = "localhost"):
        self.parser = UnstructuredParser()
        self.normalizer = DocumentNormalizer()
        self.chunker = SemanticChunker()
        self.vector_store = SentinelVectorStore(host=qdrant_host)
        self.ingestor = BatchIngestor(self.vector_store, openai_api_key)

    def _calculate_hash(self, text: str) -> str:
        """Calculates SHA-256 hash for duplicate detection."""
        return hashlib.sha256(text.encode()).hexdigest()

    def process_file(self, file_path: str, org_id: str, user_id: str, acl_tags: List[str] = None) -> bool:
        """
        Processes a single file through the high-assurance pipeline.
        """
        try:
            logger.info(f"Starting high-assurance ingestion for: {file_path}")

            # 1. MIME Detection & Parsing
            mime_type = MimeDetector.detect(file_path)
            logger.info(f"Detected MIME type: {mime_type}")
            elements = self.parser.parse(file_path)

            # 2. Normalization
            normalized_text = self.normalizer.normalize(elements)

            # 3. Semantic Chunking
            # Initial metadata for the document
            base_metadata = {
                "source": file_path,
                "filename": Path(file_path).name,
            }
            semantic_chunks = self.chunker.chunk(normalized_text, base_metadata)

            # 4. Indexing with Data Boundaries
            # We prepare the metadata for the VectorPayload
            doc_metadata = DocumentMetadata(
                org_id=org_id,
                user_id=user_id,
                acl_tags=acl_tags or [],
                source_url=file_path,
                # content_hash is handled per chunk in a real scenario,
                # but we provide a doc-level hash here for simplicity
                content_hash=self._calculate_hash(normalized_text)
            )

            chunk_texts = [chunk.content for chunk in semantic_chunks]
            self.ingestor.ingest_chunks(chunk_texts, doc_metadata)

            # 5. Verification Gate 1: Integrity Check
            # Simplified: Ensure we actually created chunks
            if not chunk_texts:
                logger.warning(f"Ingestion produced no chunks for {file_path}")
                return False

            logger.info(f"Successfully ingested {len(chunk_texts)} chunks from {file_path}")
            return True

        except Exception as e:
            logger.error(f"Pipeline failed for {file_path}: {str(e)}", exc_info=True)
            return False

if __name__ == "__main__":
    # Example Usage
    import os

    API_KEY = os.getenv("OPENAI_API_KEY", "sk-...")
    pipeline = IngestionPipeline(openai_api_key=API_KEY)

    # Test with a dummy file
    with open("test_doc.md", "w") as f:
        f.write("# SentinelRAG Security Policy\n\nThis is a confidential document.\n\n## Section 1\nData boundaries are strict.")

    success = pipeline.process_file(
        file_path="test_doc.md",
        org_id="550e8400-e29b-41d4-a716-446655440000", # Example UUID
        user_id="a1b2c3d4-e5f6-g7h8-i9j0-k1l2m3n4o5p6",
        acl_tags=["CONFIDENTIAL", "SECURITY"]
    )
    print(f"Ingestion successful: {success}")
