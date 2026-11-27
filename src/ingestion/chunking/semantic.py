import re
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from src.core.embedder import get_embedder

@dataclass
class SemanticChunk:
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    start_index: int
    end_index: int

class SemanticChunker:
    """
    Implements high-assurance semantic chunking using the 'Similarity Gap' mechanism.
    Splits documents where the semantic meaning changes significantly.
    Uses FastEmbed (ONNX) and NumPy for lightweight, zero-dependency local execution.
    """

    def __init__(self, model_name: Optional[str] = None, threshold: float = 0.85):
        """
        Args:
            model_name: Optional embedding model name.
            threshold: Cosine similarity threshold. Below this, a break is triggered.
        """
        self._embedder = get_embedder()
        self.threshold = threshold

    def _get_sentences(self, text: str) -> List[str]:
        """Splits text into sentences cleanly using regex."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _calc_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Calculates cosine similarity between two vectors using NumPy."""
        a = np.array(v1, dtype=float)
        b = np.array(v2, dtype=float)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def chunk(self, text: str, base_metadata: Dict[str, Any]) -> List[SemanticChunk]:
        """
        Performs semantic chunking on the provided text.
        """
        sentences = self._get_sentences(text)
        if not sentences:
            return []

        # 1. Generate embeddings for all sentences using FastEmbed in memory-safe batches
        embeddings = []
        batch_size = 32
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            embeddings.extend(self._embedder.embed_documents(batch))

        # 2. Identify boundaries using cosine similarity
        chunks = []
        current_chunk_sentences = [sentences[0]]

        for i in range(len(sentences) - 1):
            sim = self._calc_similarity(embeddings[i], embeddings[i + 1])

            if sim < self.threshold:
                # Semantic break detected
                chunk_text = " ".join(current_chunk_sentences)
                chunks.append(self._create_chunk(
                    chunk_text,
                    base_metadata,
                    len(chunks),
                    0,
                    i
                ))
                current_chunk_sentences = [sentences[i + 1]]
            else:
                current_chunk_sentences.append(sentences[i + 1])

        # Add the final chunk
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append(self._create_chunk(
                chunk_text,
                base_metadata,
                len(chunks),
                0,
                len(sentences) - 1
            ))

        return chunks

    def _create_chunk(self, text: str, metadata: Dict[str, Any], index: int, start: int, end: int) -> SemanticChunk:
        return SemanticChunk(
            content=text,
            metadata=metadata.copy(),
            chunk_id=f"chunk_{index}",
            start_index=start,
            end_index=end
        )
