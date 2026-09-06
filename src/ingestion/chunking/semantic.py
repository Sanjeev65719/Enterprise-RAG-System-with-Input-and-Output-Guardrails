import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure nltk data is available
nltk.download('punkt', quiet=True)

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
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", threshold: float = 0.85):
        """
        Args:
            model_name: Local embedding model for boundary detection.
            threshold: Cosine similarity threshold. Below this, a break is triggered.
        """
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold

    def _get_sentences(self, text: str) -> List[str]:
        """Splits text into sentences using NLTK."""
        return nltk.sent_tokenize(text)

    def chunk(self, text: str, base_metadata: Dict[str, Any]) -> List[SemanticChunk]:
        """
        Performs semantic chunking on the provided text.
        """
        sentences = self._get_sentences(text)
        if not sentences:
            return []

        # 1. Generate embeddings for all sentences
        embeddings = self.model.encode(sentences)

        # 2. Identify boundaries using cosine similarity
        chunks = []
        current_chunk_sentences = [sentences[0]]

        for i in range(len(sentences) - 1):
            # Calculate similarity between sentence i and i+1
            sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]

            if sim < self.threshold:
                # Semantic break detected
                chunk_text = " ".join(current_chunk_sentences)
                chunks.append(self._create_chunk(
                    chunk_text,
                    base_metadata,
                    len(chunks),
                    0, # start_index placeholder
                    i
                ))
                current_chunk_sentences = [sentences[i+1]]
            else:
                current_chunk_sentences.append(sentences[i+1])

        # Add the final chunk
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append(self._create_chunk(
                chunk_text,
                base_metadata,
                len(chunks),
                0, # start_index placeholder
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
