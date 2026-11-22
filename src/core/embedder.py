import os
from typing import List, Optional
from src.core.logger import logger

# Try loading .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_embedder_instance = None

# Known dimensions for common embedding models
MODEL_DIMENSIONS = {
    "BAAI/bge-small-en-v1.5": 384,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "nomic-ai/nomic-embed-text-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "BAAI/bge-m3": 1024,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}


class FastEmbedder:
    """
    Lightweight, 100% free, local embedding generator using FastEmbed (ONNX).
    Defaults to 'BAAI/bge-small-en-v1.5' (384 dimensions) with zero external API calls.
    Supports fallback to OpenAI if EMBEDDING_PROVIDER=openai is explicitly set.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.provider = os.getenv("EMBEDDING_PROVIDER", "fastembed").strip().lower()
        if self.provider == "openai":
            self.model_name = model_name or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
        else:
            self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

        self.dimension = MODEL_DIMENSIONS.get(self.model_name, 384)
        self._model = None
        self._openai_client = None

    def _init_model(self):
        if self.provider == "openai":
            if self._openai_client is None:
                from openai import OpenAI
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY must be set when EMBEDDING_PROVIDER=openai")
                self._openai_client = OpenAI(api_key=api_key)
        else:
            if self._model is None:
                try:
                    from fastembed import TextEmbedding
                    logger.info(f"Initializing FastEmbed local model '{self.model_name}' (dim={self.dimension})...")
                    self._model = TextEmbedding(model_name=self.model_name)
                    logger.info("FastEmbed model initialized successfully.")
                except Exception as e:
                    logger.error(f"Failed to initialize FastEmbed: {e}")
                    raise

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Generates embeddings for a batch of documents/chunks."""
        if not documents:
            return []

        self._init_model()

        if self.provider == "openai":
            response = self._openai_client.embeddings.create(
                input=documents,
                model=self.model_name
            )
            return [data.embedding for data in response.data]
        else:
            # fastembed returns generator of numpy arrays
            embeddings = list(self._model.embed(documents))
            return [emb.tolist() if hasattr(emb, "tolist") else list(emb) for emb in embeddings]

    def embed_query(self, query: str) -> List[float]:
        """Generates embedding vector for a single search query."""
        results = self.embed_documents([query])
        return results[0] if results else []


def get_embedder() -> FastEmbedder:
    """Returns the singleton embedder instance."""
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = FastEmbedder()
    return _embedder_instance
