import logging
from typing import List, Union
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """
    Singleton manager for generating vector embeddings using SentenceTransformers.
    Default model: BAAI/bge-large-en-v1.5 (1024 dimensions).
    """
    _instance = None
    _model = None

    def __new__(cls, model_name: str = "BAAI/bge-large-en-v1.5"):
        if cls._instance is None:
            cls._instance = super(EmbeddingManager, cls).__new__(cls)
            cls._instance.model_name = model_name
            cls._instance._load_model()
        return cls._instance

    def _load_model(self):
        """Lazy load the sentence transformer model into memory (on CPU to conserve GPU VRAM for LLM)."""
        logger.info(f"Loading embedding model: {self.model_name} (device: cpu)")
        try:
            self._model = SentenceTransformer(self.model_name, device="cpu")
            logger.info("Embedding model loaded successfully on CPU.")
        except Exception as e:
            logger.error(f"Failed to load embedding model {self.model_name}: {e}")
            raise e

    def get_embedding(self, text: str, is_query: bool = True) -> List[float]:
        """
        Generate a 1024-dimensional vector embedding for a single string.
        For BGE models, we prepend the query instruction if is_query is True.
        """
        if not text or not text.strip():
            return [0.0] * self.embedding_dimension
            
        if is_query and "bge" in self.model_name.lower():
            text = f"Represent this sentence for searching relevant passages: {text}"
            
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def get_embeddings_batch(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """
        Generate embeddings for a batch of strings (typically for document ingestion, so no query prefix).
        """
        if not texts:
            return []
        
        # Replace empty strings with single space to avoid encoding errors
        cleaned_texts = [t if t and t.strip() else " " for t in texts]
        embeddings = self._model.encode(
            cleaned_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings.tolist()

    @property
    def embedding_dimension(self) -> int:
        return 1024
