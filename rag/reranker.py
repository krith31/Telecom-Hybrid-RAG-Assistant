import logging
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    """
    Re-scores candidate chunks using CrossEncoder (BAAI/bge-reranker-base).
    Improves MRR and Top-K retrieval precision targets.
    """
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            logger.info(f"Loading Cross-Encoder model: {self.model_name} (device: cpu)")
            self.model = CrossEncoder(self.model_name, device="cpu")
            logger.info("Cross-Encoder model loaded successfully on CPU.")
        except Exception as e:
            logger.warning(f"Failed to load CrossEncoder ({e}). Will fallback to initial scores.")
            self.model = None

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Re-score and reorder retrieved documents by cross-encoder relevance score.
        """
        if not documents:
            return []

        if self.model is None:
            # Fallback to top_n from input docs
            return documents[:top_n]

        # Prepare pairs for Cross-Encoder
        pairs = [[query, doc.get("content", "")] for doc in documents]

        try:
            scores = self.model.predict(pairs)
            
            # Attach rerank score
            reranked_docs = []
            for doc, score in zip(documents, scores):
                d = doc.copy()
                d["rerank_score"] = float(score)
                reranked_docs.append(d)

            # Sort by rerank_score in descending order
            reranked_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
            return reranked_docs[:top_n]

        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            return documents[:top_n]
