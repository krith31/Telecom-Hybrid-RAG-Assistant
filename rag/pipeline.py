import logging
from typing import List, Dict, Any, Optional

from .query_enhancer import QueryEnhancer
from .retriever import HybridRetriever
from .reranker import CrossEncoderReranker
from .vectorstore import VectorStoreManager

logger = logging.getLogger(__name__)

class TelecomRAGPipeline:
    """
    Unified entrypoint for the deterministic Telecom RAG Pipeline.
    Encapsulates Enhancement, Hybrid Retrieval, and Cross-Encoder Reranking
    against a single central ChromaDB instance.
    """
    def __init__(self):
        # Initialize singletons / managers
        self.vector_store = VectorStoreManager()
        self.enhancer = QueryEnhancer()
        self.retriever = HybridRetriever(vector_store=self.vector_store)
        self.reranker = CrossEncoderReranker()

    def retrieve_context(self, query: str, top_k: int = 3, collection_name: str = "telecom_specs") -> List[Dict[str, Any]]:
        """
        Executes the full retrieval pipeline:
        1. Enhance query (expand acronyms)
        2. Hybrid search (Dense + Sparse BM25 + RRF)
        3. Rerank using CrossEncoder
        """
        logger.info(f"Pipeline processing query: '{query[:60]}...'")

        # 1. Query Enhancement
        enhanced = self.enhancer.enhance_query(query)
        search_query = enhanced.get("enhanced_query", query)

        # 2. Hybrid Retrieval
        candidate_chunks = self.retriever.retrieve(
            query=search_query,
            collection_name=collection_name,
            top_k=max(8, top_k * 2)  # retrieve more for reranking
        )

        if not candidate_chunks:
            logger.warning(f"No candidate chunks found for query in collection '{collection_name}'")
            return []

        # 3. Cross-Encoder Re-Ranking
        reranked_chunks = self.reranker.rerank(
            query=query, 
            documents=candidate_chunks, 
            top_n=top_k
        )

        logger.info(f"Pipeline successfully retrieved and reranked {len(reranked_chunks)} chunks.")
        return reranked_chunks
