import logging
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from .vectorstore import VectorStoreManager

logger = logging.getLogger(__name__)

class HybridRetriever:
    """
    Hybrid Retrieval Engine combining ChromaDB Dense Vector Search with BM25 Sparse Keyword Search
    and Reciprocal Rank Fusion (RRF).
    """
    def __init__(self, vector_store: Optional[VectorStoreManager] = None):
        if vector_store is None:
            vector_store = VectorStoreManager()
        self.vector_store = vector_store

    def reciprocal_rank_fusion(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        rrf_k: int = 60,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Merge vector and BM25 search results using Reciprocal Rank Fusion (RRF).
        RRF Score = 1/(k + rank_vector) + 1/(k + rank_bm25)
        """
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict[str, Any]] = {}

        # Process vector results
        for rank, doc in enumerate(vector_results):
            doc_id = doc["id"]
            doc_map[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + (rank + 1)))

        # Process BM25 results
        for rank, doc in enumerate(bm25_results):
            doc_id = doc["id"]
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + (rank + 1)))

        # Sort by final RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        final_results = []
        for doc_id in sorted_ids[:top_k]:
            item = doc_map[doc_id].copy()
            item["rrf_score"] = round(rrf_scores[doc_id], 6)
            final_results.append(item)

        return final_results

    def retrieve(
        self,
        query: str,
        collection_name: str = "telecom_specs",
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid retrieval for a query.
        """
        logger.info(f"Executing hybrid retrieval for query: '{query[:50]}...' in collection '{collection_name}'")

        # 1. Vector Search
        vector_docs = self.vector_store.similarity_search(
            query=query,
            collection_name=collection_name,
            top_k=top_k * 2,
            filters=filters
        )

        # If no vector results, return empty
        if not vector_docs:
            return []

        # 2. BM25 Search over retrieved candidate pool
        corpus = [doc["content"] for doc in vector_docs]
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        
        if tokenized_corpus:
            bm25 = BM25Okapi(tokenized_corpus)
            tokenized_query = query.lower().split()
            bm25_scores = bm25.get_scores(tokenized_query)

            # Sort docs by BM25 score
            bm25_indexed = list(zip(vector_docs, bm25_scores))
            bm25_indexed.sort(key=lambda x: x[1], reverse=True)

            bm25_docs = []
            for doc, score in bm25_indexed:
                d = doc.copy()
                d["bm25_score"] = float(score)
                d["search_type"] = "bm25"
                bm25_docs.append(d)
        else:
            bm25_docs = vector_docs

        # 3. Combine with RRF
        hybrid_results = self.reciprocal_rank_fusion(
            vector_results=vector_docs,
            bm25_results=bm25_docs,
            top_k=top_k
        )

        return hybrid_results
