import os
import json
import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from .embeddings import EmbeddingManager

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """
    Manages ChromaDB vector storage and document ingestion for 3GPP and Telecom knowledge bases.
    """
    def __init__(self, persist_directory: Optional[str] = None):
        if persist_directory is None:
            # Default to data/chroma_db directory in workspace root
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            persist_directory = os.path.join(base_dir, "data", "chroma_db")
        
        os.makedirs(persist_directory, exist_ok=True)
        self.persist_directory = persist_directory
        
        logger.info(f"Initializing ChromaDB persistent client at {self.persist_directory}")
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.embedding_manager = EmbeddingManager()

    def get_or_create_collection(self, collection_name: str = "telecom_specs"):
        """Retrieve or create a ChromaDB collection."""
        return self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        ChromaDB metadata requires primitive values (str, int, float, bool).
        Convert lists/dicts into JSON strings or flattened representations.
        """
        clean_meta = {}
        for key, val in metadata.items():
            if isinstance(val, (str, int, float, bool)):
                clean_meta[key] = val
            elif isinstance(val, list):
                clean_meta[key] = " > ".join(str(item) for item in val)
            elif isinstance(val, dict):
                clean_meta[key] = json.dumps(val)
            elif val is None:
                clean_meta[key] = ""
            else:
                clean_meta[key] = str(val)
        return clean_meta

    def ingest_vector_chunks(self, json_path: str, collection_name: str = "telecom_specs", batch_size: int = 128) -> int:
        """
        Ingest chunks from vector_chunks.json into ChromaDB.
        
        Each item in json_path is expected to be:
        {
           "id": "3_0",
           "content": "...",
           "metadata": { "spec": "...", "section_id": "...", ... }
        }
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"vector_chunks file not found at: {json_path}")

        logger.info(f"Loading chunks from {json_path}...")
        with open(json_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        logger.info(f"Total chunks loaded: {len(chunks)}. Starting ingestion into '{collection_name}' collection...")
        collection = self.get_or_create_collection(collection_name)

        total_ingested = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            
            ids = [str(item.get("id", f"chunk_{i+j}")) for j, item in enumerate(batch)]
            contents = [item.get("content", "") for item in batch]
            metadatas = [self.sanitize_metadata(item.get("metadata", {})) for item in batch]
            
            # Generate embeddings
            embeddings = self.embedding_manager.get_embeddings_batch(contents, batch_size=batch_size)
            
            # Upsert into ChromaDB
            collection.upsert(
                ids=ids,
                documents=contents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            total_ingested += len(batch)
            if total_ingested % 1024 == 0 or total_ingested == len(chunks):
                logger.info(f"Ingested {total_ingested}/{len(chunks)} chunks into ChromaDB...")

        logger.info(f"Successfully finished ingestion of {total_ingested} chunks.")
        return total_ingested

    def similarity_search(
        self,
        query: str,
        collection_name: str = "telecom_specs",
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute vector similarity search in ChromaDB.
        """
        collection = self.get_or_create_collection(collection_name)
        query_embedding = self.embedding_manager.get_embedding(query)

        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"]
        }
        if filters:
            query_kwargs["where"] = filters

        results = collection.query(**query_kwargs)

        search_results = []
        if results and results["ids"] and results["ids"][0]:
            ids = results["ids"][0]
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0]

            for chunk_id, doc, meta, dist in zip(ids, docs, metas, distances):
                # Cosine distance to similarity score
                similarity_score = 1.0 - max(0.0, dist)
                search_results.append({
                    "id": chunk_id,
                    "content": doc,
                    "metadata": meta,
                    "score": round(similarity_score, 4),
                    "search_type": "vector"
                })

        return search_results

    def get_collection_count(self, collection_name: str = "telecom_specs") -> int:
        """Return total document count in a collection."""
        collection = self.get_or_create_collection(collection_name)
        return collection.count()
