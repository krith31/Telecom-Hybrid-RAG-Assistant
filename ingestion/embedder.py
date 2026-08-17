import os
import json
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

class TelecomEmbedder:
    def __init__(self, chunks_path, db_dir, collection_name="telecom_specs"):
        self.chunks_path = chunks_path
        self.db_dir = db_dir
        
        print("Loading BGE model... (This will download model weights on first run)")
        # BAAI/bge-large-en-v1.5 is a highly accurate, standard sentence transformer model
        self.model = SentenceTransformer("BAAI/bge-large-en-v1.5")
        
        print(f"Initializing ChromaDB at {db_dir}")
        self.chroma_client = chromadb.PersistentClient(path=db_dir)
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"} # Cosine similarity is best for these embeddings
        )

    def process(self):
        print(f"Loading chunks from {self.chunks_path}")
        with open(self.chunks_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        # ChromaDB optimal batch size is around 100-500
        batch_size = 100
        
        for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding and Storing"):
            batch = chunks[i:i + batch_size]
            
            ids = [chunk["id"] for chunk in batch]
            documents = [chunk["content"] for chunk in batch]
            
            # Metadata must be strings, ints, floats, or bools for ChromaDB
            metadatas = []
            for chunk in batch:
                meta = chunk["metadata"]
                safe_meta = {
                    "spec": meta.get("spec", ""),
                    "section_id": meta.get("section_id", ""),
                    "title": meta.get("title", ""),
                    "path": " > ".join(meta.get("path", [])),
                    "chunk_index": meta.get("chunk_index", 0)
                }
                # Optional: stringify acronyms dictionary
                if "acronyms" in meta and meta["acronyms"]:
                    safe_meta["acronyms"] = json.dumps(meta["acronyms"])
                metadatas.append(safe_meta)
            
            # Generate embeddings
            embeddings = self.model.encode(documents, show_progress_bar=False)
            
            # Add to ChromaDB
            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings.tolist(),
                metadatas=metadatas
            )

        print(f"\nSuccessfully stored {len(chunks)} chunks in ChromaDB at {self.db_dir}")

if __name__ == "__main__":
    embedder = TelecomEmbedder(
        chunks_path=r"d:\Antigravity\TeleComRAG\Data\chunks\vector_chunks.json",
        db_dir=r"d:\Antigravity\TeleComRAG\Data\chroma_db"
    )
    embedder.process()
