# RAG Architecture (Retrieval-Augmented Generation)

The `rag/` module is the knowledge retrieval engine of the application. It employs a multi-stage retrieval architecture to guarantee that highly specific telecom jargon is accurately mapped to 3GPP clauses.

## 1. Query Enhancer (`query_enhancer.py`)
Intercepts raw user queries (e.g. "UE RLF PRB") and explicitly expands telecom acronyms (e.g. "User Equipment Radio Link Failure Physical Resource Block") using a predefined telecom lexicon dictionary. This significantly increases dense retrieval matching probability.

## 2. Hybrid Retriever (`retriever.py`)
Executes a dual-path retrieval strategy:
- **Dense Vector Search**: Uses the BGE model to compute semantic cosine similarity against the ChromaDB vector store.
- **Sparse Keyword Search**: Uses BM25 to guarantee exact lexical matches for numbers, specific clause IDs (e.g. `TS 38.331`), and parameters.
- **Reciprocal Rank Fusion (RRF)**: Merges the results of both searches into a single mathematically sorted candidate list.

## 3. Cross-Encoder Reranker (`reranker.py`)
Takes the top 10 candidates from the Hybrid Retriever and scores them using `BAAI/bge-reranker-base`. Unlike bi-encoders, the cross-encoder feeds the query and chunk together into the transformer, capturing deep contextual nuance to select the definitive top 3 most relevant chunks.
