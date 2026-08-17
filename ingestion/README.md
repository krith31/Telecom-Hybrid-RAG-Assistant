# Telecom RAG Preprocessing & Chunking Pipeline

This pipeline is optimized for 3GPP Technical Specifications (e.g., TS 23.501). It extracts hierarchical section structures, handles acronyms, cleans non-retrieval content, and creates chunks with context-rich breadcrumbs.

## Prerequisites

Install dependencies:
```bash
pip install -r requirements.txt
```

## Workflow

### 1. Preprocessing
Run `preprocess.py` to convert raw `.docx` files into structured JSON.
- **Extracts Abbreviations**: Builds a lookup from Section 3.2.
- **Cleans Content**: Removes figure placeholders, change history, and void clauses.
- **Identifies Hierarchy**: Maps paragraphs to their section IDs (e.g., 5.34.6.2).
- **Table Handling**: Detects and extracts tables.

```bash
python scripts/preprocess.py
```

### 2. Chunking
Run `chunker.py` to generate chunks for your Vector Database.
- **Breadcrumb Injection**: Prepends `[Spec | Section Path]` to every chunk.
- **Whitespace Normalization**: Collapses excessive whitespace and removes tab characters (`\t`) to ensure clean embeddings, while keeping newlines (`\n`) to preserve paragraph and table structure.
- **Acronym Metadata**: Identifies acronyms present in each chunk and adds their definitions to metadata.
- **Table Normalization**: Formats tables for better retrieval performance.

```bash
python scripts/chunker.py
```

### 3. Embedding and Storing
Run `embedder.py` to generate embeddings using the BAAI/bge-large-en-v1.5 model and store them into a ChromaDB vector database.
- **Top-Performing Embeddings**: Uses `BAAI/bge-large-en-v1.5` which is one of the leading open-source models for text retrieval and semantic similarity.
- **ChromaDB**: Persistently stores the document embeddings and metadata for highly efficient Retrieval-Augmented Generation (RAG).
- **Metadata Preservation**: Stores acronyms, section titles, and hierarchy natively inside the Vector DB for structured filtering.

```bash
python scripts/embedder.py
```

## Data Directory Structure
- `Data/specs/`: Put your raw `.docx` files here.
- `Data/processed/`: Intermediate structured JSON files.
- `Data/chunks/`: Final `vector_chunks.json` ready for embedding.
- `Data/chroma_db/`: Persistent local ChromaDB database.
