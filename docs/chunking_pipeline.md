# Ingestion & Chunking Pipeline

The `ingestion/` module is responsible for parsing raw 3GPP standards (PDFs or `.txt`), splitting them into mathematically manageable chunks, and ingesting them into the vector database.

## 1. Preprocessing (`preprocess.py`)
Cleans raw text to remove useless OCR artifacts, standardizes 3GPP clause headers (e.g. `5.3.2.1`), and enforces structural consistency.

## 2. Chunking (`chunker.py`)
Uses `RecursiveCharacterTextSplitter` from LangChain to break large documents into chunks of roughly 500-800 words. It preserves structural integrity by refusing to split in the middle of a 3GPP clause block, ensuring that context remains strictly grouped by logical sections.

## 3. Embedding (`embedder.py`)
Vectorizes the chunks using `BAAI/bge-large-en-v1.5` (a 1024-dimensional model) and inserts them into the ChromaDB persistent store located in `data/chroma_db/`.
