# Telecom Hybrid RAG Agent

This project implements an advanced Hybrid Agent architecture designed for 3GPP/O-RAN telecommunications diagnostics. It fuses **Retrieval-Augmented Generation (RAG)** over 3GPP specifications with automated **Network Anomaly Detection**.

## 🏗️ Architecture

The system consists of two major pipelines running behind a Gemini routing agent:

1. **Path A: Automated Anomaly Detection (`anomaly/`)**
   - Monitors raw O-RAN KPI streams (RSRP, SINR, Throughput, MCS, PRB).
   - Uses Isolation Forest and PELT (Change Point Detection) to isolate failures.
   - Extracts localized anomaly reports mapping time-windows to root causes.

2. **Path B: 3GPP RAG Knowledge Base (`ingestion/` & `rag/`)**
   - Ingests raw 3GPP standards (`data/specs/`).
   - Uses dense (BAAI/bge-large-en) and sparse (BM25) hybrid retrieval.
   - Cross-encoder reranking (bge-reranker-base) for ultra-precise clause matching.

3. **Hybrid Routing Agent (`main.py`)**
   - A lightweight `gemini-3.5-flash-lite` model acts as a router.
   - Extracts diagnostic questions from anomaly reports, pulls 3GPP context, and routes a fully constructed prompt to a local Fine-Tuned Mistral-7B expert model via GGUF endpoint.

## 🚀 Getting Started

### 1. Setup Environment
```bash
python -m venv TeleCom
TeleCom\Scripts\activate
pip install -r requirements.txt
```
Copy `.env.example` to `.env` and fill in your Gemini API key and Local Model URL.

### 2. Start the Fine-Tuned Model Server (Kaggle GPU)
The fine-tuned Mistral-7B model is served from a Kaggle GPU notebook, not locally. Before running the agent:

1. Open [`telecom-rag.ipynb`](https://www.kaggle.com/code/harshitanan/telecom-rag) in Kaggle with GPU + Internet enabled.
2. Run all cells — this downloads the GGUF model, starts an inference server, and launches a cloudflared tunnel that prints a public URL.
3. Add that URL to your `.env` as the Local Model URL: `LOCAL_MODEL_URL=https://<your-tunnel-url>/v1`

> [!IMPORTANT]
> Keep the notebook running while you use the agent — if the session ends, restart it and update the URL in `.env`.

### 3. Populate Knowledge Base
First, ensure your raw 3GPP specification documents (.docx) are placed inside the `data/specs/` folder.
Then, run the chunker and embedder to populate your ChromaDB vector store:
```bash
python ingestion/preprocess.py
python ingestion/chunker.py
python ingestion/embedder.py
```

### 4. Generate Anomaly Reports
First, download the raw KPI dataset from the [SpotLight Repository](https://github.com/netsys-edinburgh/SpotLight) and drop the `.zip` files into the `data/Anomaly Dataset/` folder.
Then, run the anomaly pipeline to process your KPI data and generate `anomaly_prompts.json`:
```bash
python anomaly/run_pipeline.py
```

### 5. Run the Agent
Once the data is populated, start the interactive CLI agent:
```bash
python main.py
```

## 📚 Documentation
- [Ingestion & Chunking Pipeline](docs/chunking_pipeline.md)
- [RAG Architecture](docs/rag_architecture.md)
- [Anomaly Detection](docs/anomaly_detection.md)
