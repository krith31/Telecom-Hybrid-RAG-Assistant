# Telecom RAG: Data Processing & Pipeline Documentation

This document explains the data architecture of the Telecom RAG application. It covers what the data is, how it is structured, the transformation pipeline from numerical signals to Large Language Model (LLM) prompts, and the complete text processing pipeline from raw documents into the ChromaDB vector database.

---

## 1. What is the Data and How Does It Look?

The system ingests three primary types of data:

### A. 3GPP Technical Specifications (Knowledge Base)
- **What it is**: Official telecom standard documents (e.g., TS 23.501).
- **How it looks initially**: Raw Microsoft Word (`.docx`) files. These documents are highly structured, featuring deep numerical hierarchies (e.g., `5.34.6.2`), embedded tables, "Void" clauses, Figure placeholders, and specific abbreviation sections (typically Section 3.2).

### B. TeleQnA Dataset (Evaluation & QA Data)
- **What it is**: A benchmark dataset containing ~10,000 multiple-choice questions for telecom.
- **How it looks**: A large JSON-like structure mapping question IDs to question details, options, the correct answer, an explanation, and a category.
```json
{
    "question 0":  {
        "question":  "What is the purpose of the Nmfaf_3daDataManagement_Deconfigure service operation?",
        "option 1":  "To configure the MFAF...",
        "answer":  "option 2: To configure the MFAF to stop mapping...",
        "explanation":  "The Nmfaf_3daDataManagement_Deconfigure service operation is used to stop...",
        "category":  "Standards specifications"
    }
}
```

### C. SpotLight Anomaly KPI Datasets (Signal Data)
- **What it is**: Static `.csv` datasets containing 5G network Key Performance Indicators (KPIs) from the SpotLight dataset. 
- **How it looks**: Tabular numerical data organized into domain-specific folders (RADIO, MAC, NETWORK, PDCP, MIXED). Each folder contains a `radio.csv` and `platform.csv` with fields like `timestamp`, `mac_sinr_update_mean`, `mac_dl_harq_nack_rate`, and `l1_avg_mcs`.

---

## 2. Signal Processing: From KPI Datasets to "Prompt Conversion"

The anomaly detection pipeline (`run_pipeline.py`) processes the static SpotLight datasets step-by-step to identify faults and translate them into RAG prompts.

### Step 1: Merging Datasets (`merge_kpis.py`)
- The pipeline scans the `Anomaly Dataset` directory for folders containing both `radio.csv` and `platform.csv`.
- It loads both files, filters for selected KPIs, standardizes timestamps, and merges them using an "as-of" merge (`pd.merge_asof`) with a 500ms tolerance to align the radio and platform metrics chronologically.
- It determines the specific domain (RADIO, MAC, PDCP, etc.) based on the folder path.

### Step 2: Cleaning & Normalization (`clean_dataset.py`)
- Converts `inf` and `-inf` values to `NaN`.
- Applies linear interpolation to fill missing values over time. If an entire column is empty, it falls back to the median or zero.
- Identifies and drops "invariant" columns (columns where the value never changes).
- Ensures all feature columns are strictly numeric.

### Step 3: Ensemble Anomaly Detection (`detectors.py`)
The detection is run **independently per domain** to prevent metrics from one domain skewing the baseline of another. The pipeline uses two parallel algorithms:
1. **Isolation Forest**: Trained only on the "normal" (non-anomalous) rows of the specific domain to detect multi-dimensional outliers.
2. **Rolling Z-Score**: Computes a rolling mean and standard deviation (window=30) for each KPI time-series to detect sudden localized spikes or drops (threshold=3.0).

An ensemble flag is triggered if **either** detector flags the row as anomalous.

### Step 4: Window Detection (`window_detector.py`)
- Instead of treating every row individually, the system groups contiguous anomalous rows into distinct "Anomaly Windows" (e.g., `WIN_001`).
- A new window is formed if there is a time gap > 2 seconds or if the domain changes. Micro-windows (< 3 rows) are discarded.
- It calculates the `peak_z_score`, the primary affected KPI, and the percentage deviation (`pct_change`) of each KPI compared to the domain's normal baseline.
- **Idle vs Fault handling**: It recognizes that if `mac_ul_CRC_Loss` or `mac_dl_harq_total` drops to 0, it means the link is idle or the UE detached, rather than an active performance fault.

### Step 5: The Signal-to-Language Bridge (`anomaly_to_prompt.py`)
This step converts the mathematical deviations of an anomaly window into a natural language prompt for the LLM.
- **Pattern Classification**: It evaluates the KPI deviations against hardcoded telecom rules. For example, if MCS drops by >50% and CRC errors spike by >200%, it classifies the pattern as `"Radio Channel Degradation / Interference"`. If PRB allocation drops by >40% and MCS drops by >20%, it classifies it as `"PHY/MAC Scheduler Resource Starvation"`.
- **Query Generation**: It uses templates to generate a structured English query that summarizes the observed faults, calculates the percentage changes from baseline, and poses a diagnostic question.

**Final Prompt Output:**
> *"A 5G NR gNB cell experienced a 'Uplink Frame Integrity Failure (CRC Errors)' lasting 3.1 seconds. The following active fault indicators were observed: uplink CRC frame error rate spiked by 1948%... What is the most likely sequence of events that produced this pattern? What are the specific root causes for the MCS fallback and PRB starvation, and what recovery procedures should the operator execute to restore normal cell performance?"*

---

## 3. The Full Chunking Process: Raw Docs to ChromaDB

To enable the LLM to answer telecom queries accurately, the raw 3GPP `.docx` specifications undergo a rigorous preprocessing, chunking, and embedding pipeline.

### Step 1: Preprocessing (`preprocess.py`)
Raw Word documents are parsed into structured JSON to preserve context.
1. **Abbreviation Extraction**: Scans Section 3.2 for telecom acronyms (e.g., `AMF -> Access and Mobility Management Function`) and stores them in a lookup dictionary.
2. **Noise Cleaning**: Strips out non-informational text such as "Figure X", "Void" clauses, and "Editor's notes". Non-breaking spaces are normalized.
3. **Hierarchical Parsing**: Matches paragraphs to 3GPP heading formats (e.g., `5.1.2`) using RegEx. It builds a "breadcrumb stack" so every piece of text knows its parent sections.
4. **Table Handling**: Tables are extracted row by row and saved as structured objects.

### Step 2: The Chunking Process in Detail (`chunker.py`)
The processed JSON is split into smaller pieces (chunks) optimized for vector search.

**Criteria & Splitting Logic**:
- Uses LangChain's `RecursiveCharacterTextSplitter`.
- **Chunk Size**: 600 tokens. Tokens are measured using the `tiktoken` library with the `cl100k_base` encoding (standard for GPT-3.5/GPT-4), ensuring chunks fit perfectly into LLM context windows.
- **Chunk Overlap**: 120 tokens. This overlap prevents sentences or thoughts from being abruptly cut off across chunk boundaries.
- **Separators**: Splitter uses the hierarchy `["\n\n", "\n", ". ", " ", ""]`, meaning it will try to split by paragraphs first, then by single newlines, then by sentences, preserving structural integrity.

**What Each Chunk Contains**:
1. **Breadcrumb Header**: To ensure the chunk doesn't lose its global context, the system prepends a breadcrumb string: `[Spec ID | Path > To > Section Title] \n\n`.
2. **Text Content**: The actual prose of the document section. Consecutive spaces and tabs are normalized to a single space to improve embedding quality.
3. **Normalized Tables**: Small tables are converted into Markdown-like string representations (`row1_col1 | row1_col2`) and injected directly into the chunk text block.
4. **Acronym Metadata**: A custom function scans the chunk's specific text for uppercase acronyms. It checks these against the abbreviation dictionary extracted in preprocessing. Any matched acronyms are expanded and attached directly to the chunk's metadata.
5. **Metadata Payload**: Every chunk includes a JSON metadata object containing:
   - `spec`: The specification document ID.
   - `section_id`: The specific 3GPP section number (e.g., 5.34.6.2).
   - `title`: The section title.
   - `path`: The array of parent section titles.
   - `acronyms`: A dictionary of acronyms and their full definitions.
   - `chunk_index`: The sequence number of the chunk.

**Output**: A massive list of unified chunks saved to `vector_chunks.json`.

### Step 3: Embedding & Storage (`embedder.py`)
1. **Model Initialization**: Loads `BAAI/bge-large-en-v1.5`, a highly accurate open-source sentence transformer model.
2. **Vectorization**: Processes the chunks in batches of 100, converting the text into floating-point vector embeddings.
3. **Database Upsert**: Connects to `ChromaDB`.
4. **Metadata Filtering**: Stores the chunks alongside ChromaDB-safe metadata (dictionaries/lists are stringified if necessary).
5. **Final State**: The `telecom_specs` collection in ChromaDB is indexed using cosine similarity space, ready for retrieval by the RAG router.
