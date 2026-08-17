# Anomaly Detection Pipeline

The `anomaly/` module is responsible for analyzing raw streaming or static KPI arrays from O-RAN nodes (e.g., cell throughput, HARQ error rates, PRB allocations) to autonomously flag network degradation events.

## 1. Isolation & Detection (`detectors.py` & `window_detector.py`)
- **Isolation Forest**: We use `scikit-learn`'s Isolation Forest algorithm to detect multi-dimensional outliers across the dataset.
- **PELT (Pruned Exact Linear Time)**: We run a Change Point Detection algorithm from the `ruptures` library to pinpoint the precise timestamp boundary where the signal transitioned from "healthy" to "degraded".
- **Z-Score Normalization**: Used as an intensity metric to gauge the severity of the fault pattern.

## 2. Signal-to-Language Bridge (`anomaly_to_prompt.py`)
Once a specific time-window is isolated by the detectors, the metrics are extracted and passed through the bridge. The bridge mathematically calculates the % change of each KPI and formats it into a highly structured, plain-text "Anomaly Report" (e.g. `WIN_033`). 

This text report contains a synthesized **Diagnostic Question** which serves as the input query for the RAG knowledge retriever.

## 3. Storage (`data/anomaly_prompts.json`)
The generated reports are serialized into a JSON database, serving as the immutable ground-truth fault catalog that the Agno agent reads from when triggered via the `fetch_anomaly_window` tool.
