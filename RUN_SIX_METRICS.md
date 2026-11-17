# Run the six metrics from the evaluation checklist

The new entry point is `python -m evals.complete`. It builds or reads your corpus, creates a labeled benchmark when one is not supplied, runs the actual retriever and selected generator, judges the resulting answers, and exports MRR, Top-k Accuracy, Accuracy, Recall, Relevance and Faithfulness.

This evaluates RAG retrieval and answer generation. It does not run or evaluate the anomaly detector or Gemini routing agent. Diagnostic queries extracted from anomalies can be evaluated as benchmark questions, but router tool execution is a separate scope.

## What is needed

- A Gemini API key, used for benchmark generation and LLM judging.
- Your source documents, chunk JSON, or an already populated `telecom_specs` collection.
- Installed embedding/reranking dependencies and access to their BGE model weights, either cached or downloadable.
- For the original expert backend: its running OpenAI-compatible endpoint and the correct model ID.

An API key does not supply your documents or fine-tuned model. If the expert server is unavailable, explicitly choosing Gemini generation allows the run to use the same Gemini key for generation and judging. Those results are marked as Gemini-generated and are not scores for the fine-tuned Mistral model.

## Install dependencies

In your existing repository environment:

```sh
python -m pip install -r evals/full_requirements.txt
```

The dependency file includes evaluation, ChromaDB, SentenceTransformers, BM25, DOCX and expert-client packages. Existing source files and the original requirements file are unchanged.

## Put secrets in the environment

Use your repository `.env` or process environment:

```dotenv
GOOGLE_API_KEY=your_key
LOCAL_MODEL_URL=https://your-running-expert-endpoint/v1
LOCAL_MODEL_API_KEY=not-needed
EXPERT_MODEL_ID=/tmp/mistral-7b-instruct-v0.3.Q4_K_M.gguf
```

Alternatively `--ask-key` asks for the Gemini key without displaying it and keeps it only in the process environment. Do not put actual keys in GitHub, benchmark JSON or chat messages. Reports and manifests do not contain API keys or endpoint credentials.

## Start to finish with your original expert

If your database is already populated:

```sh
python -m evals.complete --ask-key --samples 20 --k 3 --depth 20 --context-k 3
```

To start from raw `.docx`, `.txt` or `.md` specifications:

```sh
python -m evals.complete --specs data/specs --ask-key --samples 20 --k 3 --depth 20 --context-k 3
```

To start from your chunk JSON instead:

```sh
python -m evals.complete --chunks data/chunks/vector_chunks.json --ask-key --samples 20
```

Raw-source ingestion preserves DOCX table order, uses globally unique source/section IDs and respects the embedding tokenizer window. Source chunks are ingested into an isolated content-addressed evaluation collection; the original collection is preserved. Supplied chunk JSON must already have unique IDs and appropriate chunk sizes.

With no source flags, the runner reads the selected existing collection, preserving the exact IDs used for relevance labeling. It does not require a separate exported corpus file. Use `--collection NAME` for a non-default populated collection.

The original expert's system prompt, temperature 0.2, and 512-token generation budget are preserved. The actual answer sees `--context-k` chunks (default 3). `--depth` controls how many ranked chunks are retrieved and reranked for MRR. In the existing pipeline, increasing depth also enlarges the dense/hybrid candidate pool; this can affect the top chunks. Record the parameters when comparing experiments.

## One Gemini key, with Gemini answer generation

```sh
python -m evals.complete --specs data/specs --generation-backend gemini --ask-key --samples 20
```

This still uses your real BGE/BM25/cross-encoder retrieval pipeline; Gemini replaces the fine-tuned answer generator only because the backend was explicitly selected. The report identifies this backend. `LOCAL_MODEL_URL` is not required in Gemini mode. `--generation-model`, `--benchmark-model` and `--judge-model` are configurable.

## How the labeled benchmark is made

By default the runner samples your indexed chunks with a fixed seed and creates question/reference pairs grounded in them. It verifies each quoted evidence span occurs verbatim in its source and saves the source ID as a known positive relevance label. At least the requested number of source chunks of 80 or more characters must be available; otherwise choose fewer `--samples` or supply more source material.

These references and labels are **synthetic silver labels**, not human-verified ground truth. A valid source quote does not prove every reference claim is correct. Known positive IDs may omit other relevant chunks. Review the saved `benchmark.json`, add other relevant IDs and correct references before using scores as reviewed benchmark results. Same-model benchmark generation and judging can introduce correlated bias, and source-derived questions favor single-passage answers. The report preserves reference provenance.

References are kept out of the answer-generation prompt. The generator receives the question and retrieved context only.

To use your own labeled benchmark:

```sh
python -m evals.complete --questions evals/reviewed_benchmark.json --samples 20 --ask-key
```

Each row needs a unique string `id`, `question`, `reference_answer`, `relevant_doc_ids`, and preferably `reference_source` and `reference_reviewed`. Relevant IDs must exist in the selected corpus. A benchmark built from an isolated evaluation collection must be run against the same collection or same source input. By default unlabeled questions are rejected before generation because all six metrics cannot be derived from them. `--allow-partial` permits missing labels but reports affected scores as unmeasured. Unanswerable questions with no relevant documents have undefined retrieval recall under this definition and need partial mode.

## Exact metric definitions

| Requested metric | Computation |
|---|---|
| MRR | Mean reciprocal rank of the first labeled relevant chunk within `--depth`; 0 if no relevant chunk occurs there |
| Top-k Accuracy | Fraction of labeled questions with at least one relevant chunk in the first k results; Hit@k, not Precision@k |
| Accuracy | Fraction of judged answers whose reference-correctness rating meets `--correctness-threshold`; default 1.0 means fully correct |
| Recall | Mean of relevant chunks in the first k results divided by all labeled relevant chunks per question |
| Relevance | Mean LLM rating for how directly and completely the answer addresses the question |
| Faithfulness | Mean fraction of atomic factual claims supported by the exact context given to the answer generator |

Quality values range from 0 to 1. MRR is truncated at the captured ranking depth; it is not an unrestricted full-corpus MRR measurement. Duplicate retrieval IDs do not inflate scores. Accuracy is binary per answer, while the underlying correctness rating is retained in the judgments. A different threshold must be disclosed.

Every metric includes scored count and coverage over all attempted questions. Judge outages, missing references and no-claim responses are null, not perfect or zero quality scores. Pipeline failures remain in traces and reports. Means over scored samples must be interpreted alongside coverage. The command returns nonzero for a partial run even though it saves the partial report. It never invents an overall composite score.

## Outputs

Every run gets a new timestamped directory under `outputs/` by default. You can choose a new `--output` path. Existing runs are not overwritten or automatically resumed.

- `corpus_chunks.json`: exact corpus snapshot used for benchmark construction.
- `benchmark.json`: questions, references, labels and provenance.
- `traces.json`: actual questions, ranked retrieval, generator context, answers, timings and failures.
- `per_question_judgments.json`: atomic claims, judge reasoning, correctness ratings and all six scores per row.
- `per_question_scores.csv`: the six metrics for each question.
- `final_evaluation.json`: aggregate values and coverage, machine-readable.
- `final_evaluation.csv`: exactly the six requested aggregate metric rows.
- `evaluation_report.md`: readable six-metric scorecard and definitions.
- `run_manifest.json`: stage, status, corpus digest, collection, selected models and parameters; never the key.

A failed stage is recorded in the manifest. Successful API authentication is checked before loading/ingesting models; the expert endpoint's advertised model IDs are checked in expert mode. The API key does not guarantee model availability or quota. Model errors and unavailable dependencies are reported without publishing raw API payloads.

## Check prerequisites without calls

```sh
python -m evals.complete --specs data/specs --generation-backend gemini --check
```

`--check` only inspects installed dependencies, presence of credentials and source paths. It makes no API requests or model downloads and does not prove API quota, endpoint health or corpus relevance. Do not use `--ask-key` if you want a fully noninteractive check.

## Validation performed

The current repository snapshot plus this six-metric workflow passed 34 offline tests on Python 3.12.14, including a controlled-service source-to-benchmark-to-retrieval-to-answer-to-scorecard run, hand-calculated metric cases and DOCX ingestion fixtures. These are workflow validation results, not production RAG scores. Live API/model execution has not been performed in the delivery environment because credentials and retrieval runtime dependencies are absent.

Run tests:

```sh
python -m unittest discover -s tests -v
```
