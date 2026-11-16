# Evaluate actual Telecom RAG outputs

This is an additive installation. All pre-existing files are preserved, including main.py and the original evals/accuracy.py, evals/faithfulness.py and evals/dataset_generator.py. The new runner uses result_accuracy.py, result_faithfulness.py and capture.py instead.

The runner calls the existing TelecomRAGPipeline.retrieve_context(), captures exact ranked chunks, source metadata, context and expert answers, then judges those outputs. rag/expert.py is an evaluation adapter extracted from main.py with the same system prompt, model default, temperature 0.2 and 512-token budget. It avoids importing main.py and starting its router.

Scope: retrieval plus expert generation. Gemini router tool selection and anomaly detector performance are not measured. Anomaly inputs are evaluated using their extracted diagnostic question, not the full anomaly report.

## Setup

Use your existing working RAG environment. Install evaluation dependencies:

    python -m pip install -r evals/requirements.txt

Set the following in the repository's .env (see evals/.env.example):

    GOOGLE_API_KEY=your_google_api_key
    LOCAL_MODEL_URL=https://your-running-expert-endpoint/v1
    LOCAL_MODEL_API_KEY=not-needed
    EXPERT_MODEL_ID=/tmp/mistral-7b-instruct-v0.3.Q4_K_M.gguf

The populated telecom_specs ChromaDB collection, embedding model, reranker and running expert endpoint are required for live capture. The package does not supply private documents, weights or credentials.

## Run the included 20 questions

From the repository root:

    python -m evals.run --questions evals/telecom_questions.json --limit 20 --top-k 3 --output outputs/rag_evaluation

The starter questions cover RLF, CQI/MCS, scheduling, HARQ, PDCP/RLC, mobility, QoS, diagnostics, unavailable private information and an adversarial request. They are not an expert-validated benchmark. Most have no reference or relevant-document labels, so correctness and ranking metrics remain unmeasured for them. The expert is called even when retrieval returns no chunks, allowing evaluation of actual abstention behavior.

## Build reference-based questions from your own corpus

    python -m evals.build_benchmark --chunks data/chunks/vector_chunks.json --count 20 --seed 31 --output evals/corpus_benchmark.json
    python -m evals.run --questions evals/corpus_benchmark.json --limit 20 --top-k 3 --output outputs/rag_evaluation_corpus

Set --chunks to the actual chunk JSON matching the active index. IDs must be globally unique. Duplicate IDs are rejected to avoid misleading retrieval scores.

The builder samples source chunks, generates answerable question/reference pairs and verifies that their evidence quote occurs verbatim in the source. These are silver labels, not independently verified truth. A matching quote does not guarantee reference correctness. Known positive chunk IDs are not exhaustive relevance judgments. Review references and add other relevant IDs before treating scores as reviewed-reference evaluation. Provenance is retained. Using the same model to generate references and judge answers can introduce correlated bias; model selection is configurable.

References and source evidence quotes are never passed to the expert as answer hints.

## Capture or rescore separately

    python -m evals.capture --input evals/telecom_questions.json --output outputs/captured_answers.json --sample-size 20
    python -m evals.run --dataset outputs/captured_answers.json --limit 20 --output outputs/rescored_answers

The score-only command does not run retrieval or expert generation again.

Anomaly inputs can also be used:

    python -m evals.run --questions data/anomaly_prompts.json --limit 20 --output outputs/rag_evaluation_anomalies

Supported query fields: question, rag_diagnostic_query, diagnostic_question, or full_prompt_text containing DIAGNOSTIC QUESTION:.

Use a different output directory per experiment. Checkpointing is implemented; automatic resume and concurrency are not. Existing outputs at the same path are overwritten.

## Metrics

| Metric | Meaning | Required labels |
|---|---|---|
| Faithfulness | Supported atomic answer claims / extracted factual claims | Exact supplied context |
| Answer relevance | Judge rating of how directly and completely the answer addresses the question | None |
| Correctness | Judge rating against reference answer | Reference, with provenance |
| Context relevance | Judge-estimated fraction of useful retrieved chunks | None |
| Reference context recall | Reference claims supported by combined context / reference claims | Reference |
| Precision@k, Recall@k, Hit@k | Binary relevance of ranked chunk IDs | Relevant IDs |
| MRR@k, nDCG@k, AP@k | Rank-sensitive binary relevance | Relevant IDs |
| Citation source validity | Fraction of parsed citations matching retrieved spec/clause IDs | Source metadata |
| Citation support | Judge-estimated support from the specifically cited chunks | Cited claims and chunks |
| Citation completeness | Fraction of technical claims with supporting citations, judged | Answer and chunks |
| Abstention accuracy | Judged abstention matches supplied expectation | expected_abstention boolean |
| Latency | Mean and p95 retrieval, generation and judge wall times | Real calls |

Scores range from 0 to 1. Precision@k uses k even if fewer results are returned. Ranking IDs are deduplicated. AP@k normalizes by min(k, number of relevant IDs). Context relevance is not gold context precision; reference-claim recall is not corpus-wide recall. Citation identity does not establish semantic support.

Empty answers, no factual claims, unavailable labels and judge errors do not receive perfect scores. Means include scored samples only, with coverage over all attempted samples. Failed samples remain in the output. Read coverage and pipeline success rate alongside quality scores. Macro averages and claim-weighted micro faithfulness are reported separately. An abstention may have no factual claims; its behavioral score is separate.

Structured judge responses are validated. API calls use a 60-second request timeout and bounded retries for transient errors. Models can still vary at temperature zero. Review per-question reasoning: automated judgments can be wrong.

Token usage/cost, initial model loading time, router accuracy and anomaly detector F1 are not measured. References generated from corpus chunks favor single-passage questions and may share model biases with the judge.

## Outputs

- traces.json: actual questions, chunks, metadata, expert answers, timings and failures, in live mode.
- judgments.json: per-question scores, atomic claims, context judgments, citation reasoning and errors.
- scores.csv: one row per attempted question.
- summary.json: means, coverage, status counts, provenance, claim totals and latency.
- report.md: readable scorecard.

The CLI returns a nonzero status when any pipeline sample or judge call fails while retaining reports. Missing optional labels alone are not an error. Empty databases, missing judge keys and absent expert URLs stop live execution with clear messages.

## Offline tests

    python -m unittest discover -s tests -p test_result_evaluation.py -v

Tests use injected services. They verify evaluation accounting and formulas; they are not production answer-quality measurements and make no paid API calls.

This additive publication was not re-executed in the disconnected delivery runtime. See EVALUATION_VALIDATION.md.
