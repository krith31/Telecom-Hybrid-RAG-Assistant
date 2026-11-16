"""Capture actual answers without importing main.py; retain every attempted sample."""
import argparse
import json
import os
from pathlib import Path
import time

def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)

def extract_query(record):
    for field in ("question", "rag_diagnostic_query", "diagnostic_question"):
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    report = record.get("full_prompt_text", "")
    if isinstance(report, str) and "DIAGNOSTIC QUESTION:" in report:
        return report.rsplit("DIAGNOSTIC QUESTION:", 1)[1].strip()
    return ""

def generate_evaluation_dataset(prompts_json_path, output_dataset_path, sample_size=10,
                                pipeline=None, generator=None, top_k=3):
    if sample_size < 1 or top_k < 1:
        raise ValueError("sample_size and top_k must be positive")
    windows = json.loads(Path(prompts_json_path).read_text(encoding="utf-8"))
    if not isinstance(windows, list) or not windows:
        raise ValueError("Input must be a nonempty JSON list")
    dataset = []
    for index, window in enumerate(windows[:sample_size]):
        record = {"id":str(index), "question":"", "status":"invalid_input", "error":None,
                  "retrieved_chunks":[], "retrieved_context":"", "generated_answer":"",
                  "reference_answer":None, "relevant_doc_ids":None, "retrieval_ms":None,
                  "generation_ms":None, "reranker_active":None}
        try:
            if not isinstance(window, dict):
                raise ValueError("Each input record must be an object")
            record.update(id=str(window.get("id", window.get("window_id", index))),
                          question=extract_query(window), reference_answer=window.get("reference_answer"),
                          relevant_doc_ids=window.get("relevant_doc_ids"), domain=window.get("domain"),
                          expected_abstention=window.get("expected_abstention"),
                          reference_source=window.get("reference_source", "user_supplied"),
                          reference_reviewed=window.get("reference_reviewed", False),
                          category=window.get("category"))
            if not record["question"]:
                raise ValueError("Missing diagnostic question")
            record["status"] = "retrieval_error"
            if pipeline is None:
                from rag.pipeline import TelecomRAGPipeline
                pipeline = TelecomRAGPipeline()
            start = time.perf_counter()
            try:
                chunks = pipeline.retrieve_context(record["question"], top_k=top_k)
            finally:
                record["retrieval_ms"] = (time.perf_counter()-start)*1000
            record["retrieved_chunks"] = chunks
            reranker = getattr(pipeline, "reranker", None)
            record["reranker_active"] = (reranker.model is not None if reranker is not None else None)
            record["retrieved_context"] = "\n\n".join(
                f"[Spec: {c.get('metadata', {}).get('spec', 'unknown')} | Clause "
                f"{c.get('metadata', {}).get('section_id', 'unknown')}]\n{c['content']}" for c in chunks)
            record["retrieval_empty"] = not bool(chunks)
            record["status"] = "generation_error"
            if generator is None:
                from rag.expert import call_expert
                generator = call_expert
            prompt = "=== 3GPP SPECS ===\n"+record["retrieved_context"]+"\n=== TASK ===\n"+record["question"]
            start = time.perf_counter()
            try:
                answer = generator(prompt)
                if not isinstance(answer, str) or not answer.strip():
                    raise ValueError("Empty generated answer")
                record["generated_answer"] = answer
            finally:
                record["generation_ms"] = (time.perf_counter()-start)*1000
            record["status"] = "ok"
        except Exception as error:
            record["error"] = type(error).__name__
        dataset.append(record)
        write_json(output_dataset_path, dataset)
    return dataset

def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root/"evals/telecom_questions.json")
    parser.add_argument("--output", type=Path, default=root/"outputs/captured_answers.json")
    parser.add_argument("--sample-size", type=int, default=20)
    args = parser.parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv(root/".env")
    except ImportError:
        pass
    rows = generate_evaluation_dataset(args.input, args.output, args.sample_size)
    good = sum(r["status"] == "ok" for r in rows)
    print(f"Generated {good}/{len(rows)} successful samples; all attempts saved to {args.output}")
    return 0 if good == len(rows) else 1

if __name__ == "__main__":
    raise SystemExit(main())
