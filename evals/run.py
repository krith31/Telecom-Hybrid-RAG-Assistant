"""Evaluate actual retrieval + expert outputs; export per-question and aggregate reports."""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import time
from .result_accuracy import AccuracyEvaluator
from .context_quality import BehaviorEvaluator, ContextQualityEvaluator
from .capture import generate_evaluation_dataset, write_json
from .result_faithfulness import FaithfulnessEvaluator
from .metrics import citation_validity, retrieval_metrics
from .schema import CapturedAnswer

def evaluate_records(records, judge_model="gemini-3.8-flash", client=None, k=3, checkpoint=None):
    if not isinstance(records,list) or not records:
        raise ValueError("No records to evaluate")
    judges = {"faithfulness":FaithfulnessEvaluator(judge_model,client),
              "answer":AccuracyEvaluator(judge_model,client),
              "context":ContextQualityEvaluator(judge_model,client),
              "behavior":BehaviorEvaluator(judge_model,client)}
    results = []
    for index, record in enumerate(records):
        try:
            record=CapturedAnswer.model_validate(record).model_dump()
        except Exception as error:
            results.append({"id":str(index),"status":"invalid_record", "question":"", "metrics":{},
                            "judgments":{}, "error":type(error).__name__})
            if checkpoint:
                write_json(checkpoint,results)
            continue
        row = {**record, "judge_model":judge_model, "metrics":{}, "judgments":{}}
        question, answer = record.get("question",""), record.get("generated_answer","")
        chunks = record.get("retrieved_chunks",[])
        context = record.get("retrieved_context","")
        reference = record.get("reference_answer")
        status = record.get("status", "ok")
        row["status"] = status
        if status == "ok" and answer.strip() and question.strip():
            started = time.perf_counter()
            faith = judges["faithfulness"].evaluate(answer,context).model_dump()
            relevance = judges["answer"].evaluate(question,answer).model_dump()
            row["judgments"].update(faithfulness=faith, answer_relevance=relevance)
            row["metrics"]["faithfulness"] = faith["score"]
            row["metrics"]["answer_relevance"] = relevance["score"]
            if reference:
                correctness = judges["answer"].evaluate(question,answer,reference).model_dump()
                row["judgments"]["answer_correctness"] = correctness
                row["metrics"]["answer_correctness"] = correctness["score"]
            cq = judges["context"].evaluate(question,chunks,reference)
            behavior = judges["behavior"].evaluate(question,answer,chunks,record.get("expected_abstention"))
            row["judgments"].update(context=cq, behavior=behavior)
            for name in ("context_relevance", "reference_context_recall"):
                row["metrics"][name] = cq.get(name)
            for name in ("citation_support", "citation_completeness", "abstention_accuracy"):
                row["metrics"][name] = behavior.get(name)
            row["judge_ms"] = (time.perf_counter()-started)*1000
        elif status == "ok":
            row["status"] = "invalid_output"
        try:
            ranking = retrieval_metrics([c["id"] for c in chunks],record.get("relevant_doc_ids"),k)
            row["retrieval_labels_status"] = ranking.pop("status")
            row["metrics"].update(ranking)
        except (KeyError,TypeError,ValueError) as error:
            row["retrieval_labels_status"] = "error"
            row["retrieval_labels_error"] = type(error).__name__
        if answer and chunks:
            identity = citation_validity(answer,chunks)
            row["citation_identity"] = identity
            row["metrics"]["citation_source_validity"] = identity["source_validity"]
        results.append(row)
        if checkpoint:
            write_json(checkpoint,results)
    return results

def summarize(results, k=3):
    names = ["faithfulness","answer_relevance","answer_correctness","context_relevance",
             "reference_context_recall","citation_support","citation_completeness",
             "citation_source_validity","abstention_accuracy"]
    names += [f"{m}@{k}" for m in ("precision","recall","hit","mrr","ndcg","ap")]
    attempted = len(results)
    summary = {"attempted":attempted, "pipeline_successes":sum(r["status"]=="ok" for r in results),
               "metrics":{},"pipeline_status_counts":{},"judge_status_counts":{},"reference_sources":{}}
    summary["pipeline_success_rate"] = summary["pipeline_successes"]/attempted if attempted else None
    claims=[c for r in results for c in r.get("judgments",{}).get("faithfulness",{}).get("verifications",[])]
    summary["claim_totals"]={"extracted":len(claims),"supported":sum(c["is_supported"] for c in claims),
                             "micro_faithfulness":sum(c["is_supported"] for c in claims)/len(claims) if claims else None,
                             "unsupported_claim_fraction":sum(not c["is_supported"] for c in claims)/len(claims) if claims else None}
    for row in results:
        status=row["status"]
        summary["pipeline_status_counts"][status]=summary["pipeline_status_counts"].get(status,0)+1
        source=row.get("reference_source","unspecified") if row.get("reference_answer") else "no_reference"
        summary["reference_sources"][source]=summary["reference_sources"].get(source,0)+1
        for name, judgment in row.get("judgments",{}).items():
            key=name+":"+judgment["status"]
            summary["judge_status_counts"][key]=summary["judge_status_counts"].get(key,0)+1
    for name in names:
        vals=[r["metrics"].get(name) for r in results]
        valid=[v for v in vals if isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)]
        summary["metrics"][name]={"mean":statistics.mean(valid) if valid else None,
                                  "scored":len(valid),"attempted":attempted,
                                  "coverage":len(valid)/attempted if attempted else None,
                                  "stddev":statistics.stdev(valid) if len(valid)>1 else None}
    for timing in ("retrieval_ms","generation_ms","judge_ms"):
        vals=sorted(r[timing] for r in results if isinstance(r.get(timing),(int,float)) and math.isfinite(r[timing]))
        summary[timing]={"mean":statistics.mean(vals) if vals else None,
                         "p95":vals[max(0,math.ceil(.95*len(vals))-1)] if vals else None,"samples":len(vals)}
    summary["correctness_by_reference_source"]={}
    for source in summary["reference_sources"]:
        vals=[r["metrics"].get("answer_correctness") for r in results if r.get("reference_source")==source]
        vals=[v for v in vals if isinstance(v,(int,float)) and math.isfinite(v)]
        summary["correctness_by_reference_source"][source]={"mean":statistics.mean(vals) if vals else None,"scored":len(vals)}
    return summary

def save_report(output, results, summary, metadata):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    write_json(output/"judgments.json",results)
    write_json(output/"summary.json",{**metadata,**summary})
    names=list(summary["metrics"])
    with (output/"scores.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["id","status","question","reference_source"]+names)
        writer.writeheader()
        for r in results:
            writer.writerow({"id":r.get("id"),"status":r["status"],"question":r.get("question"),
                             "reference_source":r.get("reference_source"),**{n:r["metrics"].get(n) for n in names}})
    lines=["# RAG answer evaluation", "", f"Mode: {metadata.get('mode','live')}",
           f"Judge: {metadata['judge_model']}",f"Pipeline successes: {summary['pipeline_successes']}/{summary['attempted']}",
           "", "Means use scored samples only. Coverage uses every attempted sample.",
           "Missing values and judge errors are not zero scores. No composite score is invented.",
           "", "| Metric | Mean (0-1) | Scored / attempted |", "|---|---:|---:|"]
    for name,m in summary["metrics"].items():
        value=f"{m['mean']:.4f}" if m["mean"] is not None else "Unmeasured"
        lines.append(f"| {name} | {value} | {m['scored']}/{m['attempted']} |")
    lines += ["", "Context relevance is a judge estimate, not gold context precision.",
              "Reference context recall measures coverage of reference claims, not corpus-wide recall.",
              "Ranking labels may be incomplete. Synthetic references are silver labels, not verified ground truth.",
              "Citation source validity checks identity; citation support judges entailment.",
              "Reference sources: "+json.dumps(summary["reference_sources"]),
              "Correctness by reference source: "+json.dumps(summary["correctness_by_reference_source"]),
              "Atomic claim totals: "+json.dumps(summary["claim_totals"]),
              "Inspect judgments.json for per-question evidence and judge errors.",
              "Latency excludes initial model loading; token usage/cost are not measured.", ""]
    (output/"report.md").write_text("\n".join(lines),encoding="utf-8")

def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group()
    source.add_argument("--questions",type=Path,default=None)
    source.add_argument("--dataset",type=Path,help="Score captured outputs without generating again")
    parser.add_argument("--output",type=Path,default=root/"outputs/rag_evaluation")
    parser.add_argument("--judge-model",default="gemini-3.8-flash")
    parser.add_argument("--limit",type=int,default=20)
    parser.add_argument("--top-k",type=int,default=3)
    args=parser.parse_args()
    if args.limit<1 or args.top_k<1:
        parser.error("limit and top-k must be positive")
    try:
        from dotenv import load_dotenv
        load_dotenv(root/".env")
    except ImportError:
        pass
    import os
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        parser.error("Set GOOGLE_API_KEY in .env or the environment")
    args.output.mkdir(parents=True,exist_ok=True)
    if args.dataset:
        records=json.loads(args.dataset.read_text(encoding="utf-8"))
        if not isinstance(records,list):
            parser.error("dataset must be a JSON list")
        records=records[:args.limit]
    else:
        if not os.getenv("LOCAL_MODEL_URL"):
            parser.error("Set LOCAL_MODEL_URL to the expert's /v1 endpoint")
        from rag.pipeline import TelecomRAGPipeline
        pipeline=TelecomRAGPipeline()
        if pipeline.vector_store.get_collection_count()==0:
            parser.error("telecom_specs collection is empty; ingest the corpus first")
        if pipeline.reranker.model is None:
            print("WARNING: reranker unavailable; measuring fallback behavior")
        questions=args.questions or root/"evals/telecom_questions.json"
        records=generate_evaluation_dataset(questions,args.output/"traces.json",args.limit,
                                            pipeline=pipeline,top_k=args.top_k)
    results=evaluate_records(records,args.judge_model,k=args.top_k,checkpoint=args.output/"judgments.json")
    summary=summarize(results,args.top_k)
    metadata={"mode":"live", "judge_model":args.judge_model,"top_k":args.top_k,
              "created_at":datetime.now(timezone.utc).isoformat(),
              "scope":"Existing retriever + equivalent expert prompt/settings; router not exercised"}
    save_report(args.output,results,summary,metadata)
    print(f"Report: {args.output/'report.md'}")
    judge_errors=any(j.get("status")=="error" for r in results for j in r["judgments"].values())
    return 1 if summary["pipeline_successes"] != summary["attempted"] or judge_errors else 0

if __name__ == "__main__":
    raise SystemExit(main())
