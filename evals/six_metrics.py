"""The six requested result metrics, with explicit definitions and coverage."""
import csv
from pathlib import Path
import statistics

from .capture import write_json
from .result_accuracy import AccuracyEvaluator
from .result_faithfulness import FaithfulnessEvaluator
from .schema import CapturedAnswer
from .judge import safe_api_error

METRICS = ("MRR", "Top-k Accuracy", "Accuracy", "Recall", "Relevance", "Faithfulness")


def retrieval_scores(ids, relevant_ids, k, depth):
    if k < 1 or depth < k:
        raise ValueError("Require 1 <= k <= depth")
    if relevant_ids is None or not relevant_ids:
        return {"MRR":None,"Top-k Accuracy":None,"Recall":None}
    if not isinstance(relevant_ids,list) or any(not isinstance(x,str) for x in relevant_ids):
        raise ValueError("Relevant IDs must be a list of strings")
    relevant=set(relevant_ids)
    ranked=list(dict.fromkeys(ids))[:depth]
    first=next((rank for rank,doc_id in enumerate(ranked,1) if doc_id in relevant),None)
    top=set(ranked[:k])
    return {"MRR":1/first if first else 0.0,
            "Top-k Accuracy":float(bool(top & relevant)), "Recall":len(top & relevant)/len(relevant)}


def evaluate_six(records, k=3, depth=20, correctness_threshold=1.0,
                 judge_model="gemini-3.8-flash", client=None, checkpoint=None, fail_fast=False):
    if not isinstance(records,list) or not records:
        raise ValueError("Need a nonempty list of captured pipeline outputs")
    if k<1 or depth<k or not 0<correctness_threshold<=1:
        raise ValueError("Require 1 <= k <= depth and 0 < correctness_threshold <= 1")
    faith=FaithfulnessEvaluator(judge_model,client,raise_errors=fail_fast)
    answer_judge=AccuracyEvaluator(judge_model,client,raise_errors=fail_fast)
    results=[]
    for index,record in enumerate(records):
        failure=None
        row={"id":str(index),"status":"invalid_record","question":"",
             "metrics":dict.fromkeys(METRICS),"judgments":{}}
        try:
            data=CapturedAnswer.model_validate(record).model_dump()
            row.update({key:data.get(key) for key in ("id","status","question","reference_source")})
            row["reference_reviewed"]=data.get("reference_reviewed",False)
            row["relevant_doc_ids"]=data["relevant_doc_ids"]
            ranking=data.get("retrieval_ranking",data["retrieved_chunks"])
            row["retrieved_doc_ids"]=[c["id"] for c in ranking]
            if data["status"] in ("ok","generation_error","invalid_output"):
                row["metrics"].update(retrieval_scores(row["retrieved_doc_ids"],data["relevant_doc_ids"],k,depth))
            if data["status"]=="ok" and data["question"].strip():
                print(f"Scoring {index+1}/{len(records)}: faithfulness",flush=True)
                f=faith.evaluate(data["generated_answer"],data["retrieved_context"]).model_dump()
                row["judgments"]["faithfulness"]=f
                row["metrics"]["Faithfulness"]=f["score"]
                if checkpoint:
                    write_json(checkpoint,results+[row])
                print(f"Scoring {index+1}/{len(records)}: relevance",flush=True)
                r=answer_judge.evaluate(data["question"],data["generated_answer"]).model_dump()
                row["judgments"]["relevance"]=r
                row["metrics"]["Relevance"]=r["score"]
                if checkpoint:
                    write_json(checkpoint,results+[row])
                if data["reference_answer"] and data["reference_answer"].strip():
                    print(f"Scoring {index+1}/{len(records)}: accuracy",flush=True)
                    a=answer_judge.evaluate(data["question"],data["generated_answer"],data["reference_answer"]).model_dump()
                    row["judgments"]["accuracy"]=a
                    row["correctness_rating"]=a["score"]
                    row["metrics"]["Accuracy"]=float(a["score"]>=correctness_threshold) if a["score"] is not None else None
                if not data["generated_answer"].strip():
                    row["status"]="empty_answer"
        except Exception as error:
            row["error"]=type(error).__name__
            if fail_fast:
                row["error_detail"]=safe_api_error(error)
                failure=error
        results.append(row)
        if checkpoint:
            write_json(checkpoint,results)
        measured=sum(value is not None for value in row["metrics"].values())
        print(f"Scored {index+1}/{len(records)}: {measured}/6 metric values available",flush=True)
        if failure is not None:
            raise failure
    definitions={
        "MRR":f"Mean reciprocal rank of first labeled relevant chunk within captured depth {depth}; zero on a labeled miss.",
        "Top-k Accuracy":f"Fraction of labeled questions with a relevant chunk in the first {k} results (Hit@{k}).",
        "Accuracy":f"Fraction of scored answers whose reference-correctness rating >= {correctness_threshold}.",
        "Recall":f"Macro mean of relevant chunks in the first {k} / all labeled relevant chunks.",
        "Relevance":"Mean LLM answer-relevance rating (0 to 1).",
        "Faithfulness":"Macro mean supported factual claims / extracted factual claims, judged only against supplied context."
    }
    summary={"attempted":len(results),"k":k,"retrieval_depth":depth,
             "correctness_threshold":correctness_threshold,"judge_model":judge_model,"metrics":{},
             "pipeline_status_counts":{},"judge_status_counts":{},"reference_sources":{}}
    for metric in METRICS:
        values=[r["metrics"][metric] for r in results if r["metrics"][metric] is not None]
        summary["metrics"][metric]={"value":statistics.mean(values) if values else None,
                                     "scored":len(values),"attempted":len(results),
                                     "coverage":len(values)/len(results),"definition":definitions[metric]}
    for row in results:
        status=row["status"]
        summary["pipeline_status_counts"][status]=summary["pipeline_status_counts"].get(status,0)+1
        source=row.get("reference_source") or "unspecified"
        summary["reference_sources"][source]=summary["reference_sources"].get(source,0)+1
        for name,j in row["judgments"].items():
            key=name+":"+j["status"]
            summary["judge_status_counts"][key]=summary["judge_status_counts"].get(key,0)+1
    summary["complete"]=all(m["scored"]==len(results) for m in summary["metrics"].values())
    summary["interpretation"]=("Scores use scored samples only; inspect coverage and failures. MRR is truncated at retrieval_depth. "
                               "Recall is against supplied labels, which may be incomplete. Synthetic references are silver labels, not human gold. "
                               "No claims, missing labels and judge failures have null scores, not fabricated values.")
    return results,summary


def export_six(output, results, summary, manifest):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    write_json(output/"per_question_judgments.json",results)
    write_json(output/"final_evaluation.json",{**summary,"run":manifest})
    with (output/"final_evaluation.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.writer(f);writer.writerow(["metric","value_0_to_1","scored","attempted","coverage"])
        for metric in METRICS:
            m=summary["metrics"][metric]
            writer.writerow([metric,m["value"],m["scored"],m["attempted"],m["coverage"]])
    with (output/"per_question_scores.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["id","status","question"]+list(METRICS))
        writer.writeheader()
        for row in results:
            writer.writerow({k:row.get(k) for k in ("id","status","question")} | row["metrics"])
    lines=["# Final RAG evaluation", "",f"Generator: {manifest.get('generation_backend')}",
           f"Judge: {summary['judge_model']}",f"Questions attempted: {summary['attempted']}","",
           "| Metric | Value (0–1) | Scored / attempted |","|---|---:|---:|"]
    for metric in METRICS:
        m=summary["metrics"][metric];value=f"{m['value']:.4f}" if m["value"] is not None else "Unmeasured"
        lines.append(f"| {metric} | {value} | {m['scored']}/{m['attempted']} |")
    lines += ["",summary["interpretation"],"","## Definitions",""]
    lines += [f"- **{metric}:** {summary['metrics'][metric]['definition']}" for metric in METRICS]
    lines += ["", "See run_manifest.json for backend, models and corpus identity. "
              "Gemini generation evaluates a different generator from the fine-tuned expert. "
              "Router tool selection and anomaly detection are outside this RAG evaluation.", ""]
    (output/"evaluation_report.md").write_text("\n".join(lines),encoding="utf-8")

