"""One command: corpus -> benchmark -> retrieval -> answers -> six-metric report."""
import argparse
from datetime import datetime, timezone
import getpass
import importlib.util
import json
import os
from pathlib import Path

from pydantic import BaseModel, StrictBool
from .build_benchmark import build_benchmark
from .capture import generate_evaluation_dataset, write_json
from .corpus import prepare_corpus
from .generators import GeminiGenerator
from .judge import Judge, safe_api_error
from .six_metrics import evaluate_six, export_six


class Ready(BaseModel):
    ready: StrictBool


class CollectionPipeline:
    def __init__(self,pipeline,name,context_k):
        self.pipeline=pipeline;self.name=name
        self.reranker=pipeline.reranker
        self.context_k=context_k;self.rankings=[]
    def retrieve_context(self,query,top_k):
        self.rankings.append([])
        ranked=self.pipeline.retrieve_context(query,top_k=top_k,collection_name=self.name)
        self.rankings[-1]=ranked
        return ranked[:self.context_k]


def validate_questions(rows,known_ids,allow_partial=False):
    if not isinstance(rows,list) or not rows:
        raise ValueError("Question benchmark must be a nonempty JSON list")
    seen=set()
    for row in rows:
        if not isinstance(row,dict) or not isinstance(row.get("question"),str) or not row["question"].strip():
            raise ValueError("Each benchmark row needs a nonempty question")
        id_=row.get("id",row.get("window_id"))
        if not isinstance(id_,str) or not id_ or id_ in seen:
            raise ValueError("Benchmark rows need unique nonempty string IDs")
        seen.add(id_)
        reference=row.get("reference_answer")
        relevant=row.get("relevant_doc_ids")
        if not allow_partial and (not isinstance(reference,str) or not reference.strip() or not relevant):
            raise ValueError("All six metrics need reference_answer and relevant_doc_ids. Omit --questions to build a corpus benchmark automatically, or use --allow-partial.")
        if relevant is not None:
            if not isinstance(relevant,list) or any(not isinstance(x,str) for x in relevant):
                raise ValueError("Relevant IDs must be a list of strings")
            unknown=set(relevant)-known_ids
            if unknown:
                raise ValueError("Relevant IDs do not match the selected indexed corpus")
    return rows


def run_workflow(config,pipeline=None,generator=None,judge_client=None,benchmark_builder=build_benchmark):
    output=Path(config.output)
    output.mkdir(parents=True,exist_ok=True)
    if (output/"run_manifest.json").exists():
        raise ValueError("Output directory already contains a run; choose a new --output path")
    manifest={"state":"running","stage":"initialize","created_at":datetime.now(timezone.utc).isoformat(),
              "generation_backend":config.generation_backend,"generation_model":config.generation_model,
              "judge_model":config.judge_model,"benchmark_model":config.benchmark_model,
              "k":config.k,"retrieval_depth":config.depth,"context_k":config.context_k,"requested_samples":config.samples,
              "correctness_threshold":config.correctness_threshold,"injected_services":bool(pipeline or generator or judge_client),
              "scope":"RAG retrieval and generation. Router and anomaly detector excluded."}
    def stage(name):
        manifest["stage"]=name
        write_json(output/"run_manifest.json",manifest)
        print("Evaluation stage:",name)
    try:
        stage("initialize")
        if pipeline is None:
            from rag.pipeline import TelecomRAGPipeline
            pipeline=TelecomRAGPipeline()
        if pipeline.reranker.model is None and not config.allow_reranker_fallback:
            raise RuntimeError("Reranker failed to load. Restore its model or explicitly choose --allow-reranker-fallback")
        manifest["reranker_active"]=pipeline.reranker.model is not None
        stage("prepare_corpus")
        chunks,collection,digest=prepare_corpus(pipeline,output,config.specs,config.chunks,config.collection)
        manifest.update(collection=collection,corpus_sha256=digest,corpus_chunks=len(chunks))
        stage("prepare_benchmark")
        if config.questions:
            questions=json.loads(Path(config.questions).read_text(encoding="utf-8"))[:config.samples]
            manifest["benchmark_type"]="supplied"
        else:
            resumed=getattr(config,"resume_benchmark",None)
            extra={"existing":json.loads(Path(resumed).read_text(encoding="utf-8"))} if resumed else {}
            questions=benchmark_builder(chunks,output/"benchmark.json",config.samples,
                                        config.seed,config.benchmark_model,judge_client,**extra)
            manifest["benchmark_type"]="synthetic_from_corpus"
            if resumed:
                manifest["resumed_benchmark_questions"]=len(extra["existing"])
        validate_questions(questions,{c["id"] for c in chunks},config.allow_partial)
        write_json(output/"benchmark.json",questions)
        stage("generate_answers")
        if generator is None:
            if config.generation_backend=="gemini":
                generator=GeminiGenerator(config.generation_model,judge_client)
            else:
                from rag.expert import call_expert
                generator=call_expert
        adapter=CollectionPipeline(pipeline,collection,config.context_k)
        traces=generate_evaluation_dataset(output/"benchmark.json",output/"traces.json",config.samples,
                                           adapter,generator,config.depth)
        if len(adapter.rankings)!=len(traces):
            raise RuntimeError("Retrieval trace accounting mismatch")
        for trace,ranking in zip(traces,adapter.rankings):
            trace["retrieval_ranking"]=ranking
            trace["context_k"]=config.context_k
        write_json(output/"traces.json",traces)
        manifest["pipeline_successes"]=sum(r["status"]=="ok" for r in traces)
        stage("judge_answers")
        results,summary=evaluate_six(traces,config.k,config.depth,config.correctness_threshold,
                                     config.judge_model,judge_client,output/"per_question_judgments.json")
        manifest["state"]="complete" if summary["complete"] else "partial"
        manifest["stage"]="finished"
        manifest["finished_at"]=datetime.now(timezone.utc).isoformat()
        export_six(output,results,summary,manifest)
        write_json(output/"run_manifest.json",manifest)
        return summary
    except Exception as error:
        manifest["state"]="failed"
        manifest["error_type"]=type(error).__name__
        manifest["error_detail"]=safe_api_error(error)
        write_json(output/"run_manifest.json",manifest)
        raise


def package_available(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError,ValueError,ModuleNotFoundError):
        return False


def preflight(config):
    required=["pydantic","google.genai","chromadb","sentence_transformers","rank_bm25"]
    if config.specs:
        required.append("docx")
    if config.generation_backend=="expert":
        required.append("openai")
    checks={"dependencies":{name:package_available(name) for name in required},
            "judge_key_present":bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")),
            "expert_url_present":bool(os.getenv("LOCAL_MODEL_URL")) if config.generation_backend=="expert" else None,
            "sources_present":Path(config.specs).is_dir() if config.specs else Path(config.chunks).is_file() if config.chunks else None}
    return checks


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    sources=parser.add_mutually_exclusive_group()
    sources.add_argument("--specs",type=Path,help="Ingest .docx/.txt/.md sources into an isolated evaluation collection")
    sources.add_argument("--chunks",type=Path,help="Ingest existing chunk JSON into an isolated evaluation collection")
    parser.add_argument("--collection",default="telecom_specs",help="Existing collection used when neither source flag is supplied")
    benchmarks=parser.add_mutually_exclusive_group()
    benchmarks.add_argument("--questions",type=Path,help="Optional labeled benchmark; by default create one from the corpus")
    benchmarks.add_argument("--resume-benchmark",type=Path,help="Keep accepted questions from a saved corpus benchmark and generate only the remainder")
    parser.add_argument("--samples",type=int,default=20)
    parser.add_argument("--k",type=int,default=3)
    parser.add_argument("--depth",type=int,default=20,help="Ranked retrieval depth used for MRR")
    parser.add_argument("--context-k",type=int,default=3,help="How many top-ranked chunks the expert actually receives")
    parser.add_argument("--correctness-threshold",type=float,default=1.0)
    parser.add_argument("--generation-backend",choices=["expert","gemini"],default="expert")
    parser.add_argument("--generation-model",default="gemini-2.5-flash",help="Used only with Gemini generation")
    parser.add_argument("--judge-model",default="gemini-2.5-flash")
    parser.add_argument("--benchmark-model",default="gemini-2.5-flash")
    parser.add_argument("--seed",type=int,default=31)
    parser.add_argument("--allow-partial",action="store_true",help="Permit benchmarks missing reference/relevance labels")
    parser.add_argument("--allow-reranker-fallback",action="store_true")
    parser.add_argument("--ask-key",action="store_true",help="Read Gemini key without echo; never writes it to disk")
    parser.add_argument("--check",action="store_true",help="Check local prerequisites without model downloads or API calls")
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    parser.add_argument("--output",type=Path,default=root/"outputs"/("evaluation_"+stamp))
    config=parser.parse_args()
    if config.samples<1 or min(config.k,config.context_k)<1 or config.depth<max(config.k,config.context_k) or not 0<config.correctness_threshold<=1:
        parser.error("Require samples >= 1, 1 <= k/context-k <= depth, and 0 < correctness-threshold <= 1")
    try:
        from dotenv import load_dotenv
        load_dotenv(root/".env")
    except ImportError:
        pass
    if config.generation_backend=="expert":
        config.generation_model=os.getenv("EXPERT_MODEL_ID","/tmp/mistral-7b-instruct-v0.3.Q4_K_M.gguf")
    if config.ask_key:
        os.environ["GOOGLE_API_KEY"]=getpass.getpass("Gemini API key (hidden): ").strip()
    checks=preflight(config)
    if config.check:
        print(json.dumps(checks,indent=2))
        return 0 if all(checks["dependencies"].values()) and checks["judge_key_present"] and checks["expert_url_present"] is not False and checks["sources_present"] is not False else 1
    if not all(checks["dependencies"].values()):
        parser.error("Missing runtime dependencies. Install: python -m pip install -r evals/full_requirements.txt")
    if not checks["judge_key_present"]:
        parser.error("Set GOOGLE_API_KEY/GEMINI_API_KEY or use --ask-key")
    if checks["expert_url_present"] is False:
        parser.error("Expert mode needs LOCAL_MODEL_URL. To evaluate Gemini generation explicitly choose --generation-backend gemini")
    if checks["sources_present"] is False:
        parser.error("The selected source directory or chunk file does not exist")
    # Authenticate before expensive ingestion; never print the client or the key.
    try:
        print("Checking Gemini judge access...",flush=True)
        ready=Judge(config.judge_model).call("Return JSON ready=true.",Ready,Ready.model_validate)
        if not ready.ready:
            raise ValueError("Judge preflight failed")
        if config.generation_backend=="expert":
            print("Checking Kaggle expert endpoint...",flush=True)
            from openai import OpenAI
            client=OpenAI(base_url=os.environ["LOCAL_MODEL_URL"],api_key=os.getenv("LOCAL_MODEL_API_KEY","not-needed"),timeout=30.0,max_retries=0)
            advertised=client.models.list()
            if advertised.data and config.generation_model not in {m.id for m in advertised.data}:
                parser.error("EXPERT_MODEL_ID is not advertised by the expert endpoint")
        summary=run_workflow(config)
    except Exception as error:
        print("Evaluation stopped:",type(error).__name__)
        detail=safe_api_error(error)
        print("HTTP status:",detail["api_code"])
        print("Reason:",detail["message"])
        print("Inspect run_manifest.json if created. Known API keys are redacted.")
        return 1
    for metric,result in summary["metrics"].items():
        value=f"{result['value']:.4f}" if result["value"] is not None else "Unmeasured"
        print(f"{metric}: {value} ({result['scored']}/{result['attempted']} scored)")
    print("Report:",config.output/"evaluation_report.md")
    return 0 if summary["complete"] else 1


if __name__=="__main__":
    raise SystemExit(main())

