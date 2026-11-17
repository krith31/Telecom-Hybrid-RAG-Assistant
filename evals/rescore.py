"""Compute the six metrics from saved answers, without retrieval or generation."""
import argparse
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path

from .capture import write_json
from .judge import safe_api_error
from .schema import CapturedAnswer
from .six_metrics import evaluate_six, export_six


def rescore(dataset, output, judge_model="gemini-3-flash-preview", samples=None,
            k=None, depth=None, correctness_threshold=None, client=None):
    dataset,output=Path(dataset),Path(output)
    raw=dataset.read_bytes()
    records=json.loads(raw.decode("utf-8"))
    if not isinstance(records,list) or not records:
        raise ValueError("Saved answers must be a nonempty JSON list")
    if samples is not None:
        if samples<1:
            raise ValueError("Samples must be positive")
        records=records[:samples]
    for record in records:
        CapturedAnswer.model_validate(record)
    source_manifest=dataset.parent/"run_manifest.json"
    source=json.loads(source_manifest.read_text(encoding="utf-8")) if source_manifest.exists() else {}
    k=source.get("k",3) if k is None else k
    depth=source.get("retrieval_depth",20) if depth is None else depth
    correctness_threshold=source.get("correctness_threshold",1.0) if correctness_threshold is None else correctness_threshold
    if k<1 or depth<k or not 0<correctness_threshold<=1:
        raise ValueError("Require 1 <= k <= depth and 0 < correctness-threshold <= 1")
    if (output/"run_manifest.json").exists():
        raise ValueError("Output already contains a run; choose a new --output directory")
    manifest={key:source[key] for key in ("generation_backend","generation_model","collection",
              "corpus_sha256","corpus_chunks","context_k","scope") if key in source}
    manifest.update(state="running",stage="judge_answers",score_only=True,
                    source_dataset=str(dataset.resolve()),source_dataset_sha256=hashlib.sha256(raw).hexdigest(),
                    judge_model=judge_model,requested_samples=len(records),k=k,retrieval_depth=depth,
                    correctness_threshold=correctness_threshold,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    judge_timeout_seconds=float(os.getenv("EVAL_JUDGE_TIMEOUT_SECONDS","180")),
                    judge_thinking_level="low" if judge_model.startswith("gemini-3") and "flash" in judge_model else None)
    write_json(output/"run_manifest.json",manifest)
    try:
        results,summary=evaluate_six(records,k,depth,correctness_threshold,judge_model,client,
                                      output/"per_question_judgments.json",fail_fast=True)
        manifest.update(state="complete" if summary["complete"] else "partial",stage="finished",
                        finished_at=datetime.now(timezone.utc).isoformat())
        export_six(output,results,summary,manifest)
        write_json(output/"run_manifest.json",manifest)
        return summary
    except KeyboardInterrupt:
        manifest.update(state="interrupted")
        write_json(output/"run_manifest.json",manifest)
        raise
    except Exception as error:
        manifest.update(state="failed",error_detail=safe_api_error(error))
        write_json(output/"run_manifest.json",manifest)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--judge-model",default="gemini-3-flash-preview")
    parser.add_argument("--samples",type=int)
    parser.add_argument("--k",type=int)
    parser.add_argument("--depth",type=int)
    parser.add_argument("--correctness-threshold",type=float)
    parser.add_argument("--ask-key",action="store_true")
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    try:
        from dotenv import load_dotenv
        load_dotenv(root/".env")
    except ImportError:
        pass
    if args.ask_key:
        os.environ["GOOGLE_API_KEY"]=getpass.getpass("Gemini API key (hidden): ").strip()
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        parser.error("Set GOOGLE_API_KEY/GEMINI_API_KEY or use --ask-key")
    output=args.output or root/"outputs"/("rescore_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    try:
        print("Scoring saved answers only; no indexing, retrieval or model generation.",flush=True)
        summary=rescore(args.dataset,output,args.judge_model,args.samples,args.k,args.depth,args.correctness_threshold)
    except KeyboardInterrupt:
        print("Scoring interrupted. Saved answers are unchanged; available judgments were checkpointed.")
        return 130
    except Exception as error:
        detail=safe_api_error(error)
        print("Scoring stopped:",detail["error_type"],"HTTP",detail["api_code"])
        print("Reason:",detail["message"])
        return 1
    for metric,result in summary["metrics"].items():
        value=f"{result['value']:.4f}" if result["value"] is not None else "Unmeasured"
        print(f"{metric}: {value} ({result['scored']}/{result['attempted']} scored)")
    print("Report:",output/"evaluation_report.md")
    return 0 if summary["complete"] else 1


if __name__=="__main__":
    raise SystemExit(main())

