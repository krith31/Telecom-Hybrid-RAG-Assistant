"""Create silver question/reference pairs from the actual source chunks."""
import argparse
import json
from pathlib import Path
import random
from pydantic import BaseModel, ConfigDict, Field
from .capture import write_json
from .judge import Judge, evaluation_prompt, safe_api_error

class SourceQuestion(BaseModel):
    model_config=ConfigDict(extra="forbid")
    question: str=Field(min_length=10)
    reference_answer: str=Field(min_length=10)
    evidence_quote: str=Field(min_length=10)

def build_benchmark(chunks, output, count=20, seed=31, model="gemini-2.5-flash", client=None, existing=None):
    if count<1:
        raise ValueError("count must be positive")
    if not isinstance(chunks,list) or not chunks:
        raise ValueError("Chunks must be a nonempty JSON list")
    ids=[c.get("id") for c in chunks if isinstance(c,dict)]
    if len(ids)!=len(chunks) or any(not isinstance(x,str) or not x for x in ids):
        raise ValueError("Every chunk needs a nonempty string id")
    if len(set(ids))!=len(ids):
        raise ValueError("Duplicate chunk IDs; make IDs globally unique and rebuild the index first")
    candidates=[c for c in chunks if isinstance(c.get("content"),str) and len(c["content"].strip())>=80]
    if len(candidates)<count:
        raise ValueError(f"Need {count} source chunks of at least 80 characters; found {len(candidates)}")
    random.Random(seed).shuffle(candidates)
    judge=Judge(model,client)
    records=[]; failures=[]
    if existing is not None:
        if not isinstance(existing,list) or len(existing)>count:
            raise ValueError("Resumed benchmark must be a list no longer than requested samples")
        seen_ids=set();seen_questions=set()
        for record in existing:
            SourceQuestion.model_validate({key:record.get(key) for key in SourceQuestion.model_fields})
            if record.get("id") in seen_ids or record["question"].casefold() in seen_questions:
                raise ValueError("Resumed benchmark has duplicate IDs or questions")
            relevant=record.get("relevant_doc_ids")
            if not relevant or any(id_ not in ids for id_ in relevant):
                raise ValueError("Resumed benchmark references a different corpus")
            if record.get("reference_source")!="synthetic_from_corpus":
                raise ValueError("Resume expects the generated corpus benchmark")
            seen_ids.add(record.get("id"));seen_questions.add(record["question"].casefold())
            records.append(record)
        print(f"Resuming benchmark: {len(records)}/{count} accepted questions already saved",flush=True)
    def checkpoint():
        write_json(output,records)
        write_json(str(output)+".build.json",{"requested":count,"generated":len(records),"failures":failures,
                                            "model":model,"seed":seed,"label_type":"silver"})
    checkpoint()
    used={id_ for record in records for id_ in record.get("relevant_doc_ids",[])}
    for chunk in candidates:
        if len(records)>=count:
            break
        if chunk["id"] in used:
            continue
        prompt=evaluation_prompt(
            "Create one clear telecom question and a concise reference answer fully entailed by this source. "
            "Do not ask for unstated facts or an answer outside this source. Include an exact contiguous "
            "evidence_quote copied from the source, supporting the answer. "
            "Return question, reference_answer, evidence_quote. No outside knowledge.",source=chunk["content"])
        try:
            item=judge.call(prompt,SourceQuestion,SourceQuestion.model_validate)
            if item.evidence_quote not in chunk["content"]:
                raise ValueError("Evidence quote is absent from source")
            if any(r["question"].casefold()==item.question.casefold() for r in records):
                raise ValueError("Duplicate generated question")
            next_id=len(records)+1
            while f"SILVER_{next_id:03d}" in {r["id"] for r in records}:
                next_id+=1
            records.append({"id":f"SILVER_{next_id:03d}",**item.model_dump(),
                            "relevant_doc_ids":[chunk["id"]],"reference_source":"synthetic_from_corpus",
                            "reference_reviewed":False,"expected_abstention":False,
                            "source_metadata":chunk.get("metadata",{}),"generator_model":model,"seed":seed,
                            "label_limitations":"Reference needs review; relevant IDs are known positives, not exhaustive."})
        except Exception as error:
            code=getattr(error,"code",getattr(error,"status_code",None))
            failures.append({"chunk_id":chunk["id"],"error":type(error).__name__,
                             "api_code":code if isinstance(code,int) else None,
                             "detail":safe_api_error(error)["message"]})
            checkpoint()
            # Retrying a different source cannot fix a rejected API request.
            # Judge.call already performed bounded retries for transient errors.
            if code is not None or not isinstance(error,(ValueError,TypeError)):
                raise RuntimeError(f"Benchmark API request failed ({type(error).__name__}, HTTP {code}); stopped. Partial output saved.") from error
            if len(failures)>=max(5,count*3):
                raise RuntimeError("Benchmark validation failure limit reached; partial output saved.") from error
        checkpoint()
        print(f"Benchmark questions: {len(records)}/{count}; rejected candidates: {len(failures)}",flush=True)
    if len(records)<count:
        raise RuntimeError(f"Only {len(records)}/{count} questions generated; partial output and failures saved")
    return records

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    root=Path(__file__).resolve().parents[1]
    parser.add_argument("--chunks",type=Path,default=root/"data/chunks/vector_chunks.json")
    parser.add_argument("--output",type=Path,default=root/"evals/corpus_benchmark.json")
    parser.add_argument("--count",type=int,default=20)
    parser.add_argument("--seed",type=int,default=31)
    parser.add_argument("--model",default="gemini-2.5-flash")
    args=parser.parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv(root/".env")
    except ImportError:
        pass
    chunks=json.loads(args.chunks.read_text(encoding="utf-8"))
    records=build_benchmark(chunks,args.output,args.count,args.seed,args.model)
    print(f"Saved {len(records)} silver questions to {args.output}")

if __name__=="__main__":
    main()

