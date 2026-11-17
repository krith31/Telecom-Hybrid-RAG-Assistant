"""Controlled-service tests; these are not actual project evaluation scores."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from evals.complete import CollectionPipeline, run_workflow, validate_questions
from evals.corpus import document_sections, source_chunks, validate_chunks
from evals.six_metrics import METRICS, evaluate_six, export_six, retrieval_scores


CHUNKS=[{"id":"a","content":"Alpha source: the scheduler uses CQI. "+"Supporting evidence. "*8,
         "metadata":{"spec":"A","section_id":"5"}},
        {"id":"b","content":"Beta source: the timer controls monitoring. "+"Supporting evidence. "*8,
         "metadata":{"spec":"B","section_id":"6"}}]


class Client:
    def __init__(self,fail=False):
        self.fail=fail;self.calls=0
        self.models=SimpleNamespace(generate_content=self.generate)
    def generate(self,**kwargs):
        self.calls+=1
        if self.fail:
            raise ValueError("simulated invalid response")
        prompt=kwargs["contents"]
        data=json.loads(prompt.split("INPUT_DATA:\n",1)[1])
        if "Create one clear telecom question" in prompt:
            source=data["source"]
            result={"question":"What does "+source[:30]+" describe?",
                    "reference_answer":source[:50],"evidence_quote":source[:30]}
        elif "Extract every atomic" in prompt:
            result=[{"claim":"Claim A","is_supported":True,"reasoning":"fixture"},
                    {"claim":"Claim B","is_supported":False,"reasoning":"fixture"}]
        elif "Compare the answer against" in prompt:
            result={"score":1.0,"reasoning":"fixture reference match"}
        else:
            result={"score":.75,"reasoning":"fixture partially relevant"}
        return SimpleNamespace(text=json.dumps(result))


class Tokenizer:
    def encode(self,text,add_special_tokens=False): return list(text)
    def decode(self,ids,skip_special_tokens=True): return "".join(ids)


class Store:
    def __init__(self): self.collections={}
    def ingest_vector_chunks(self,path,collection_name):
        self.collections[collection_name]=json.loads(Path(path).read_text(encoding="utf-8"))


class Pipeline:
    def __init__(self):
        self.vector_store=Store()
        self.reranker=SimpleNamespace(model=object())
    def retrieve_context(self,query,top_k,collection_name):
        return list(reversed(self.vector_store.collections[collection_name]))[:top_k]


def config(output,chunks):
    return SimpleNamespace(output=output,specs=None,chunks=chunks,collection="telecom_specs",
                           generation_backend="gemini",generation_model="fixture",judge_model="fixture",
                           benchmark_model="fixture",k=1,depth=2,context_k=1,samples=2,
                           correctness_threshold=1.0,seed=31,questions=None,allow_partial=False,
                           allow_reranker_fallback=False)


class SixMetricTests(unittest.TestCase):
    def test_ranked_metrics(self):
        m=retrieval_scores(["x","a","b"],["a","b"],2,3)
        self.assertEqual(m,{"MRR":.5,"Top-k Accuracy":1.0,"Recall":.5})
    def test_hit_is_not_precision(self):
        self.assertEqual(retrieval_scores(["a","x","y"],["a"],3,3)["Top-k Accuracy"],1)
    def test_depth_cutoff(self):
        self.assertEqual(retrieval_scores(["x","a"],["a"],1,1)["MRR"],0)
    def test_duplicates_do_not_inflate_recall(self):
        self.assertEqual(retrieval_scores(["a","a","a"],["a","b"],3,3)["Recall"],.5)
    def test_missing_gold_is_not_zero(self):
        self.assertTrue(all(v is None for v in retrieval_scores(["a"],None,1,1).values()))
    def test_invalid_k(self):
        with self.assertRaises(ValueError): retrieval_scores([],[],4,3)
    def test_six_exact_names(self):
        self.assertEqual(METRICS,("MRR","Top-k Accuracy","Accuracy","Recall","Relevance","Faithfulness"))
    def test_binary_accuracy_distinct_from_rating(self):
        row={"id":"1","question":"q","generated_answer":"a","retrieved_context":"a",
             "retrieved_chunks":[CHUNKS[0]],"reference_answer":"a","relevant_doc_ids":["a"]}
        results,summary=evaluate_six([row],k=1,depth=1,client=Client())
        self.assertEqual(summary["metrics"]["Accuracy"]["value"],1)
        self.assertEqual(summary["metrics"]["Relevance"]["value"],.75)
        self.assertEqual(summary["metrics"]["Faithfulness"]["value"],.5)
    def test_judge_errors_have_null_scores(self):
        row={"question":"q","generated_answer":"a","retrieved_context":"a","reference_answer":"ref"}
        _,summary=evaluate_six([row],client=Client(fail=True))
        for metric in ("Accuracy","Relevance","Faithfulness"):
            self.assertIsNone(summary["metrics"][metric]["value"])
            self.assertEqual(summary["metrics"][metric]["scored"],0)
        self.assertFalse(summary["complete"])
    def test_unknown_relevance_id_rejected(self):
        with self.assertRaises(ValueError):
            validate_questions([{"id":"1","question":"q","reference_answer":"ref","relevant_doc_ids":["unknown"]}],{"a"})
    def test_missing_refs_require_explicit_partial(self):
        rows=[{"id":"1","question":"q"}]
        with self.assertRaises(ValueError): validate_questions(rows,{"a"})
        self.assertEqual(validate_questions(rows,{"a"},True),rows)
    def test_invalid_record_retained(self):
        results,summary=evaluate_six(["bad"],client=Client())
        self.assertEqual(results[0]["status"],"invalid_record")
        self.assertEqual(summary["attempted"],1)
    def test_docs_preserve_table_owner(self):
        from docx import Document
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"spec.docx"
            doc=Document();doc.add_paragraph("5 First")
            table=doc.add_table(rows=1,cols=1);table.cell(0,0).text="belongs to section five"
            doc.add_paragraph("6 Second");doc.add_paragraph("Avoid interference.");doc.save(path)
            sections=document_sections(path)
            self.assertIn("belongs to section five",sections[0]["text"])
            self.assertEqual(sections[0]["section_id"],"5")
            self.assertIn("Avoid interference.",sections[1]["text"])
    def test_cross_spec_ids_unique(self):
        with tempfile.TemporaryDirectory() as td:
            for name in ("one.txt","two.txt"):
                (Path(td)/name).write_text("A long enough telecom statement. "*8,encoding="utf-8")
            chunks=source_chunks(td,Tokenizer(),max_tokens=100,overlap=10)
            self.assertEqual(len({c["id"] for c in chunks}),len(chunks))
            self.assertTrue(all(len(c["content"])<=100 for c in chunks))
    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError): validate_chunks([CHUNKS[0],CHUNKS[0]])
    def test_workflow_start_to_finish_with_controlled_services(self):
        with tempfile.TemporaryDirectory() as td:
            source=Path(td)/"chunks.json";source.write_text(json.dumps(CHUNKS),encoding="utf-8")
            output=Path(td)/"run"
            prompts=[]
            def generate(prompt):
                prompts.append(prompt)
                return "Fixture answer"
            summary=run_workflow(config(output,source),Pipeline(),generate,Client())
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["metrics"]["MRR"]["value"],.75)
            self.assertEqual(summary["metrics"]["Top-k Accuracy"]["value"],.5)
            self.assertEqual(summary["metrics"]["Recall"]["value"],.5)
            self.assertEqual(len(prompts),2)
            for name in ("traces.json","benchmark.json","corpus_chunks.json","run_manifest.json",
                         "final_evaluation.json","final_evaluation.csv","evaluation_report.md",
                         "per_question_scores.csv","per_question_judgments.json"):
                self.assertTrue((output/name).exists(),name)
            traces=json.loads((output/"traces.json").read_text(encoding="utf-8"))
            self.assertEqual(len(traces[0]["retrieved_chunks"]),1)
            self.assertEqual(len(traces[0]["retrieval_ranking"]),2)
            manifest=json.loads((output/"run_manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["injected_services"])
            self.assertNotIn("api_key",manifest)
            with self.assertRaises(ValueError): run_workflow(config(output,source),Pipeline(),generate,Client())
    def test_missing_sources_fail_with_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            output=Path(td)/"run";conf=config(output,Path(td)/"missing.json")
            with self.assertRaises(FileNotFoundError): run_workflow(conf,Pipeline(),lambda x:"a",Client())
            self.assertEqual(json.loads((output/"run_manifest.json").read_text())["state"],"failed")


if __name__=="__main__":
    unittest.main()
