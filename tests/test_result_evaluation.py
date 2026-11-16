"""Offline tests use fake services; their results are not RAG quality scores."""
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from evals.capture import extract_query, generate_evaluation_dataset
from evals.result_accuracy import AccuracyEvaluator
from evals.result_faithfulness import FaithfulnessEvaluator
from evals.judge import Judge
from evals.metrics import citation_validity, retrieval_metrics
from evals.run import evaluate_records, save_report, summarize

CHUNK={"id":"38.321::5.1::0","content":"The scheduler allocates PRBs using BSR and CQI.",
       "metadata":{"spec":"38.321","section_id":"5.1"}}

class Client:
    def __init__(self,payload=None,error=None):
        self.payload=payload;self.error=error;self.calls=0
        self.models=SimpleNamespace(generate_content=self.generate)
    def generate(self,**kwargs):
        self.calls+=1
        if self.error:
            raise self.error
        payload=self.payload(kwargs) if callable(self.payload) else self.payload
        return SimpleNamespace(text=json.dumps(payload))

def dispatch(kwargs):
    text=kwargs["contents"]
    if "Extract every atomic" in text:
        return [{"claim":"a","is_supported":True,"reasoning":"fixture"}]
    if "For EVERY retrieved" in text:
        return {"chunks":[{"chunk_index":0,"relevant":True,"reasoning":"fixture"}],"reference_claims":[]}
    if "Determine whether the answer abstains" in text:
        return {"abstained":False,"citation_support":1.0,"citation_completeness":1.0,"reasoning":"fixture"}
    return {"score":1.0,"reasoning":"fixture"}

class ResultEvaluationTests(unittest.TestCase):
    def test_legacy_query(self):
        self.assertEqual(extract_query({"full_prompt_text":"Report\nDIAGNOSTIC QUESTION:\nWhat is BSR?"}),"What is BSR?")

    def test_capture_and_preserve_failure(self):
        pipeline=SimpleNamespace(retrieve_context=lambda query,top_k:[CHUNK])
        with tempfile.TemporaryDirectory() as td:
            ip=Path(td)/"in.json";op=Path(td)/"out.json"
            ip.write_text(json.dumps([{},{"question":"q"}]),encoding="utf-8")
            rows=generate_evaluation_dataset(ip,op,2,pipeline,lambda p:"answer")
            self.assertEqual([r["status"] for r in rows],["invalid_input","ok"])
            self.assertEqual(rows[1]["retrieved_chunks"],[CHUNK])
            self.assertEqual(json.loads(op.read_text(encoding="utf-8")),rows)

    def test_empty_context_captures_abstention(self):
        pipeline=SimpleNamespace(retrieve_context=lambda query,top_k:[])
        with tempfile.TemporaryDirectory() as td:
            ip=Path(td)/"in.json"
            ip.write_text(json.dumps([{"question":"q","expected_abstention":True}]),encoding="utf-8")
            rows=generate_evaluation_dataset(ip,Path(td)/"out.json",1,pipeline,lambda p:"Insufficient evidence.")
            self.assertEqual(rows[0]["status"],"ok")
            self.assertTrue(rows[0]["retrieval_empty"])

    def test_faithfulness_fraction(self):
        payload=[{"claim":"a","is_supported":True,"reasoning":"supported"},
                 {"claim":"b","is_supported":False,"reasoning":"absent"}]
        result=FaithfulnessEvaluator(client=Client(payload)).evaluate("a b","a")
        self.assertEqual(result.score,.5)

    def test_no_claims_not_perfect(self):
        result=FaithfulnessEvaluator(client=Client([])).evaluate("Cannot answer.","")
        self.assertEqual(result.status,"no_claims")
        self.assertIsNone(result.score)

    def test_empty_answer_no_judge_call(self):
        client=Client([])
        result=FaithfulnessEvaluator(client=client).evaluate("","context")
        self.assertIsNone(result.score)
        self.assertEqual(client.calls,0)

    def test_outage_not_zero(self):
        result=FaithfulnessEvaluator(client=Client(error=ValueError())).evaluate("a","a")
        self.assertEqual(result.status,"error")
        self.assertIsNone(result.score)

    def test_out_of_range_rejected(self):
        result=AccuracyEvaluator(client=Client({"score":9.0,"reasoning":"bad"})).evaluate("q","a")
        self.assertEqual(result.status,"error")

    def test_distinct_answer_metrics(self):
        judge=AccuracyEvaluator(client=Client({"score":.5,"reasoning":"partial"}))
        self.assertEqual(judge.evaluate("q","a").metric,"answer_relevance")
        self.assertEqual(judge.evaluate("q","a","ref").metric,"answer_correctness")

    def test_retry_bounded(self):
        client=Client(error=TimeoutError())
        with self.assertRaises(TimeoutError):
            Judge(client=client,max_attempts=2,sleep=lambda x:None).call("p",dict,lambda x:x)
        self.assertEqual(client.calls,2)

    def test_ranking_formula(self):
        result=retrieval_metrics(["a","x","b"],["a","b"],3)
        self.assertAlmostEqual(result["precision@3"],2/3)
        self.assertEqual(result["recall@3"],1)
        self.assertEqual(result["mrr@3"],1)
        self.assertAlmostEqual(result["ndcg@3"],1.5/(1+1/math.log2(3)))
        self.assertAlmostEqual(result["ap@3"],(1+2/3)/2)

    def test_missing_labels_not_zero(self):
        self.assertEqual(retrieval_metrics(["a"],None)["status"],"unlabeled")
        self.assertEqual(retrieval_metrics([],[])["status"],"no_relevant_documents")

    def test_duplicate_ranks(self):
        self.assertAlmostEqual(retrieval_metrics(["a","a","a"],["a"],3)["precision@3"],1/3)

    def test_citation_identity(self):
        result=citation_validity("a [Spec: 38.321 | Clause 5.1] b [Spec: fake | Clause 9]",[CHUNK])
        self.assertEqual(result["source_validity"],.5)

    def test_report_and_coverage(self):
        rows=[{"question":"q","generated_answer":"a","retrieved_context":"a","retrieved_chunks":[CHUNK]},
              {"question":"q","status":"generation_error","retrieved_chunks":[]}]
        results=evaluate_records(rows,client=Client(dispatch))
        summary=summarize(results)
        self.assertEqual(summary["metrics"]["faithfulness"]["coverage"],.5)
        self.assertEqual(summary["pipeline_success_rate"],.5)
        self.assertIsNone(summary["metrics"]["answer_correctness"]["mean"])
        with tempfile.TemporaryDirectory() as td:
            save_report(td,results,summary,{"judge_model":"fake","mode":"offline_fixture"})
            self.assertTrue((Path(td)/"report.md").is_file())
            self.assertTrue((Path(td)/"scores.csv").is_file())

    def test_invalid_record_retained(self):
        results=evaluate_records(["invalid"],client=Client(dispatch))
        self.assertEqual(results[0]["status"],"invalid_record")

    def test_empty_dataset_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_records([])

if __name__=="__main__":
    unittest.main()
