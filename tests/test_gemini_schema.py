"""Regression coverage for Gemini schema compatibility and bounded failures."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from pydantic import TypeAdapter, ValidationError
from evals.build_benchmark import SourceQuestion, build_benchmark
from evals.judge import Judge, gemini_response_schema, safe_api_error
from evals.result_faithfulness import ClaimVerification


class SchemaTests(unittest.TestCase):
    def test_pacing_is_shared_between_judge_instances(self):
        sleeps=[]
        with patch("evals.judge._NEXT_REQUEST",0),patch("evals.judge.time.monotonic",return_value=100):
            Judge(interval=15,sleep=sleeps.append).pace()
            Judge(interval=15,sleep=sleeps.append).pace()
        self.assertEqual(sleeps,[15])

    def test_error_details_redact_key_and_preserve_quota_reason(self):
        class APIError(Exception):
            code=429
        error=APIError("Quota exhausted for api-secret Bearer another-secret")
        with patch.dict("os.environ",{"GOOGLE_API_KEY":"api-secret"}):
            detail=safe_api_error(error)
        self.assertEqual(detail["api_code"],429)
        self.assertNotIn("api-secret",detail["message"])
        self.assertNotIn("another-secret",detail["message"])
        self.assertIn("Quota exhausted",detail["message"])

    def test_resume_preserves_existing_and_only_generates_remainder(self):
        existing={"id":"SILVER_001","question":"What does the first source describe?",
                  "reference_answer":"A telecom source description.","evidence_quote":"Telecom first evidence.",
                  "relevant_doc_ids":["a"],"reference_source":"synthetic_from_corpus"}
        calls=[]
        source="Telecom second evidence. "*8
        def generate(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(text=json.dumps({"question":"What does the second source describe?",
                "reference_answer":"The second telecom source.","evidence_quote":"Telecom second evidence."}))
        client=SimpleNamespace(models=SimpleNamespace(generate_content=generate))
        chunks=[{"id":"a","content":"Telecom first evidence. "*8},{"id":"b","content":source}]
        with tempfile.TemporaryDirectory() as td:
            records=build_benchmark(chunks,Path(td)/"benchmark.json",count=2,client=client,existing=[existing])
        self.assertEqual(records[0],existing)
        self.assertEqual(len(calls),1)
        self.assertEqual(records[1]["id"],"SILVER_002")
        self.assertEqual(records[1]["relevant_doc_ids"],["b"])

    def test_strict_constraints_stay_local(self):
        wire=gemini_response_schema(SourceQuestion)
        self.assertNotIn("additionalProperties",wire)
        self.assertNotIn("minLength",wire["properties"]["question"])
        self.assertEqual(wire["required"],["question","reference_answer","evidence_quote"])
        with self.assertRaises(ValidationError):
            SourceQuestion.model_validate({"question":"short","reference_answer":"reference long enough",
                                          "evidence_quote":"evidence long enough","extra":True})

    def test_nested_list_schema_resolves_refs(self):
        wire=gemini_response_schema(list[ClaimVerification])
        encoded=json.dumps(wire)
        self.assertNotIn("additionalProperties",encoded)
        self.assertNotIn("$ref",encoded)
        self.assertNotIn("$defs",encoded)
        self.assertEqual(wire["items"]["properties"]["is_supported"]["type"],"boolean")

    def test_server_receives_compatible_schema_and_output_still_validated(self):
        def generate(**kwargs):
            self.assertNotIn("additionalProperties",json.dumps(kwargs["config"]["response_schema"]))
            return SimpleNamespace(text='[{"claim":"Claim","is_supported":"false","reasoning":"reason"}]')
        client=SimpleNamespace(models=SimpleNamespace(generate_content=generate))
        with self.assertRaises(ValidationError):
            Judge(client=client).call("p",list[ClaimVerification],TypeAdapter(list[ClaimVerification]).validate_python)

    def test_http400_stops_after_one_candidate_and_saves_checkpoint(self):
        calls=[]
        class APIError(Exception):
            code=400
        def generate(**kwargs):
            calls.append(kwargs)
            raise APIError("Rejected schema")
        client=SimpleNamespace(models=SimpleNamespace(generate_content=generate))
        chunks=[{"id":str(i),"content":"Telecom source evidence. "*8} for i in range(30)]
        with tempfile.TemporaryDirectory() as td:
            output=Path(td)/"benchmark.json"
            with self.assertRaisesRegex(RuntimeError,"HTTP 400"):
                build_benchmark(chunks,output,count=20,client=client)
            self.assertEqual(len(calls),1)
            checkpoint=json.loads(Path(str(output)+".build.json").read_text())
            self.assertEqual(checkpoint["failures"][0]["api_code"],400)
            self.assertEqual(checkpoint["generated"],0)

    def test_bad_outputs_have_candidate_budget(self):
        calls=[]
        def generate(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(text='{}')
        client=SimpleNamespace(models=SimpleNamespace(generate_content=generate))
        chunks=[{"id":str(i),"content":"Telecom source evidence. "*8} for i in range(30)]
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(RuntimeError,"failure limit"):
                build_benchmark(chunks,Path(td)/"benchmark.json",count=1,client=client)
        self.assertEqual(len(calls),5)


if __name__=="__main__":
    unittest.main()

