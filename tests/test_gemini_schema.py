"""Regression coverage for Gemini schema compatibility and bounded failures."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from pydantic import TypeAdapter, ValidationError
from evals.build_benchmark import SourceQuestion, build_benchmark
from evals.judge import Judge, gemini_response_schema
from evals.result_faithfulness import ClaimVerification


class SchemaTests(unittest.TestCase):
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

