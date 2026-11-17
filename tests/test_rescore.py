"""Scoring recovery must reuse saved outputs and preserve partial judgments."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from evals.judge import Judge
from evals.rescore import rescore


RECORD={"id":"one","question":"What is RRC?","status":"ok",
        "generated_answer":"RRC is Radio Resource Control.",
        "retrieved_context":"RRC is Radio Resource Control.",
        "reference_answer":"Radio Resource Control.","relevant_doc_ids":["a"],
        "retrieved_chunks":[{"id":"a","content":"RRC is Radio Resource Control.","metadata":{}}]}


class RescoreTests(unittest.TestCase):
    def test_judge_configuration_limits_flash_thinking(self):
        config=Judge("gemini-3-flash-preview",client=object()).generation_config({"type":"object"})
        self.assertEqual(config["thinking_config"],{"thinking_level":"low"})
        self.assertNotIn("thinking_config",Judge("gemini-2.5-flash",client=object()).generation_config({}))
        with patch.dict("os.environ",{"EVAL_JUDGE_TIMEOUT_SECONDS":"nan"}):
            with self.assertRaises(ValueError):
                Judge(client=object())

    def test_saved_answers_produce_six_scores_without_regeneration(self):
        calls=[]
        def generate(**kwargs):
            calls.append(kwargs)
            payload=[{"claim":"RRC is Radio Resource Control.","is_supported":True,"reasoning":"Supported"}] if "Extract every atomic" in kwargs["contents"] else {"score":1.0,"reasoning":"Supported"}
            return SimpleNamespace(text=json.dumps(payload))
        client=SimpleNamespace(models=SimpleNamespace(generate_content=generate))
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td);dataset=folder/"traces.json";out=folder/"scores"
            dataset.write_text(json.dumps([RECORD,RECORD]),encoding="utf-8")
            before=dataset.read_bytes()
            (folder/"run_manifest.json").write_text(json.dumps({"generation_backend":"expert","generation_model":"telecom-mistral-lora","k":3,"retrieval_depth":20}))
            summary=rescore(dataset,out,samples=1,client=client)
            self.assertTrue(summary["complete"])
            self.assertEqual(len(summary["metrics"]),6)
            self.assertTrue(all(m["value"]==1.0 for m in summary["metrics"].values()))
            self.assertEqual(len(calls),3)
            self.assertEqual(dataset.read_bytes(),before)
            manifest=json.loads((out/"run_manifest.json").read_text())
            self.assertTrue(manifest["score_only"])
            self.assertEqual(manifest["generation_model"],"telecom-mistral-lora")
            self.assertTrue((out/"evaluation_report.md").exists())

    def test_interruption_preserves_first_judgment(self):
        calls=[]
        def generate(**kwargs):
            calls.append(kwargs)
            if len(calls)>1:
                raise KeyboardInterrupt()
            return SimpleNamespace(text=json.dumps([{"claim":"RRC is Radio Resource Control.","is_supported":True,"reasoning":"Supported"}]))
        client=SimpleNamespace(models=SimpleNamespace(generate_content=generate))
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td);dataset=folder/"traces.json";out=folder/"scores"
            dataset.write_text(json.dumps([RECORD]),encoding="utf-8")
            with self.assertRaises(KeyboardInterrupt):
                rescore(dataset,out,client=client)
            self.assertEqual(json.loads((out/"run_manifest.json").read_text())["state"],"interrupted")
            rows=json.loads((out/"per_question_judgments.json").read_text())
            self.assertEqual(rows[0]["metrics"]["Faithfulness"],1.0)
            self.assertIsNone(rows[0]["metrics"]["Accuracy"])

    def test_provider_failure_stops_and_keeps_status_detail(self):
        calls=[]
        class APIError(Exception):
            code=400
        def generate(**kwargs):
            calls.append(kwargs)
            raise APIError("Invalid request")
        client=SimpleNamespace(models=SimpleNamespace(generate_content=generate))
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td);dataset=folder/"traces.json";out=folder/"scores"
            dataset.write_text(json.dumps([RECORD,RECORD]),encoding="utf-8")
            with self.assertRaises(APIError):
                rescore(dataset,out,client=client)
            self.assertEqual(len(calls),1)
            self.assertEqual(json.loads((out/"run_manifest.json").read_text())["error_detail"]["api_code"],400)
            rows=json.loads((out/"per_question_judgments.json").read_text())
            self.assertEqual(rows[0]["metrics"]["MRR"],1.0)
            self.assertIsNone(rows[0]["metrics"]["Faithfulness"])


if __name__=="__main__":
    unittest.main()

