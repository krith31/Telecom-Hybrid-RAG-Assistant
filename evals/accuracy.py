import os
import json
import logging
from typing import Optional
from google import genai
from google.genai import types
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class AccuracyResult(BaseModel):
    score: float
    reasoning: str

class AccuracyEvaluator:
    """
    Evaluates Answer Accuracy. 
    If a reference_answer is provided, it compares the generated answer against the reference.
    If no reference_answer is provided, it acts as an Answer Relevance evaluator (does the answer address the question?).
    """
    def __init__(self, model_name: str = "gemini-3.8-flash"):
        self.client = genai.Client()
        self.model_name = model_name

    def evaluate(self, question: str, generated_answer: str, reference_answer: Optional[str] = None) -> AccuracyResult:
        if reference_answer:
            return self._evaluate_correctness(question, generated_answer, reference_answer)
        else:
            return self._evaluate_relevance(question, generated_answer)

    def _evaluate_correctness(self, question: str, generated_answer: str, reference_answer: str) -> AccuracyResult:
        """Evaluates semantic equivalence to a ground-truth reference answer."""
        prompt = f"""
You are an expert grading a telecom exam.
Question: {question}
Reference Correct Answer: {reference_answer}
Student's Answer: {generated_answer}

Is the student's answer factually correct and semantically equivalent to the reference answer?
Output a JSON object with 'score' (1.0 for correct/equivalent, 0.5 for partially correct, 0.0 for incorrect) and 'reasoning'.
"""
        return self._call_llm(prompt)

    def _evaluate_relevance(self, question: str, generated_answer: str) -> AccuracyResult:
        """Evaluates if the answer directly addresses the question when no ground-truth is available."""
        prompt = f"""
You are an expert evaluating AI-generated responses.
Question: {question}
Generated Answer: {generated_answer}

Does the generated answer directly, comprehensively, and clearly address the question being asked? 
Ignore whether it is factually true (assume it is); focus only on RELEVANCE to the question prompt.
Output a JSON object with 'score' (1.0 for highly relevant/direct, 0.5 for tangential/partial, 0.0 for irrelevant) and 'reasoning'.
"""
        return self._call_llm(prompt)

    def _call_llm(self, prompt: str) -> AccuracyResult:
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=AccuracyResult,
                    temperature=0.0,
                ),
            )
            result_data = json.loads(response.text)
            return AccuracyResult(**result_data)
        except Exception as e:
            logger.error(f"Error during accuracy evaluation: {e}")
            return AccuracyResult(score=0.0, reasoning=f"Error evaluating: {e}")

if __name__ == "__main__":
    evaluator = AccuracyEvaluator()
    q = "What causes an MCS fallback?"
    ans = "MCS fallback is typically caused by channel degradation, increased interference, or a drop in CQI."
    
    # Test Answer Relevance (No reference)
    res_rel = evaluator.evaluate(q, ans)
    print(f"Relevance Score: {res_rel.score}\nReasoning: {res_rel.reasoning}\n")
    
    # Test Correctness (With reference)
    ref = "It is caused by reduced Signal to Interference and Noise Ratio (SINR)."
    res_cor = evaluator.evaluate(q, ans, reference_answer=ref)
    print(f"Correctness Score: {res_cor.score}\nReasoning: {res_cor.reasoning}")
