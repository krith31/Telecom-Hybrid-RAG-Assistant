import os
import json
import logging
from typing import List, Dict, Any
from google import genai
from google.genai import types
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class ClaimVerification(BaseModel):
    claim: str
    is_supported: bool
    reasoning: str

class FaithfulnessResult(BaseModel):
    score: float
    verifications: List[ClaimVerification]

class FaithfulnessEvaluator:
    """
    Evaluates whether the generated answer is strictly supported by the retrieved context.
    Penalizes hallucinations.
    """
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        # Relies on GOOGLE_API_KEY being set in the environment
        self.client = genai.Client()
        self.model_name = model_name

    def evaluate(self, generated_answer: str, retrieved_context: str) -> FaithfulnessResult:
        """
        Calculates the Faithfulness score.
        1. Extracts claims from the generated answer.
        2. Verifies each claim against the retrieved context.
        """
        prompt = f"""
You are an expert evaluator. Your task is to determine if the claims made in a generated answer are completely supported by the provided retrieved context.

--- RETRIEVED CONTEXT ---
{retrieved_context}

--- GENERATED ANSWER ---
{generated_answer}

--- INSTRUCTIONS ---
1. Break down the GENERATED ANSWER into discrete, verifiable factual claims.
2. For each claim, check if it is explicitly or implicitly supported by the RETRIEVED CONTEXT.
3. If a claim contains information not present in the context, it is NOT supported (hallucination).
4. Output a JSON list of objects, each with 'claim', 'is_supported' (boolean), and 'reasoning'.
"""
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=list[ClaimVerification],
                    temperature=0.1,
                ),
            )
            
            # The SDK handles the parsing into the Pydantic schema structure (or returns JSON string to parse)
            verifications_data = json.loads(response.text)
            verifications = [ClaimVerification(**v) for v in verifications_data]
            
            if not verifications:
                return FaithfulnessResult(score=1.0, verifications=[])
            
            supported_count = sum(1 for v in verifications if v.is_supported)
            score = supported_count / len(verifications)
            
            return FaithfulnessResult(score=score, verifications=verifications)
            
        except Exception as e:
            logger.error(f"Error during faithfulness evaluation: {e}")
            return FaithfulnessResult(score=0.0, verifications=[])

if __name__ == "__main__":
    # Test example
    evaluator = FaithfulnessEvaluator()
    context = "The gNB scheduler allocates PRBs based on the active BSR and reported CQI. If CQI drops, modulation is lowered."
    answer = "The gNB scheduler relies on BSR and CQI to allocate PRBs. It will lower the modulation if the CQI drops. It also uses the AMF to allocate resources."
    
    result = evaluator.evaluate(answer, context)
    print(f"Faithfulness Score: {result.score}")
    for v in result.verifications:
        print(f"- Claim: {v.claim}\n  Supported: {v.is_supported}\n  Reasoning: {v.reasoning}\n")
