"""Result judge kept separate from the repository's original evaluator."""
from typing import List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, StrictBool, TypeAdapter
from .judge import Judge, evaluation_prompt

class ClaimVerification(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim: str = Field(min_length=1)
    is_supported: StrictBool
    reasoning: str = Field(min_length=1)

class FaithfulnessResult(BaseModel):
    status: Literal["ok", "empty_answer", "no_claims", "error"]
    score: Optional[float] = Field(default=None, ge=0, le=1)
    verifications: List[ClaimVerification] = Field(default_factory=list)
    error: Optional[str] = None

class FaithfulnessEvaluator:
    def __init__(self, model_name="gemini-2.5-flash", client=None, max_attempts=3):
        self.judge = Judge(model_name, client, max_attempts)

    def evaluate(self, generated_answer: str, retrieved_context: str) -> FaithfulnessResult:
        if not generated_answer.strip():
            return FaithfulnessResult(status="empty_answer")
        prompt = evaluation_prompt(
            "Extract every atomic factual claim from the answer. Split compound statements. "
            "Determine whether supplied context entails each claim. External knowledge is forbidden. "
            "Unsupported or contradictory claims get is_supported=false. Return a JSON list of "
            "claim, is_supported, reasoning. A nonfactual abstention may have no claims.",
            answer=generated_answer, context=retrieved_context)
        try:
            claims = self.judge.call(prompt, list[ClaimVerification],
                                     TypeAdapter(list[ClaimVerification]).validate_python)
            if not claims:
                return FaithfulnessResult(status="no_claims")
            return FaithfulnessResult(status="ok", score=sum(c.is_supported for c in claims)/len(claims),
                                      verifications=claims)
        except Exception as error:
            return FaithfulnessResult(status="error", error=type(error).__name__)
