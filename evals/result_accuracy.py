"""Result judge kept separate from the repository's original evaluator."""
from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict, Field
from .judge import Judge, evaluation_prompt

class JudgeScore(BaseModel):
    model_config = ConfigDict(extra="forbid")
    score: float = Field(ge=0, le=1, strict=True, allow_inf_nan=False)
    reasoning: str = Field(min_length=1)

class AccuracyResult(BaseModel):
    metric: Literal["answer_correctness", "answer_relevance"]
    status: Literal["ok", "empty_answer", "error"]
    score: Optional[float] = Field(default=None, ge=0, le=1)
    reasoning: str = ""
    error: Optional[str] = None

class AccuracyEvaluator:
    def __init__(self, model_name="gemini-2.5-flash", client=None, max_attempts=3):
        self.judge = Judge(model_name, client, max_attempts)

    def evaluate(self, question, generated_answer, reference_answer=None):
        metric = "answer_correctness" if reference_answer and reference_answer.strip() else "answer_relevance"
        if not generated_answer.strip():
            return AccuracyResult(metric=metric, status="empty_answer")
        rubric = ("Compare the answer against the verified reference for factual correctness and completeness."
                  if metric == "answer_correctness" else
                  "Grade how directly and completely the answer addresses the question. Do not grade factual truth.")
        prompt = evaluation_prompt(rubric + " Score 1 for fully satisfying, 0.5 for partial, "
                                  "0 for incorrect/irrelevant; intermediate values allowed. Return score and reasoning.",
                                  question=question, answer=generated_answer, reference=reference_answer)
        try:
            result = self.judge.call(prompt, JudgeScore, JudgeScore.model_validate)
            return AccuracyResult(metric=metric, status="ok", **result.model_dump())
        except Exception as error:
            return AccuracyResult(metric=metric, status="error", error=type(error).__name__)
