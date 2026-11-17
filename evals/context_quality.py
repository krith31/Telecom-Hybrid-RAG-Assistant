from typing import Optional
from pydantic import BaseModel, ConfigDict, Field, StrictBool
from .judge import Judge, evaluation_prompt

class EvidenceDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1)
    supported: StrictBool
    reasoning: str = Field(min_length=1)

class ChunkDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chunk_index: int = Field(ge=0, strict=True)
    relevant: StrictBool
    reasoning: str = Field(min_length=1)

class ContextJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chunks: list[ChunkDecision]
    reference_claims: list[EvidenceDecision]

class ContextQualityEvaluator:
    def __init__(self, model_name="gemini-3.8-flash", client=None):
        self.judge = Judge(model_name,client)

    def evaluate(self, question, chunks, reference=None):
        if not chunks:
            return {"status":"no_context", "context_relevance":None, "reference_context_recall":None}
        prompt = evaluation_prompt(
            "For EVERY retrieved chunk, classify whether it contains evidence useful for answering the question. "
            "Use zero-based chunk_index, exactly once per input chunk. Also split the verified reference "
            "into atomic factual claims, marking supported=true only if the combined chunks entail that claim. "
            "Use no external knowledge. Without a reference return reference_claims=[]. Return chunks and reference_claims.",
            question=question, chunks=chunks, verified_reference=reference)
        try:
            result = self.judge.call(prompt, ContextJudgment, ContextJudgment.model_validate)
            if sorted(c.chunk_index for c in result.chunks) != list(range(len(chunks))):
                raise ValueError("Judge must label every chunk exactly once")
            if not reference and result.reference_claims:
                raise ValueError("Reference claims without a reference")
            relevance = sum(c.relevant for c in result.chunks)/len(chunks)
            recall = (sum(c.supported for c in result.reference_claims)/len(result.reference_claims)
                      if result.reference_claims else None)
            return {"status":"ok", "context_relevance":relevance, "reference_context_recall":recall,
                    "evidence":result.model_dump()}
        except Exception as error:
            return {"status":"error", "error":type(error).__name__,
                    "context_relevance":None,"reference_context_recall":None}

class BehaviorJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    abstained: StrictBool
    citation_support: Optional[float] = Field(default=None, ge=0, le=1, strict=True, allow_inf_nan=False)
    citation_completeness: Optional[float] = Field(default=None, ge=0, le=1, strict=True, allow_inf_nan=False)
    reasoning: str = Field(min_length=1)

class BehaviorEvaluator:
    def __init__(self, model_name="gemini-3.8-flash", client=None):
        self.judge = Judge(model_name,client)

    def evaluate(self, question, answer, chunks, expected_abstention=None):
        prompt = evaluation_prompt(
            "Determine whether the answer abstains because evidence is insufficient. "
            "For exact [Spec: NAME | Clause ID] citations, citation_support is the fraction of cited "
            "technical claims entailed by their specifically cited chunks. citation_completeness is the "
            "fraction of all factual technical claims that have a valid supporting citation. "
            "Use null for citation_support if there are no cited claims, and null for citation_completeness "
            "if there are no factual technical claims. Return abstained, citation_support, "
            "citation_completeness and reasoning. Do not assume a source supports a claim just because its ID exists.",
            question=question, answer=answer, chunks=chunks)
        try:
            result = self.judge.call(prompt,BehaviorJudgment,BehaviorJudgment.model_validate)
            if expected_abstention is not None and not isinstance(expected_abstention,bool):
                raise ValueError("expected_abstention must be bool or null")
            return {"status":"ok", **result.model_dump(),
                    "abstention_accuracy":float(result.abstained==expected_abstention) if expected_abstention is not None else None}
        except Exception as error:
            return {"status":"error","error":type(error).__name__}
