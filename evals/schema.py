"""Strict input contracts for result evaluation."""
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr

class Chunk(BaseModel):
    model_config=ConfigDict(extra="allow")
    id: StrictStr=Field(min_length=1)
    content: StrictStr
    metadata: dict[str,Any]=Field(default_factory=dict)

class CapturedAnswer(BaseModel):
    model_config=ConfigDict(extra="allow")
    id: StrictStr=""
    question: StrictStr
    generated_answer: StrictStr=""
    retrieved_context: StrictStr=""
    retrieved_chunks: list[Chunk]=Field(default_factory=list)
    reference_answer: Optional[StrictStr]=None
    relevant_doc_ids: Optional[list[StrictStr]]=None
    expected_abstention: Optional[StrictBool]=None
    reference_source: StrictStr="unspecified"
    status: StrictStr="ok"
