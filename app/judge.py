from typing import Literal

from pydantic import BaseModel, Field


class JudgeVerdict(BaseModel):
    grounded: bool
    retrieval_quality: float = Field(ge=0.0, le=1.0)
    answer_quality: float = Field(ge=0.0, le=1.0)
    failure_type: Literal["retrieval", "generation", "none"]
    reason: str