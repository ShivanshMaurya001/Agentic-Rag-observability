from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from app.prompts.judge_prompt import build_judge_prompt
from app.config import JUDGE_MODEL
from app.llm_service import LLMServiceError, generate_structured


class JudgeVerdict(BaseModel):
    grounded: bool
    retrieval_quality: float = Field(ge=0.0, le=1.0)
    answer_quality: float = Field(ge=0.0, le=1.0)
    failure_type: Literal["retrieval", "generation", "none"]
    reason: str


def judge_answer(query: str, context: str, answer: str) -> JudgeVerdict:
    prompt = build_judge_prompt(
        query=query,
        context=context,
        answer=answer,
    )

    try:
        verdict = generate_structured(
            prompt,
            model=JUDGE_MODEL,
            schema=JudgeVerdict,
        )
        return JudgeVerdict.model_validate(verdict)
    except (ValidationError, LLMServiceError):
        raise ValueError("OpenAI judge did not return a valid verdict.") from None
