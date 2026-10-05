import os
from typing import Literal

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field

from app.prompts.judge_prompt import build_judge_prompt


load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(
    api_key=API_KEY
)


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

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": JudgeVerdict,
        },
    )

    if response is None:
        raise ValueError("Gemini judge returned no response.")

    if response.parsed is None:
        raise ValueError("Gemini judge returned no structured verdict.")

    return JudgeVerdict.model_validate(response.parsed)