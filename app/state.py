from typing import Optional, TypedDict

from app.judge import JudgeVerdict


class GraphState(TypedDict):
    original_query: str
    current_query: str
    retrieved_chunks: list[dict]
    answer: str
    judge_verdict: Optional[JudgeVerdict]
    attempt_number: int
    failure_type: list[str]
    final_outcome: Optional[str]

