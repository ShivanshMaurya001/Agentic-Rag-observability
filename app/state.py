from typing import Optional, TypedDict, Any

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

    # M4: Per-attempt trace data and stage latency
    attempts: list[dict[str, Any]]
    retrieve_latency_ms: float
    generate_latency_ms: float

    # M4: Reuse baseline's initial retrieval and generation
    reuse_initial_attempt: bool