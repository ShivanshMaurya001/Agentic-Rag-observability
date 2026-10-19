
import time
from typing import Any

from app.abstain import abstain_node
from app.judge import judge_answer
from app.llm_service import generate_text
from app.retriever import retrieve
from app.state import GraphState
from app.prompts.rewrite_prompt import build_rewrite_prompt
from langgraph.graph import END, START, StateGraph
from app.config import GENERATOR_MODEL, REWRITE_MODEL


def retrieve_node(state: GraphState) -> GraphState:
    # Reuse baseline chunks on the corrected run's first attempt.
    if (
        state.get("reuse_initial_attempt", False)
        and state["attempt_number"] == 0
        and state.get("retrieved_chunks")
    ):
        return state

    start_time = time.perf_counter()
    chunks = retrieve(state["current_query"])
    latency_ms = (time.perf_counter() - start_time) * 1000

    return {
        **state,
        "retrieved_chunks": chunks,
        "retrieve_latency_ms": latency_ms,
    }


def generate_node(state: GraphState) -> GraphState:
    # Reuse baseline answer on the first corrected attempt.
    if (
        state.get("reuse_initial_attempt", False)
        and state["attempt_number"] == 0
        and state.get("answer")
    ):
        return state

    context = "\n\n".join(
        chunk["text"] for chunk in state["retrieved_chunks"]
    )

    prompt = f"""
You are an assistant that answers questions about a document.

Use ONLY the information provided in the context below.

If the answer is not present, say:
"The answer was not found in the document."

Context:
{context}

Question:
{state["current_query"]}

Answer:
""".strip()

    start_time = time.perf_counter()
    answer = generate_text(prompt, model=GENERATOR_MODEL)
    latency_ms = (time.perf_counter() - start_time) * 1000

    return {
        **state,
        "answer": answer,
        "generate_latency_ms": latency_ms,
    }


def _verdict_to_dict(verdict: Any) -> dict[str, Any]:
    if hasattr(verdict, "model_dump"):
        return verdict.model_dump()

    if hasattr(verdict, "dict"):
        return verdict.dict()

    raise TypeError(
        "Judge verdict must support model_dump() or dict()."
    )


def judge_node(state: GraphState) -> GraphState:
    context = "\n\n".join(
        chunk["text"] for chunk in state["retrieved_chunks"]
    )

    start_time = time.perf_counter()
    verdict = judge_answer(
        query=state["current_query"],
        context=context,
        answer=state["answer"],
    )
    judge_latency_ms = (time.perf_counter() - start_time) * 1000

    attempt = {
        "attempt_number": state["attempt_number"],
        "query_used": state["current_query"],
        "retrieved_chunks": state["retrieved_chunks"],
        "generated_answer": state["answer"],
        "judge_verdict": _verdict_to_dict(verdict),
        "retrieve_latency_ms": state.get("retrieve_latency_ms", 0.0),
        "generate_latency_ms": state.get("generate_latency_ms", 0.0),
        "judge_latency_ms": judge_latency_ms,
    }

    return {
        **state,
        "judge_verdict": verdict,
        "failure_type": [
            *state["failure_type"],
            verdict.failure_type,
        ],
        "attempts": [
            *state.get("attempts", []),
            attempt,
        ],
    }


def route_after_judge(state: GraphState) -> str:
    failure_type = state["judge_verdict"].failure_type

    if failure_type == "retrieval":
        return "rewrite_query"

    if failure_type == "generation":
        return "regenerate"

    return "pass"


def rewrite_query_node(state: GraphState) -> GraphState:
    judge_reason = state["judge_verdict"].reason

    prompt = build_rewrite_prompt(
        original_query=state["original_query"],
        judge_reason=judge_reason,
    )

    rewritten_query = generate_text(prompt, model=REWRITE_MODEL)

    return {
        **state,
        "current_query": rewritten_query.strip(),
        "attempt_number": state["attempt_number"] + 1,
    }


def regenerate_node(state: GraphState) -> GraphState:
    context = "\n\n".join(
        chunk["text"] for chunk in state["retrieved_chunks"]
    )

    prompt = f"""
You are an assistant that answers questions about a document.

Use ONLY the information provided in the context below.

If the answer is not present, say:
"The answer was not found in the document."

Context:
{context}

Question:
{state["current_query"]}

Previous answer:
{state["answer"]}

Generate a corrected answer using the same retrieved context.

Answer:
""".strip()

    start_time = time.perf_counter()
    answer = generate_text(prompt, model=GENERATOR_MODEL)
    latency_ms = (time.perf_counter() - start_time) * 1000

    return {
        **state,
        "answer": answer,
        "generate_latency_ms": latency_ms,
        "attempt_number": state["attempt_number"] + 1,
    }


def route_after_attempt(state: GraphState) -> str:
    verdict = state["judge_verdict"]

    if verdict.failure_type == "none":
        if state["attempt_number"] == 0:
            return "answered"

        return "answered_after_retry"

    if state["attempt_number"] >= 2:
        return "abstain"

    if verdict.failure_type == "retrieval":
        return "rewrite_query"

    if verdict.failure_type == "generation":
        return "regenerate"

    raise ValueError(
        f"Unexpected failure type: {verdict.failure_type}"
    )


def answered_node(state: GraphState) -> GraphState:
    if state["attempt_number"] == 0:
        outcome = "answered"
    else:
        outcome = "answered_after_retry"

    return {
        **state,
        "final_outcome": outcome,
    }


builder = StateGraph(GraphState)

builder.add_node("retrieve", retrieve_node)
builder.add_node("generate", generate_node)
builder.add_node("judge", judge_node)
builder.add_node("rewrite_query", rewrite_query_node)
builder.add_node("regenerate", regenerate_node)
builder.add_node("answered", answered_node)
builder.add_node("abstain", abstain_node)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", "judge")

builder.add_conditional_edges(
    "judge",
    route_after_attempt,
    {
        "answered": "answered",
        "answered_after_retry": "answered",
        "rewrite_query": "rewrite_query",
        "regenerate": "regenerate",
        "abstain": "abstain",
    },
)

builder.add_edge("answered", END)
builder.add_edge("rewrite_query", "retrieve")
builder.add_edge("regenerate", "judge")
builder.add_edge("abstain", END)

graph = builder.compile()
