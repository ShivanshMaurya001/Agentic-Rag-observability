from app.embedding_service import client
from app.judge import judge_answer
from app.retriever import retrieve
from app.state import GraphState
from app.prompts.rewrite_prompt import build_rewrite_prompt
from langgraph.graph import END, START, StateGraph


def retrieve_node(state: GraphState) -> GraphState:
    chunks = retrieve(state["current_query"])

    return {
        **state,
        "retrieved_chunks": chunks,
    }


def generate_node(state: GraphState) -> GraphState:
    context = "\n\n".join(
        chunk["text"] for chunk in state["retrieved_chunks"]
    )

    prompt = f"""
You are an assistant that answers questions about a document.

Use ONLY the information provided in the context below.

If the answer is not present in the context, say:
"The answer was not found in the document."

Context:
{context}

Question:
{state["current_query"]}

Answer:
""".strip()

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )

    if response is None or response.text is None:
        raise ValueError("Gemini generator returned no answer.")

    return {
        **state,
        "answer": response.text,
    }


def judge_node(state: GraphState) -> GraphState:
    context = "\n\n".join(
        chunk["text"] for chunk in state["retrieved_chunks"]
    )

    verdict = judge_answer(
        query=state["current_query"],
        context=context,
        answer=state["answer"],
    )

    return {
        **state,
        "judge_verdict": verdict,
        "failure_type": [
            *state["failure_type"],
            verdict.failure_type,
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

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )

    if response is None or response.text is None:
        raise ValueError("Gemini rewrite returned no query.")

    return {
        **state,
        "current_query": response.text.strip(),
        "attempt_number": state["attempt_number"] + 1,
    }


def regenerate_node(state: GraphState) -> GraphState:
    context = "\n\n".join(
        chunk["text"] for chunk in state["retrieved_chunks"]
    )

    prompt = f"""
You are an assistant that answers questions about a document.

Use ONLY the information provided in the context below.

If the answer is not present in the context, say:
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

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )

    if response is None or response.text is None:
        raise ValueError("Gemini regeneration returned no answer.")

    return {
        **state,
        "answer": response.text,
        "attempt_number": state["attempt_number"] + 1,
    }

# answer of the state graph after the judge node, based on the judge verdict and attempt number
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

# nodes and edges for the state graph 

def answered_node(state: GraphState) -> GraphState:
    if state["attempt_number"] == 0:
        outcome = "answered"
    else:
        outcome = "answered_after_retry"

    return {
        **state,
        "final_outcome": outcome,
    }


def abstain_node(state: GraphState) -> GraphState:
    return {
        **state,
        "answer": "The answer could not be reliably determined from the document.",
        "final_outcome": "abstained_judge_rejected",
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