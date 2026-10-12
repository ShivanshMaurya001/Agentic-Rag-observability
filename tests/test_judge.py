
import pytest
from pydantic import ValidationError
from langgraph.graph import END, START, StateGraph

from app.judge import JudgeVerdict


def test_valid_judge_verdict():
    verdict = JudgeVerdict(
        grounded=True,
        retrieval_quality=0.9,
        answer_quality=0.8,
        failure_type="none",
        reason="The answer is supported by the retrieved context.",
    )

    assert verdict.grounded is True
    assert verdict.retrieval_quality == 0.9
    assert verdict.answer_quality == 0.8
    assert verdict.failure_type == "none"


def test_invalid_failure_type_fails():
    with pytest.raises(ValidationError):
        JudgeVerdict(
            grounded=True,
            retrieval_quality=0.9,
            answer_quality=0.8,
            failure_type="unknown",
            reason="Invalid failure type.",
        )


def test_score_above_one_fails():
    with pytest.raises(ValidationError):
        JudgeVerdict(
            grounded=True,
            retrieval_quality=1.5,
            answer_quality=0.8,
            failure_type="none",
            reason="Invalid score.",
        )


def test_missing_field_fails():
    with pytest.raises(ValidationError):
        JudgeVerdict(
            grounded=True,
            retrieval_quality=0.9,
            answer_quality=0.8,
            failure_type="none",
        )


def test_judge_answer_returns_structured_verdict(monkeypatch):
    from app import judge

    class FakeResponse:
        parsed = {
            "grounded": True,
            "retrieval_quality": 1.0,
            "answer_quality": 0.9,
            "failure_type": "none",
            "reason": "The answer is supported by the context.",
        }

    def fake_generate_content(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr(
        judge.client.models,
        "generate_content",
        fake_generate_content,
    )

    verdict = judge.judge_answer(
        query="What does self-attention do?",
        context=(
            "Self-attention determines how relevant each token "
            "is to other tokens."
        ),
        answer="Self-attention determines token relevance.",
    )

    assert isinstance(verdict, judge.JudgeVerdict)
    assert verdict.grounded is True
    assert verdict.retrieval_quality == 1.0
    assert verdict.answer_quality == 0.9
    assert verdict.failure_type == "none"


def test_judge_answer_raises_on_missing_structured_response(monkeypatch):
    from app import judge

    class FakeResponse:
        parsed = None

    def fake_generate_content(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr(
        judge.client.models,
        "generate_content",
        fake_generate_content,
    )

    with pytest.raises(
        ValueError,
        match="Gemini judge returned no structured verdict.",
    ):
        judge.judge_answer(
            query="What does self-attention do?",
            context="Self-attention determines token relevance.",
            answer="Self-attention determines token relevance.",
        )


def test_graph_retries_after_retrieval_failure(monkeypatch):
    from app import graph as graph_module

    verdicts = iter([
        JudgeVerdict(
            grounded=False,
            retrieval_quality=0.2,
            answer_quality=0.0,
            failure_type="retrieval",
            reason="The retrieved context is irrelevant.",
        ),
        JudgeVerdict(
            grounded=True,
            retrieval_quality=0.9,
            answer_quality=0.9,
            failure_type="none",
            reason="The answer is supported by the context.",
        ),
    ])

    retrieve_calls = []

    def fake_retrieve(query):
        retrieve_calls.append(query)
        return [
            {
                "chunk_id": "test_chunk_1",
                "text": (
                    "Transformers use self-attention "
                    "to relate tokens."
                ),
            }
        ]

    def fake_generate_content(*args, **kwargs):
        class FakeResponse:
            text = (
                "Transformers use self-attention "
                "to relate tokens."
            )

        return FakeResponse()

    def fake_judge_answer(query, context, answer):
        return next(verdicts)

    def fake_rewrite_query_node(state):
        return {
            **state,
            "current_query": (
                "How does self-attention work in transformers?"
            ),
            "attempt_number": state["attempt_number"] + 1,
        }

    monkeypatch.setattr(
        graph_module,
        "retrieve",
        fake_retrieve,
    )
    monkeypatch.setattr(
        graph_module,
        "judge_answer",
        fake_judge_answer,
    )
    monkeypatch.setattr(
        graph_module.client.models,
        "generate_content",
        fake_generate_content,
    )

    # Build a fresh graph so it registers the fake rewrite node.
    builder = StateGraph(graph_module.GraphState)

    builder.add_node("retrieve", graph_module.retrieve_node)
    builder.add_node("generate", graph_module.generate_node)
    builder.add_node("judge", graph_module.judge_node)
    builder.add_node("rewrite_query", fake_rewrite_query_node)
    builder.add_node("regenerate", graph_module.regenerate_node)
    builder.add_node("answered", graph_module.answered_node)
    builder.add_node("abstain", graph_module.abstain_node)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", "judge")

    builder.add_conditional_edges(
        "judge",
        graph_module.route_after_attempt,
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

    test_graph = builder.compile()

    result = test_graph.invoke({
        "original_query": "Explain an unrelated topic",
        "current_query": "Explain an unrelated topic",
        "retrieved_chunks": [],
        "answer": "",
        "judge_verdict": None,
        "attempt_number": 0,
        "failure_type": [],
        "final_outcome": None,
    })

    assert len(retrieve_calls) == 2
    assert retrieve_calls[0] == "Explain an unrelated topic"
    assert retrieve_calls[1] == (
        "How does self-attention work in transformers?"
    )
    assert result["final_outcome"] == "answered_after_retry"
    assert result["attempt_number"] == 1
    assert result["failure_type"] == ["retrieval", "none"]
