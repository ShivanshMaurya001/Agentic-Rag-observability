
from app.abstain import ABSTAIN_MESSAGE, abstain_node


def test_abstain_returns_expected_message_and_outcome():
    state = {
        "original_query": "What is the capital of Mars?",
        "current_query": "What is the capital of Mars?",
        "retrieved_chunks": [],
        "answer": "Some unsupported answer",
        "judge_verdict": None,
        "attempt_number": 2,
        "failure_type": ["retrieval", "generation", "generation"],
        "final_outcome": None,
    }

    result = abstain_node(state)

    assert result["answer"] == (
        "I couldn't find sufficient evidence in the provided documents."
    )
    assert result["final_outcome"] == "abstained_judge_rejected"


def test_abstain_preserves_other_state_fields():
    state = {
        "original_query": "What is the capital of Mars?",
        "current_query": "What is the capital of Mars?",
        "retrieved_chunks": [{"chunk_id": "c1", "text": "Sample context"}],
        "answer": "Unsupported answer",
        "judge_verdict": None,
        "attempt_number": 2,
        "failure_type": ["retrieval", "generation"],
        "final_outcome": None,
    }

    result = abstain_node(state)

    assert result["original_query"] == state["original_query"]
    assert result["current_query"] == state["current_query"]
    assert result["retrieved_chunks"] == state["retrieved_chunks"]
    assert result["attempt_number"] == state["attempt_number"]
    assert result["failure_type"] == state["failure_type"]
    assert result["judge_verdict"] == state["judge_verdict"]


def test_abstain_does_not_mutate_input_state():
    state = {
        "original_query": "Question",
        "current_query": "Question",
        "retrieved_chunks": [],
        "answer": "Old answer",
        "judge_verdict": None,
        "attempt_number": 2,
        "failure_type": ["generation"],
        "final_outcome": None,
    }

    original_state = state.copy()

    abstain_node(state)

    assert state == original_state
