
from eval.baseline_runner import (
    ABSTAINED_OUTCOME,
    answer_correctness_score,
    calculate_metrics,
    run_comparison,
)


def test_answer_correctness_full_match():
    assert answer_correctness_score(
        "The model predicts the next token",
        "The model predicts the next token.",
    ) == 1.0


def test_answer_correctness_partial_match():
    score = answer_correctness_score(
        "model predicts next token",
        "model predicts token",
    )
    assert score == 0.75


def test_answer_correctness_paraphrase_miss():
    score = answer_correctness_score(
        "rapid vehicle",
        "fast automobile",
    )
    assert score == 0.0


def test_answerable_abstention_is_incorrect():
    query = {
        "query": "What does the model predict?",
        "is_adversarial": False,
        "expected_answer": "model predicts token",
        "relevant_chunks": ["chunk-1"],
    }

    result = run_comparison(
        run_baseline=lambda _: {
            "answer": "",
            "final_outcome": ABSTAINED_OUTCOME,
            "total_attempts": 1,
            "retrieved_chunk_ids": ["chunk-1"],
        },
        run_corrected=lambda _: {
            "answer": "",
            "final_outcome": ABSTAINED_OUTCOME,
            "total_attempts": 1,
            "retrieved_chunk_ids": ["chunk-1"],
        },
        dataset=[query],
    )

    row = result["results"]["baseline"][0]
    assert row["answer_correctness_score"] == 0.0
    assert row["answer_correct"] is False


def test_adversarial_query_skips_answer_correctness():
    query = {
        "query": "Unanswerable question",
        "is_adversarial": True,
        "expected_answer": "",
        "relevant_chunks": [],
    }

    result = run_comparison(
        run_baseline=lambda _: {
            "answer": "some answer",
            "final_outcome": "answered",
            "total_attempts": 1,
        },
        run_corrected=lambda _: {
            "answer": "",
            "final_outcome": ABSTAINED_OUTCOME,
            "total_attempts": 1,
        },
        dataset=[query],
    )

    for mode in ("baseline", "corrected"):
        row = result["results"][mode][0]
        assert row["answer_correctness_score"] is None
        assert row["answer_correct"] is None


def test_calculate_metrics():
    rows = [
        {
            "is_adversarial": True,
            "final_outcome": ABSTAINED_OUTCOME,
            "answer_correct": None,
            "is_hallucination": False,
            "total_attempts": 1,
        },
        {
            "is_adversarial": True,
            "final_outcome": "answered",
            "answer_correct": False,
            "is_hallucination": True,
            "total_attempts": 1,
        },
        {
            "is_adversarial": False,
            "final_outcome": "answered",
            "answer_correct": True,
            "answer_correctness_score": 1.0,
            "is_hallucination": False,
            "total_attempts": 2,
            "relevant_chunks": ["c1", "c2"],
            "retrieved_chunk_ids": ["c1"],
        },
    ]

    metrics = calculate_metrics(rows)
    assert metrics["total_queries"] == 3
    assert metrics["hallucination_rate"] == 1 / 3
    assert metrics["false_answer_rate"] == 1 / 2
    assert metrics["correct_abstention_rate"] == 1 / 2
    assert metrics["abstention_rate"] == 1 / 3
    assert metrics["recall_at_k"] == 0.5
    assert metrics["hit_rate_at_k"] == 1.0
    assert metrics["answer_correctness"] == 1.0
