
from eval.baseline_runner import (
    ABSTAINED_OUTCOME,
    NOT_FOUND_SENTENCE,
    answer_correctness_score,
    calculate_metrics,
    run_comparison,
)


def test_answer_correctness_score():
    score = answer_correctness_score(
        "The sky is blue",
        "The sky is blue",
    )
    assert score == 1.0


def test_answer_correctness_score_partial_match():
    score = answer_correctness_score(
        "The sky is blue",
        "The sky is clear",
    )
    assert 0.0 < score < 1.0


def test_answer_correctness_score_empty_expected_answer():
    assert answer_correctness_score("", "some answer") == 0.0


def test_calculate_metrics():
    rows = [
        {
            "is_adversarial": True,
            "final_outcome": ABSTAINED_OUTCOME,
            "answer": "",
            "answer_correct": None,
            "total_attempts": 1,
        },
        {
            "is_adversarial": True,
            "final_outcome": "answered",
            "answer": "An unsupported answer.",
            "answer_correct": None,
            "total_attempts": 1,
        },
        {
            "is_adversarial": False,
            "final_outcome": "answered",
            "answer": "model predicts token",
            "answer_correct": True,
            "answer_correctness_score": 1.0,
            "total_attempts": 2,
            "relevant_chunks": ["c1", "c2"],
            "retrieved_chunk_ids": ["c1"],
        },
    ]

    metrics = calculate_metrics(rows)

    assert metrics["total_queries"] == 3
    assert metrics["hallucination_rate"] is None
    assert metrics["hallucination_rate_status"] == "not measured"
    assert metrics["false_answer_rate"] == 1 / 2
    assert metrics["correct_abstention_rate"] == 1 / 2
    assert metrics["abstention_rate"] == 1 / 3
    assert metrics["recall_at_k"] == 0.5
    assert metrics["hit_rate_at_k"] == 1.0
    assert metrics["answer_correctness"] == 1.0


def test_false_answer_rate_counts_answered_adversarial_queries():
    rows = [
        {
            "query": "adversarial answered",
            "is_adversarial": True,
            "final_outcome": "answered",
            "answer": "An unsupported answer.",
            "answer_correct": None,
        },
        {
            "query": "adversarial abstained",
            "is_adversarial": True,
            "final_outcome": ABSTAINED_OUTCOME,
            "answer": "",
            "answer_correct": None,
        },
    ]

    metrics = calculate_metrics(rows)

    assert metrics["false_answer_rate"] == 1 / 2
    assert metrics["correct_abstention_rate"] == 1 / 2
    assert metrics["abstention_rate"] == 1 / 2


def test_adversarial_abstention_by_final_outcome():
    rows = [
        {
            "query": "adversarial query",
            "is_adversarial": True,
            "final_outcome": ABSTAINED_OUTCOME,
            "answer": "Some other text.",
            "answer_correct": None,
        },
    ]

    metrics = calculate_metrics(rows)

    assert metrics["false_answer_rate"] == 0.0
    assert metrics["correct_abstention_rate"] == 1.0
    assert metrics["abstention_rate"] == 1.0


def test_baseline_not_found_sentence_counts_as_adversarial_abstention():
    rows = [
        {
            "query": "adversarial baseline query",
            "is_adversarial": True,
            "final_outcome": "answered",
            "answer": NOT_FOUND_SENTENCE,
            "answer_correct": None,
        },
    ]

    metrics = calculate_metrics(rows)

    assert metrics["false_answer_rate"] == 0.0
    assert metrics["correct_abstention_rate"] == 1.0
    assert metrics["abstention_rate"] == 1.0
    assert metrics["hallucination_rate"] is None
    assert metrics["hallucination_rate_status"] == "not measured"


def test_run_comparison_returns_both_modes():
    dataset = [
        {
            "query": "What is retrieval?",
            "is_adversarial": False,
            "expected_answer": "Retrieval finds relevant documents.",
            "relevant_chunks": ["c1"],
        }
    ]

    def fake_baseline(item):
        return {
            "answer": "Retrieval finds relevant documents.",
            "final_outcome": "answered",
            "total_attempts": 1,
            "retrieved_chunk_ids": ["c1"],
        }

    def fake_corrected(item):
        return {
            "answer": "Retrieval finds relevant documents.",
            "final_outcome": "answered",
            "total_attempts": 1,
            "retrieved_chunk_ids": ["c1"],
        }

    result = run_comparison(
        run_baseline=fake_baseline,
        run_corrected=fake_corrected,
        dataset=dataset,
    )

    assert set(result["results"]) == {"baseline", "corrected"}
    assert len(result["results"]["baseline"]) == 1
    assert len(result["results"]["corrected"]) == 1
