
from eval.baseline_runner import (
    ABSTAINED_OUTCOME,
    NOT_FOUND_SENTENCE,
    answer_correctness_score,
    calculate_metrics,
    is_abstained,
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
    assert metrics["hallucination_rate"] == 1 / 2
    assert metrics["hallucination_rate_status"].startswith(
        "ground-truth proxy:"
    )
    assert metrics["false_answer_rate"] == 1 / 2
    assert metrics["correct_abstention_rate"] == 1 / 2
    assert metrics["abstention_rate"] == 1 / 3
    assert metrics["recall_at_k"] == 0.5
    assert metrics["hit_rate_at_k"] == 1.0
    assert metrics["answer_correctness"] == 1.0


def test_calculate_metrics_excludes_other_fingerprints_and_legacy_rows():
    active = {
        "model_fingerprint": "active",
        "is_adversarial": False,
        "final_outcome": "answered",
        "answer": "supported answer",
        "answer_correct": True,
        "answer_correctness_score": 1.0,
    }
    other_provider = {
        **active,
        "model_fingerprint": "gemini",
        "is_adversarial": True,
        "final_outcome": "answered",
        "answer_correct": False,
    }
    legacy = {key: value for key, value in active.items()
              if key != "model_fingerprint"}

    metrics = calculate_metrics(
        [active, other_provider, legacy],
        active_fingerprint="active",
    )

    assert metrics["total_queries"] == 1
    assert metrics["false_answer_rate"] == 0.0
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
    assert metrics["hallucination_rate"] == 1.0


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
    assert metrics["hallucination_rate"] is None
    assert metrics["hallucination_rate_status"].startswith(
        "not measured: no answered queries; ground-truth proxy:"
    )


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
    assert metrics["hallucination_rate_status"].startswith(
        "not measured: no answered queries; ground-truth proxy:"
    )


def test_hallucination_rate_uses_answered_ground_truth_proxy_only():
    rows = [
        {
            "is_adversarial": True,
            "final_outcome": "answered",
            "answer": "A fabricated response.",
            "answer_correct": None,
        },
        {
            "is_adversarial": False,
            "final_outcome": "answered",
            "answer": "Incorrect answer.",
            "answer_correct": False,
        },
        {
            "is_adversarial": False,
            "final_outcome": "answered",
            "answer": "Correct answer.",
            "answer_correct": True,
        },
        {
            "is_adversarial": False,
            "final_outcome": ABSTAINED_OUTCOME,
            "answer": "I cannot answer.",
            "answer_correct": False,
        },
    ]

    metrics = calculate_metrics(rows)

    assert metrics["hallucination_rate"] == 2 / 3


def test_hallucination_rate_is_incomplete_for_missing_or_null_correctness():
    rows = [
        {
            "is_adversarial": False,
            "final_outcome": "answered",
            "answer": "An answer without a saved correctness label.",
        },
        {
            "is_adversarial": False,
            "final_outcome": "answered",
            "answer": "An answer with a null correctness label.",
            "answer_correct": None,
        },
    ]

    for row in rows:
        metrics = calculate_metrics([row])

        assert metrics["hallucination_rate"] is None
        assert metrics["hallucination_rate_status"].startswith(
            "incomplete: answer_correct is missing, null, or non-boolean "
            "for answered non-adversarial queries; "
            "ground-truth proxy:"
        )


def test_paraphrased_refusal_uses_final_judge_abstention_classification():
    row = {
        "final_outcome": "answered_after_retry",
        "answer": "I couldn't find enough information to answer that.",
        "attempts": [
            {"judge_verdict": {"is_abstention": True}},
        ],
    }

    assert is_abstained(row) is True


def test_substantive_answer_with_refusal_like_wording_is_not_abstention():
    row = {
        "final_outcome": "answered_after_retry",
        "answer": (
            "The document does not specify the exact date, but it states "
            "that training used 2024 data."
        ),
        "attempts": [
            {"judge_verdict": {"is_abstention": False}},
        ],
    }

    assert is_abstained(row) is False


def test_rows_without_explicit_judge_classification_use_legacy_fallbacks():
    assert is_abstained({
        "final_outcome": ABSTAINED_OUTCOME,
        "answer": "A different abstention message.",
        "attempts": [{"judge_verdict": {}}],
    }) is True
    assert is_abstained({
        "final_outcome": "answered",
        "answer": NOT_FOUND_SENTENCE,
        "attempts": [{"judge_verdict": {}}],
    }) is True
    assert is_abstained({
        "final_outcome": "answered",
        "answer": "A substantive legacy answer.",
    }) is False


def test_explicit_final_judge_classification_takes_precedence():
    row = {
        "final_outcome": ABSTAINED_OUTCOME,
        "answer": NOT_FOUND_SENTENCE,
        "attempts": [
            {"judge_verdict": {"is_abstention": True}},
            {"judge_verdict": {"is_abstention": False}},
        ],
    }

    assert is_abstained(row) is False


def test_reviewed_corrected_refusals_are_abstentions():
    reviewed_refusals = [
        (
            "What was the exact benchmark score of the LLM in the 2024 Mars Olympics?",
            "The document does not mention the 2024 Mars Olympics or provide an LLM benchmark score.",
        ),
        (
            "According to these documents, what exact date will the next generation of LLMs become fully conscious?",
            "The documents do not provide a date or an explicit discussion or prediction about when the next generation of LLMs will become fully conscious.",
        ),
        (
            "How many parameters did the LLM in these documents have?",
            "The document refers to LLMs generically and does not name a specific model. It does not state a parameter count.",
        ),
        (
            "What exact GPU model and number of GPUs were used to pretrain the LLM discussed in these documents?",
            "This information is not specified.",
        ),
        (
            "Which specific dataset and version were used to train the LLM described in the document?",
            "The document does not identify a specific dataset or version used to train the LLM. It only describes pretraining as using “a very large, diverse corpus of text.”",
        ),
        (
            "What exact maximum context-window size does the PDF specify for the model discussed?",
            "The PDF discusses models in general and does not identify a specific model. It does not state a maximum context-window size in tokens or give a specific numeric limit.",
        ),
        (
            "Which exact date and dataset version were used for the model's training run?",
            "The exact date of the model’s training run and the dataset version used are not stated in the document.",
        ),
        (
            "What exact hardware and GPU count were used to train the model described in the PDF?",
            "The PDF does not specify the accelerator/GPU model or the number of GPUs or nodes. It only says that pretraining “requires enormous computational resources.”",
        ),
    ]

    for query, answer in reviewed_refusals:
        assert is_abstained({
            "query": query,
            "answer": answer,
            "final_outcome": "answered_after_retry",
        }) is True


def test_query_specific_answer_veto_preserves_mixed_common_crawl_answer():
    row = {
        "query": "Which specific dataset and version were used to train the LLM?",
        "answer": (
            "The document does not specify the dataset version, but it identifies "
            "the training dataset as Common Crawl."
        ),
        "final_outcome": "answered_after_retry",
    }

    assert is_abstained(row) is False


def test_instruction_tuning_negative_is_not_an_absence_pattern():
    row = {
        "query": "How does instruction tuning affect a pretrained language model?",
        "answer": (
            "Instruction tuning further trains a pretrained model on examples "
            "pairing instructions with desired responses. It aims to make the "
            "model more helpful at following user requests and producing "
            "expected formats, but does not guarantee that it will follow every "
            "instruction or distinguish evidence from speculation."
        ),
        "final_outcome": "answered_after_retry",
    }

    assert is_abstained(row) is False


def test_explicit_benchmark_metric_value_veto_preserves_answered_classification():
    row = {
        "query": "What was the benchmark score of the model?",
        "answer": (
            "The document does not discuss the benchmark, but reports a "
            "benchmark score of 87.5%."
        ),
        "final_outcome": "answered_after_retry",
    }

    assert is_abstained(row) is False


def test_query_category_value_vetoes_preserve_answered_classification():
    answered_values = [
        (
            "What dataset version was used?",
            "The document does not specify the dataset version, but the version is 2.1.",
        ),
        (
            "What was the training date?",
            "The document does not state the training date, but the run occurred in 2023.",
        ),
        (
            "How many parameters did the model have?",
            "The document does not state the parameter count, but the model has 7 billion parameters.",
        ),
        (
            "How many GPUs were used?",
            "The document does not identify the GPU count, but it used 8 GPUs.",
        ),
        (
            "What was the context window size in tokens?",
            "The document does not specify the context window, but it supports 128k tokens.",
        ),
        (
            "Which model was used?",
            "The document does not name the model, but the model was GPT-4.",
        ),
    ]

    for query, answer in answered_values:
        assert is_abstained({
            "query": query,
            "answer": answer,
            "final_outcome": "answered",
        }) is False


def test_ambiguous_or_unsupported_absence_preserves_existing_classification():
    unsupported = {
        "query": "What is the broader significance of this discussion?",
        "answer": "The document does not specify this information.",
        "final_outcome": "answered",
    }
    ambiguous = {
        "query": "Which model was used?",
        "answer": (
            "The document does not name the model. It discusses the system "
            "in relation to several other methods."
        ),
        "final_outcome": "answered",
    }

    assert is_abstained(unsupported) is False
    assert is_abstained(ambiguous) is False


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

    metadata = {
        "baseline": {
            "providers": {"generator": "openai"},
            "models": {"generator": "model-a"},
            "model_fingerprint": "baseline-fingerprint",
        },
        "corrected": {
            "providers": {
                "generator": "openai",
                "rewrite": "openai",
                "judge": "openai",
            },
            "models": {
                "generator": "model-a",
                "rewrite": "model-b",
                "judge": "model-c",
            },
            "model_fingerprint": "corrected-fingerprint",
        },
    }

    result = run_comparison(
        run_baseline=fake_baseline,
        run_corrected=fake_corrected,
        dataset=dataset,
        metadata_by_mode=metadata,
    )

    assert set(result["results"]) == {"baseline", "corrected"}
    assert len(result["results"]["baseline"]) == 1
    assert len(result["results"]["corrected"]) == 1

    for mode in ("baseline", "corrected"):
        row = result["results"][mode][0]
        assert row["providers"] == metadata[mode]["providers"]
        assert row["models"] == metadata[mode]["models"]
        assert row["model_fingerprint"] == metadata[mode]["model_fingerprint"]
        assert result["metrics"][mode]["total_queries"] == 1
