
import re
import string
from collections.abc import Callable
from typing import Any

from app.config import ANSWER_CORRECTNESS_THRESHOLD
from eval.eval_dataset import load_eval_queries

ABSTAINED_OUTCOME = "abstained_judge_rejected"
NOT_FOUND_SENTENCE = "The answer was not found in the document."

# Lightweight offline normalization; intentionally not a semantic metric.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by",
    "for", "from", "in", "is", "it", "of", "on", "or", "that",
    "the", "this", "to", "was", "were", "will", "with",
}


def is_abstained(row: dict[str, Any]) -> bool:
    """Use the same abstention rule for baseline and corrected results."""
    if row.get("final_outcome") == ABSTAINED_OUTCOME:
        return True

    answer = row.get("answer", "")
    return (
        isinstance(answer, str)
        and NOT_FOUND_SENTENCE in answer
    )


def _key_terms(text: str) -> set[str]:
    """Return unique lowercase terms without punctuation or stopwords."""
    text = text.lower().translate(
        str.maketrans("", "", string.punctuation)
    )
    return {
        term
        for term in re.findall(r"\b\w+\b", text)
        if term not in STOPWORDS
    }


def answer_correctness_score(
    expected_answer: str,
    generated_answer: str,
) -> float:
    """Fraction of expected-answer key terms present in the generated answer."""
    expected_terms = _key_terms(expected_answer)
    if not expected_terms:
        return 0.0

    generated_terms = _key_terms(generated_answer)
    return len(expected_terms & generated_terms) / len(expected_terms)


def calculate_metrics(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Calculate M4 evaluation metrics for one evaluation mode."""
    if not rows:
        raise ValueError(
            "Cannot calculate metrics for an empty evaluation."
        )

    total = len(rows)

    adversarial_rows = [
        row for row in rows if row["is_adversarial"]
    ]
    answerable_rows = [
        row for row in rows if not row["is_adversarial"]
    ]

    abstained_rows = [
        row for row in rows if is_abstained(row)
    ]

    # By the agreed definition, every non-abstention on an
    # adversarial query is a false answer.
    false_answer_rows = [
        row for row in adversarial_rows
        if not is_abstained(row)
    ]

    correct_abstention_rows = [
        row for row in adversarial_rows
        if is_abstained(row)
    ]

    # No exact hallucination-rate formula is specified in the Blueprint.
    hallucination_rate = None
    hallucination_rate_status = "not measured"

    recall_values: list[float] = []
    hit_values: list[float] = []

    for row in rows:
        relevant = set(row.get("relevant_chunks") or [])
        retrieved = set(row.get("retrieved_chunk_ids") or [])

        if relevant:
            recall_values.append(
                len(relevant & retrieved) / len(relevant)
            )
            hit_values.append(
                float(bool(relevant & retrieved))
            )

    scored_answerable = [
        row["answer_correctness_score"]
        for row in answerable_rows
        if row.get("answer_correctness_score") is not None
    ]

    correctness_values = [
        row["answer_correct"]
        for row in answerable_rows
        if row.get("answer_correct") is not None
    ]

    retry_eligible = [
        row for row in rows
        if row.get("total_attempts", 1) > 1
    ]

    retry_successes = [
        row for row in retry_eligible
        if row["final_outcome"] == "answered_after_retry"
    ]

    adversarial_total = len(adversarial_rows)

    return {
        "total_queries": total,
        "hallucination_rate": hallucination_rate,
        "hallucination_rate_status": hallucination_rate_status,
        "false_answer_rate": (
            len(false_answer_rows) / adversarial_total
            if adversarial_total else 0.0
        ),
        "correct_abstention_rate": (
            len(correct_abstention_rows) / adversarial_total
            if adversarial_total else 0.0
        ),
        "abstention_rate": len(abstained_rows) / total,
        "recall_at_k": (
            sum(recall_values) / len(recall_values)
            if recall_values else 0.0
        ),
        "hit_rate_at_k": (
            sum(hit_values) / len(hit_values)
            if hit_values else 0.0
        ),
        "answer_correctness": (
            sum(correctness_values) / len(correctness_values)
            if correctness_values else 0.0
        ),
        "mean_answer_correctness_score": (
            sum(scored_answerable) / len(scored_answerable)
            if scored_answerable else 0.0
        ),
        "retry_success_rate": (
            len(retry_successes) / len(retry_eligible)
            if retry_eligible else 0.0
        ),
        "average_retries": (
            sum(
                max(row.get("total_attempts", 1) - 1, 0)
                for row in rows
            ) / total
        ),
    }


def _prepare_result(
    item: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    """Attach evaluation labels and compute offline answer correctness."""
    row = {**result, "is_adversarial": item["is_adversarial"]}
    row["relevant_chunks"] = item.get("relevant_chunks", [])

    if item["is_adversarial"]:
        # Adversarial correctness is not separately classified:
        # every non-abstention is a false answer by definition.
        row["answer_correctness_score"] = None
        row["answer_correct"] = None

    elif is_abstained(row):
        row["answer_correctness_score"] = 0.0
        row["answer_correct"] = False

    else:
        score = answer_correctness_score(
            item.get("expected_answer", ""),
            row.get("answer", ""),
        )
        row["answer_correctness_score"] = score
        row["answer_correct"] = (
            score >= ANSWER_CORRECTNESS_THRESHOLD
        )

    return row


def run_comparison(
    run_baseline: Callable[[dict[str, Any]], dict[str, Any]],
    run_corrected: Callable[[dict[str, Any]], dict[str, Any]],
    log_run: Callable[[dict[str, Any]], None] | None = None,
    dataset: list[dict[str, Any]] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run both supplied callables against the same evaluation items."""
    queries = list(
        dataset if dataset is not None else load_eval_queries()
    )

    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        queries = queries[:limit]

    if not queries:
        raise ValueError("Evaluation dataset is empty.")

    output: dict[str, Any] = {
        "results": {},
        "metrics": {},
    }

    for mode, runner in (
        ("baseline", run_baseline),
        ("corrected", run_corrected),
    ):
        rows: list[dict[str, Any]] = []

        for item in queries:
            row = _prepare_result(item, runner(item))
            row["mode"] = mode
            row["query"] = item["query"]
            rows.append(row)

            if log_run is not None:
                log_run(row)

        output["results"][mode] = rows
        output["metrics"][mode] = calculate_metrics(rows)

    return output
