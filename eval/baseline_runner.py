
import re
import string
from collections.abc import Callable
from typing import Any

from app.config import ANSWER_CORRECTNESS_THRESHOLD
from eval.eval_dataset import load_eval_queries

ABSTAINED_OUTCOME = "abstained_judge_rejected"
NOT_FOUND_SENTENCE = "The answer was not found in the document."
HALLUCINATION_RATE_STATUS = (
    "ground-truth proxy: answered adversarial queries plus answered "
    "non-adversarial queries with answer_correct=False; answer_correct "
    "uses expected-answer key-term coverage (threshold 0.6)"
)

# Lightweight offline normalization; intentionally not a semantic metric.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by",
    "for", "from", "in", "is", "it", "of", "on", "or", "that",
    "the", "this", "to", "was", "were", "will", "with",
}


def is_abstained(row: dict[str, Any]) -> bool:
    """Use the final judge classification, then legacy/baseline fallbacks."""
    attempts = row.get("attempts")
    if isinstance(attempts, list) and attempts:
        final_attempt = attempts[-1]
        if isinstance(final_attempt, dict):
            verdict = final_attempt.get("judge_verdict")
            if isinstance(verdict, dict):
                is_abstention = verdict.get("is_abstention")
                if isinstance(is_abstention, bool):
                    return is_abstention

    if row.get("final_outcome") == ABSTAINED_OUTCOME:
        return True

    answer = row.get("answer", "")
    if not isinstance(answer, str):
        return False
    if NOT_FOUND_SENTENCE in answer:
        return True

    query = row.get("query")
    if not isinstance(query, str):
        return False

    absence_patterns = (
        re.compile(
            r"\b(?:the\s+|these\s+|provided\s+)?"
            r"(?:documents?|pdf)\b.{0,100}\b(?:does|do)\s+not\s+"
            r"(?:mention|provide|state|specify|identify|name|contain|give|include)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:the\s+)?(?:provided\s+)?context\b.{0,100}"
            r"\b(?:does|do)\s+not\s+"
            r"(?:mention|provide|state|specify|identify|name|contain|give|include)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:this information|the exact\b.{0,100}?|"
            r"the\b.{0,100}?)\s+(?:is|are)\s+not\s+"
            r"(?:specified|stated|provided|available)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bno information\s+(?:is|was)\s+available\b"
            r"(?:.{0,60}\b(?:document|pdf|context)\b)?",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bit\s+(?:does|do)\s+not\s+"
            r"(?:mention|provide|state|specify|identify|name|contain|give|include)\b",
            re.IGNORECASE,
        ),
    )
    # A bare "it does not ..." is usable as a follow-up clause only when the
    # same answer also contains a source-qualified or passive absence claim.
    if not any(pattern.search(answer) for pattern in absence_patterns[:-1]):
        return False

    query_lower = query.lower()
    categories: set[str] = set()
    if re.search(r"\b(?:benchmark|metric|score)\b", query_lower):
        categories.add("benchmark_score")
    if re.search(r"\b(?:dataset|corpus)\b", query_lower):
        categories.add("dataset")
    if re.search(r"\bversion\b", query_lower):
        categories.add("version")
    if re.search(r"\b(?:date|when|year)\b", query_lower):
        categories.add("date")
    if re.search(r"\bparameters?\b", query_lower):
        categories.add("parameter_count")
    if re.search(r"\b(?:gpu|gpus|accelerator|hardware|nodes?)\b", query_lower):
        categories.add("hardware")
    if re.search(r"\b(?:context.window|tokens?)\b", query_lower):
        categories.add("context_limit")
    if re.search(r"\bmodel\b", query_lower):
        categories.add("model")
    if not categories:
        return False

    positive_answer_patterns = {
        "benchmark_score": re.compile(
            r"\b(?:benchmark|metric)\b.{0,30}\b(?:score|value)\b"
            r"\s*(?:is|was|of|:)?\s*\d+(?:\.\d+)?(?:\s*%|\s*/\s*\d+)?"
            r"|\b(?:score|scored)\s*(?:is|was|of|:)?\s*"
            r"\d+(?:\.\d+)?(?:\s*%|\s*/\s*\d+)?",
            re.IGNORECASE,
        ),
        "dataset": re.compile(
            r"\b(?:identifies?|names?|calls?)\s+(?:the\s+)?"
            r"(?:training\s+)?dataset\s+as\s+"
            r"[A-Z][\w.-]*(?:\s+[A-Z][\w.-]*)*"
            r"|\b(?:dataset|corpus)\s+(?:is|was|called|named)\s+"
            r"[A-Z][\w.-]*(?:\s+[A-Z][\w.-]*)*"
        ),
        "version": re.compile(
            r"\bversion\s*(?:is|was|:)?\s*v?\d+(?:\.\d+)*\b",
            re.IGNORECASE,
        ),
        "date": re.compile(
            r"\b(?:19|20)\d{2}\b|"
            r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|"
            r"may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|"
            r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+"
            r"\d{1,2}(?:,\s*\d{4})?\b",
            re.IGNORECASE,
        ),
        "parameter_count": re.compile(
            r"\b\d[\d,.]*\s*(?:million|billion|trillion)?\s*"
            r"parameters?\b|\bparameters?\s*(?:number|count)?\s*"
            r"(?:is|was|:)?\s*\d[\d,.]*\b",
            re.IGNORECASE,
        ),
        "hardware": re.compile(
            r"\b\d+\s*(?:gpus?|nodes?)\b|"
            r"\b(?:gpu|accelerator)(?:\s+model)?\s+"
            r"(?:is|was|:)?\s*(?:nvidia\s+)?[A-Z]\d{2,}\b",
            re.IGNORECASE,
        ),
        "context_limit": re.compile(
            r"\b\d[\d,.]*\s*(?:k|m)?\s*tokens?\b|"
            r"\bcontext.window\s*(?:size|limit)?\s*(?:is|was|:)?\s*"
            r"\d[\d,.]*\b",
            re.IGNORECASE,
        ),
        "model": re.compile(
            r"\bmodel\s+(?:is|was|called|named)\s+"
            r"[A-Z][\w.-]*(?:\s+[A-Z][\w.-]*)*"
        ),
    }

    # Check clauses separately so absence statements do not count as supplied
    # values. Only the two reviewed generic-context tails are allowed; other
    # non-refusal text is ambiguous and retains the old classification.
    affirmative_clauses = re.split(
        r"(?i)\b(?:but|however)\b|[.!?;]", answer
    )
    for clause in affirmative_clauses:
        if not clause.strip().strip("\"'“”‘’"):
            continue
        if any(pattern.search(clause) for pattern in absence_patterns):
            continue
        if any(
            positive_answer_patterns[category].search(clause)
            for category in categories
        ):
            return False
        if re.fullmatch(
            r"\s*it only describes pretraining as using\s+"
            r"[\"“]?a very large, diverse corpus of text[\"”]?\s*",
            clause,
            re.IGNORECASE,
        ) or re.fullmatch(
            r"\s*it only says that pretraining\s+"
            r"[\"“]?requires enormous computational resources[\"”]?\s*",
            clause,
            re.IGNORECASE,
        ):
            continue
        if clause.strip():
            return False

    return True


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
    active_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Calculate M4 evaluation metrics for one evaluation mode."""
    if active_fingerprint is not None:
        rows = [
            row for row in rows
            if row.get("model_fingerprint") == active_fingerprint
        ]

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

    answered_rows = [row for row in rows if not is_abstained(row)]
    unknown_answer_correct_rows = [
        row for row in answered_rows
        if not row["is_adversarial"]
        and not isinstance(row.get("answer_correct"), bool)
    ]

    if not answered_rows:
        hallucination_rate = None
        hallucination_rate_status = (
            "not measured: no answered queries; "
            f"{HALLUCINATION_RATE_STATUS}"
        )
    elif unknown_answer_correct_rows:
        hallucination_rate = None
        hallucination_rate_status = (
            "incomplete: answer_correct is missing, null, or non-boolean "
            "for answered "
            f"non-adversarial queries; {HALLUCINATION_RATE_STATUS}"
        )
    else:
        hallucinated_answered_rows = [
            row for row in answered_rows
            if row["is_adversarial"] or row.get("answer_correct") is False
        ]
        hallucination_rate = (
            len(hallucinated_answered_rows) / len(answered_rows)
        )
        hallucination_rate_status = HALLUCINATION_RATE_STATUS

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
    metadata_by_mode: dict[str, dict[str, Any]] | None = None,
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
            if metadata_by_mode is not None:
                row.update(metadata_by_mode[mode])
            rows.append(row)

            if log_run is not None:
                log_run(row)

        output["results"][mode] = rows
        fingerprint = (
            metadata_by_mode[mode]["model_fingerprint"]
            if metadata_by_mode is not None else None
        )
        output["metrics"][mode] = calculate_metrics(
            rows,
            active_fingerprint=fingerprint,
        )

    return output
