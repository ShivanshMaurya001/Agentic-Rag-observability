
import json
from typing import Any

from app.db import get_connection


def _insert_attempt(
    connection,
    run_id: str,
    attempt_number: int,
    query_used: str,
    retrieved_chunks: list[dict[str, Any]],
    generated_answer: str,
    judge_verdict: dict[str, Any],
    retrieve_latency_ms: float = 0,
    generate_latency_ms: float = 0,
    judge_latency_ms: float = 0,
    retrieve_cost: float = 0,
    generate_cost: float = 0,
    judge_cost: float = 0,
) -> None:
    total_latency_ms = (
        retrieve_latency_ms + generate_latency_ms + judge_latency_ms
    )
    total_cost = retrieve_cost + generate_cost + judge_cost

    connection.execute(
        """
        INSERT INTO attempts (
            run_id, attempt_number, query_used, retrieved_chunks,
            generated_answer, judge_grounded, judge_retrieval_quality,
            judge_answer_quality, judge_failure_type, judge_reason,
            retrieve_latency_ms, generate_latency_ms, judge_latency_ms,
            retrieve_cost, generate_cost, judge_cost, latency_ms, cost
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            attempt_number,
            query_used,
            json.dumps(retrieved_chunks),
            generated_answer,
            judge_verdict.get("grounded"),
            judge_verdict.get("retrieval_quality"),
            judge_verdict.get("answer_quality"),
            judge_verdict.get("failure_type"),
            judge_verdict.get("reason"),
            retrieve_latency_ms,
            generate_latency_ms,
            judge_latency_ms,
            retrieve_cost,
            generate_cost,
            judge_cost,
            total_latency_ms,
            total_cost,
        ),
    )


def log_attempt(
    run_id: str,
    attempt_number: int,
    query_used: str,
    retrieved_chunks: list[dict[str, Any]],
    generated_answer: str,
    judge_verdict: dict[str, Any],
    retrieve_latency_ms: float = 0,
    generate_latency_ms: float = 0,
    judge_latency_ms: float = 0,
    retrieve_cost: float = 0,
    generate_cost: float = 0,
    judge_cost: float = 0,
) -> None:
    with get_connection() as connection:
        _insert_attempt(
            connection=connection,
            run_id=run_id,
            attempt_number=attempt_number,
            query_used=query_used,
            retrieved_chunks=retrieved_chunks,
            generated_answer=generated_answer,
            judge_verdict=judge_verdict,
            retrieve_latency_ms=retrieve_latency_ms,
            generate_latency_ms=generate_latency_ms,
            judge_latency_ms=judge_latency_ms,
            retrieve_cost=retrieve_cost,
            generate_cost=generate_cost,
            judge_cost=judge_cost,
        )


def log_run(
    run_id: str,
    original_query: str,
    final_outcome: str,
    total_attempts: int,
    total_latency_ms: float = 0,
    total_cost: float = 0,
    mode: str = "corrected",
    is_adversarial: bool = False,
) -> None:
    if mode not in {"baseline", "corrected"}:
        raise ValueError("mode must be 'baseline' or 'corrected'")

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO runs (
                run_id, original_query, final_outcome, total_attempts,
                total_latency_ms, total_cost, mode, is_adversarial
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                original_query = excluded.original_query,
                final_outcome = excluded.final_outcome,
                total_attempts = excluded.total_attempts,
                total_latency_ms = excluded.total_latency_ms,
                total_cost = excluded.total_cost,
                mode = excluded.mode,
                is_adversarial = excluded.is_adversarial
            """,
            (
                run_id,
                original_query,
                final_outcome,
                total_attempts,
                total_latency_ms,
                total_cost,
                mode,
                is_adversarial,
            ),
        )


def log_evaluation_row(
    run_id: str,
    original_query: str,
    final_outcome: str,
    total_attempts: int,
    attempts: list[dict[str, Any]],
    total_latency_ms: float = 0,
    total_cost: float = 0,
    mode: str = "corrected",
    is_adversarial: bool = False,
) -> None:
    """Save one evaluation row atomically and safely handle reruns."""
    if mode not in {"baseline", "corrected"}:
        raise ValueError("mode must be 'baseline' or 'corrected'")

    with get_connection() as connection:
        # Replace attempts belonging to this exact run only.
        # Other run IDs and their traces remain untouched.
        connection.execute(
            "DELETE FROM attempts WHERE run_id = ?",
            (run_id,),
        )

        for attempt in attempts:
            verdict = attempt.get("judge_verdict") or {}

            _insert_attempt(
                connection=connection,
                run_id=run_id,
                attempt_number=attempt.get("attempt_number", 0),
                query_used=attempt.get("query_used", original_query),
                retrieved_chunks=attempt.get("retrieved_chunks", []),
                generated_answer=attempt.get("generated_answer", ""),
                judge_verdict=verdict,
                retrieve_latency_ms=attempt.get("retrieve_latency_ms", 0),
                generate_latency_ms=attempt.get("generate_latency_ms", 0),
                judge_latency_ms=attempt.get("judge_latency_ms", 0),
                retrieve_cost=attempt.get("retrieve_cost", 0),
                generate_cost=attempt.get("generate_cost", 0),
                judge_cost=attempt.get("judge_cost", 0),
            )

        connection.execute(
            """
            INSERT INTO runs (
                run_id, original_query, final_outcome, total_attempts,
                total_latency_ms, total_cost, mode, is_adversarial
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                original_query = excluded.original_query,
                final_outcome = excluded.final_outcome,
                total_attempts = excluded.total_attempts,
                total_latency_ms = excluded.total_latency_ms,
                total_cost = excluded.total_cost,
                mode = excluded.mode,
                is_adversarial = excluded.is_adversarial
            """,
            (
                run_id,
                original_query,
                final_outcome,
                total_attempts,
                total_latency_ms,
                total_cost,
                mode,
                is_adversarial,
            ),
        )