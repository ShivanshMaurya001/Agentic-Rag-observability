import json
import sqlite3
from typing import Any

from app.db import get_connection


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
    total_latency_ms = (
        retrieve_latency_ms + generate_latency_ms + judge_latency_ms
    )
    total_cost = retrieve_cost + generate_cost + judge_cost

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO attempts (
                run_id, attempt_number, query_used, retrieved_chunks,
                generated_answer, judge_grounded, judge_retrieval_quality,
                judge_answer_quality, judge_failure_type, judge_reason,
                retrieve_latency_ms, generate_latency_ms, judge_latency_ms,
                retrieve_cost, generate_cost, judge_cost, latency_ms, cost
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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


def log_run(
    run_id: str,
    original_query: str,
    final_outcome: str,
    total_attempts: int,
    total_latency_ms: float = 0,
    total_cost: float = 0,
) -> None:
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO runs (
                run_id, original_query, final_outcome, total_attempts,
                total_latency_ms, total_cost
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                original_query,
                final_outcome,
                total_attempts,
                total_latency_ms,
                total_cost,
            ),
        )

