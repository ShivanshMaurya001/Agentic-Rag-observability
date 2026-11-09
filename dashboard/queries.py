"""Read-only SQLite queries used by the M5 dashboard."""

import os
import sqlite3
from pathlib import Path
from typing import Any

from app.db import DB_PATH
from eval.baseline_runner import is_abstained


POPULATION_LABEL = "All recorded runs — populations not matched"
NOT_STORED = "Not stored in traces.db"
NOT_RECORDED = "Not recorded"
MODES = ("baseline", "corrected")


def open_readonly_connection(
    db_path: str | Path | None = None,
) -> sqlite3.Connection:
    """Open an existing SQLite database without permitting writes."""
    selected_path = (
        db_path
        if db_path is not None
        else os.getenv("TRACES_DB_PATH") or DB_PATH
    )
    path = Path(selected_path).resolve()
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _check_mode(mode: str | None) -> None:
    if mode is not None and mode not in MODES:
        raise ValueError("mode must be 'baseline', 'corrected', or None")


def _mode_clause(mode: str | None, alias: str = "r") -> tuple[str, tuple[str, ...]]:
    _check_mode(mode)
    if mode is None:
        return "", ()
    return f" WHERE {alias}.mode = ?", (mode,)


def _run_rows(
    connection: sqlite3.Connection,
    mode: str | None = None,
) -> list[dict[str, Any]]:
    where, params = _mode_clause(mode)
    rows = connection.execute(
        f"""
        SELECT r.run_id, r.original_query AS query, r.final_outcome,
               r.mode, r.is_adversarial, r.total_attempts,
               latest.query_used AS query_used,
               latest.generated_answer AS answer
        FROM runs AS r
        LEFT JOIN attempts AS latest ON latest.id = (
            SELECT a.id
            FROM attempts AS a
            WHERE a.run_id = r.run_id
            ORDER BY a.attempt_number DESC, a.id DESC
            LIMIT 1
        )
        {where}
        ORDER BY r.created_at, r.run_id
        """,
        params,
    ).fetchall()
    return [dict(row) for row in rows]


def _classification_rows(
    connection: sqlite3.Connection,
    mode: str | None = None,
) -> list[dict[str, Any]]:
    result = _run_rows(connection, mode)
    for row in result:
        row["abstained"] = is_abstained(row)
    return result


def get_run_summaries(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    """Return one run summary with its final attempt fields, if present."""
    rows = _run_rows(connection)
    return [
        {
            "run_id": row["run_id"],
            "original_query": row["query"],
            "final_outcome": row["final_outcome"],
            "mode": row["mode"],
            "total_attempts": row["total_attempts"],
            "query_used": row["query_used"],
            "generated_answer": row["answer"],
        }
        for row in rows
    ]


def get_total_runs(connection: sqlite3.Connection) -> int:
    """Return all recorded runs, including repeated runs of a query."""
    return int(connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0])


def get_retry_percentage(
    connection: sqlite3.Connection,
    mode: str | None = None,
) -> float | None:
    """Return the percentage of recorded runs with more than one attempt."""
    where, params = _mode_clause(mode)
    row = connection.execute(
        f"""
        SELECT COUNT(*) AS total,
               SUM(CASE WHEN total_attempts > 1 THEN 1 ELSE 0 END) AS retried
        FROM runs AS r
        {where}
        """,
        params,
    ).fetchone()
    if not row["total"]:
        return None
    return 100.0 * row["retried"] / row["total"]


def get_failure_type_breakdown(
    connection: sqlite3.Connection,
    mode: str | None = None,
) -> dict[str, int]:
    """Count non-empty judge failure types, excluding the success value none."""
    where, params = _mode_clause(mode, alias="r")
    if where:
        where += " AND "
    else:
        where = " WHERE "
    rows = connection.execute(
        f"""
        SELECT a.judge_failure_type AS failure_type, COUNT(*) AS count
        FROM attempts AS a
        JOIN runs AS r ON r.run_id = a.run_id
        {where}a.judge_failure_type IS NOT NULL
          AND lower(a.judge_failure_type) != 'none'
        GROUP BY a.judge_failure_type
        ORDER BY count DESC, failure_type
        """,
        params,
    ).fetchall()
    return {row["failure_type"]: int(row["count"]) for row in rows}


def get_final_outcome_split(
    connection: sqlite3.Connection,
    mode: str | None = None,
) -> dict[str, int]:
    """Count runs by their saved final outcome."""
    where, params = _mode_clause(mode)
    rows = connection.execute(
        f"""
        SELECT r.final_outcome, COUNT(*) AS count
        FROM runs AS r
        {where}
        GROUP BY r.final_outcome
        ORDER BY r.final_outcome
        """,
        params,
    ).fetchall()
    return {row["final_outcome"]: int(row["count"]) for row in rows}


def get_decision_trace(
    connection: sqlite3.Connection,
    run_id: str,
) -> dict[str, Any] | None:
    """Return one run and its full attempt history in attempt order."""
    run = connection.execute(
        """
        SELECT run_id, original_query, final_outcome, total_attempts,
               total_latency_ms, total_cost, mode, is_adversarial, created_at
        FROM runs WHERE run_id = ?
        """,
        (run_id,),
    ).fetchone()
    if run is None:
        return None
    attempts = connection.execute(
        """
        SELECT id, attempt_number, query_used, retrieved_chunks,
               generated_answer, judge_grounded, judge_retrieval_quality,
               judge_answer_quality, judge_failure_type, judge_reason,
               retrieve_latency_ms, generate_latency_ms, judge_latency_ms,
               retrieve_cost, generate_cost, judge_cost, latency_ms, cost,
               created_at
        FROM attempts
        WHERE run_id = ?
        ORDER BY attempt_number, id
        """,
        (run_id,),
    ).fetchall()
    return {"run": dict(run), "attempts": [dict(row) for row in attempts]}


def get_latency_cost_by_attempt_number(
    connection: sqlite3.Connection,
    mode: str | None = None,
) -> list[dict[str, Any]]:
    """Return average attempt latency and cost, grouped by attempt number."""
    where, params = _mode_clause(mode, alias="r")
    rows = connection.execute(
        f"""
        SELECT a.attempt_number, COUNT(*) AS attempt_count,
               AVG(a.latency_ms) AS average_latency_ms,
               AVG(CASE WHEN a.cost != 0 THEN a.cost END) AS average_cost
        FROM attempts AS a
        JOIN runs AS r ON r.run_id = a.run_id
        {where}
        GROUP BY a.attempt_number
        ORDER BY a.attempt_number
        """,
        params,
    ).fetchall()
    return [
        {
            "attempt_number": row["attempt_number"],
            "attempt_count": int(row["attempt_count"]),
            "average_latency_ms": row["average_latency_ms"],
            "average_cost": (
                row["average_cost"]
                if row["average_cost"] is not None else NOT_RECORDED
            ),
        }
        for row in rows
    ]


def get_mode_comparison(connection: sqlite3.Connection) -> dict[str, Any]:
    """Compare all recorded baseline and corrected runs, not matched cohorts."""
    runs = _classification_rows(connection)
    result: dict[str, Any] = {"population_label": POPULATION_LABEL, "modes": {}}
    for mode in MODES:
        selected = [row for row in runs if row["mode"] == mode]
        adversarial = [row for row in selected if row["is_adversarial"]]
        abstained = [row for row in selected if row["abstained"]]
        false_answers = [
            row for row in adversarial if not row["abstained"]
        ]
        correct_abstentions = [
            row for row in adversarial if row["abstained"]
        ]
        retry_eligible = [row for row in selected if row["total_attempts"] > 1]
        approved_after_retry = [
            row for row in retry_eligible
            if row["final_outcome"] == "answered_after_retry"
        ]
        total = len(selected)
        result["modes"][mode] = {
            "n_runs": total,
            "adversarial_n": len(adversarial),
            "retry_percentage": (
                100.0 * len(retry_eligible) / total if total else None
            ),
            "answered_n": total - len(abstained),
            "abstained_n": len(abstained),
            "false_answer_rate": (
                len(false_answers) / len(adversarial) if adversarial else None
            ),
            "correct_abstention_rate": (
                len(correct_abstentions) / len(adversarial)
                if adversarial else None
            ),
            "judge_approved_after_retry": (
                len(approved_after_retry) / len(retry_eligible)
                if retry_eligible else None
            ),
            "judge_approved_after_retry_n": len(retry_eligible),
        }
    return result


def get_answered_abstained_breakdown(
    connection: sqlite3.Connection,
) -> dict[str, Any]:
    """Count answered and abstained runs overall and separately by mode."""
    comparison = get_mode_comparison(connection)
    by_mode = {
        mode: {
            "total": values["n_runs"],
            "answered": values["answered_n"],
            "abstained": values["abstained_n"],
        }
        for mode, values in comparison["modes"].items()
    }
    return {
        "population_label": comparison["population_label"],
        "all_recorded_runs": {
            key: sum(values[key] for values in by_mode.values())
            for key in ("total", "answered", "abstained")
        },
        **by_mode,
    }


def get_per_stage_latency_cost_breakdown(
    connection: sqlite3.Connection,
    mode: str | None = None,
) -> list[dict[str, Any]]:
    """Return average retrieval, generation, and judge latency/cost."""
    where, params = _mode_clause(mode, alias="r")
    row = connection.execute(
        f"""
        SELECT COUNT(*) AS attempt_count,
               AVG(a.retrieve_latency_ms) AS retrieval_latency,
               AVG(a.generate_latency_ms) AS generation_latency,
               AVG(a.judge_latency_ms) AS judge_latency,
               AVG(CASE WHEN a.retrieve_cost != 0 THEN a.retrieve_cost END)
                   AS retrieval_cost,
               AVG(CASE WHEN a.generate_cost != 0 THEN a.generate_cost END)
                   AS generation_cost,
               AVG(CASE WHEN a.judge_cost != 0 THEN a.judge_cost END)
                   AS judge_cost
        FROM attempts AS a
        JOIN runs AS r ON r.run_id = a.run_id
        {where}
        """,
        params,
    ).fetchone()
    stages = (
        ("retrieval", "retrieval_latency", "retrieval_cost"),
        ("generation", "generation_latency", "generation_cost"),
        ("judge", "judge_latency", "judge_cost"),
    )
    return [
        {
            "stage": name,
            "attempt_count": int(row["attempt_count"]),
            "average_latency_ms": row[latency],
            "average_cost": row[cost] if row[cost] is not None else NOT_RECORDED,
        }
        for name, latency, cost in stages
    ]


def get_average_retries(
    connection: sqlite3.Connection,
    mode: str | None = None,
) -> float | None:
    """Return average retries per recorded run (attempts minus one)."""
    where, params = _mode_clause(mode)
    row = connection.execute(
        f"""
        SELECT AVG(CASE WHEN total_attempts > 1 THEN total_attempts - 1 ELSE 0 END)
        FROM runs AS r
        {where}
        """,
        params,
    ).fetchone()
    return row[0]


def get_unavailable_evaluation_metrics() -> dict[str, str]:
    """Identify evaluation metrics that are absent from the SQLite schema."""
    return {
        "Recall@K": NOT_STORED,
        "Hit Rate@K": NOT_STORED,
        "Answer Correctness": NOT_STORED,
        "Hallucination Rate": NOT_STORED,
    }
