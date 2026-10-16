import sqlite3

from app import db
from app.trace_logger import log_attempt, log_run


def test_log_attempt_writes_row(tmp_path, monkeypatch):
    test_db = tmp_path / "test_traces.db"

    def get_test_connection():
        connection = sqlite3.connect(test_db)
        connection.row_factory = sqlite3.Row
        return connection

    monkeypatch.setattr(db, "get_connection", get_test_connection)
    monkeypatch.setattr(
        "app.trace_logger.get_connection",
        get_test_connection,
    )
    db.init_db()

    log_attempt(
        run_id="run-1",
        attempt_number=0,
        query_used="What is RAG?",
        retrieved_chunks=[
            {"chunk_id": "chunk-1", "text": "RAG combines retrieval and generation."}
        ],
        generated_answer="RAG combines retrieval and generation.",
        judge_verdict={
            "grounded": True,
            "retrieval_quality": 0.9,
            "answer_quality": 0.95,
            "failure_type": "none",
            "reason": "The answer is supported by the context.",
        },
        retrieve_latency_ms=10,
        generate_latency_ms=20,
        judge_latency_ms=30,
        retrieve_cost=0.001,
        generate_cost=0.002,
        judge_cost=0.003,
    )

    with get_test_connection() as connection:
        row = connection.execute(
            "SELECT * FROM attempts WHERE run_id = ?",
            ("run-1",),
        ).fetchone()

    assert row is not None
    assert row["query_used"] == "What is RAG?"
    assert row["judge_failure_type"] == "none"
    assert row["latency_ms"] == 60
    assert abs(row["cost"] - 0.006) < 1e-9


def test_log_run_writes_row(tmp_path, monkeypatch):
    test_db = tmp_path / "test_traces.db"

    def get_test_connection():
        connection = sqlite3.connect(test_db)
        connection.row_factory = sqlite3.Row
        return connection

    monkeypatch.setattr(db, "get_connection", get_test_connection)
    monkeypatch.setattr(
        "app.trace_logger.get_connection",
        get_test_connection,
    )
    db.init_db()

    log_run(
        run_id="run-2",
        original_query="What is RAG?",
        final_outcome="answered",
        total_attempts=1,
        total_latency_ms=60,
        total_cost=0.006,
    )

    with get_test_connection() as connection:
        row = connection.execute(
            "SELECT * FROM runs WHERE run_id = ?",
            ("run-2",),
        ).fetchone()

    assert row is not None
    assert row["original_query"] == "What is RAG?"
    assert row["final_outcome"] == "answered"
    assert row["total_attempts"] == 1