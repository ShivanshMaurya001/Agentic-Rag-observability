import sqlite3

import pytest

from dashboard import queries as dashboard_queries
from dashboard.queries import (
    NOT_RECORDED,
    NOT_STORED,
    POPULATION_LABEL,
    get_answered_abstained_breakdown,
    get_average_retries,
    get_decision_trace,
    get_failure_type_breakdown,
    get_final_outcome_split,
    get_latency_cost_by_attempt_number,
    get_mode_comparison,
    get_per_stage_latency_cost_breakdown,
    get_retry_percentage,
    get_run_summaries,
    get_total_runs,
    get_unavailable_evaluation_metrics,
    open_readonly_connection,
)


def _create_database(path):
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE runs (
                run_id TEXT PRIMARY KEY,
                original_query TEXT NOT NULL,
                final_outcome TEXT NOT NULL,
                total_attempts INTEGER NOT NULL,
                total_latency_ms REAL,
                total_cost REAL,
                mode TEXT NOT NULL DEFAULT 'corrected',
                is_adversarial BOOLEAN NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                attempt_number INTEGER NOT NULL,
                query_used TEXT NOT NULL,
                retrieved_chunks TEXT NOT NULL,
                generated_answer TEXT NOT NULL,
                judge_grounded BOOLEAN,
                judge_retrieval_quality REAL,
                judge_answer_quality REAL,
                judge_failure_type TEXT,
                judge_reason TEXT,
                retrieve_latency_ms REAL,
                generate_latency_ms REAL,
                judge_latency_ms REAL,
                retrieve_cost REAL,
                generate_cost REAL,
                judge_cost REAL,
                latency_ms REAL,
                cost REAL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )


@pytest.fixture
def database_path(tmp_path):
    path = tmp_path / "dashboard_traces.db"
    _create_database(path)
    with sqlite3.connect(path) as connection:
        connection.executemany(
            """INSERT INTO runs (
                   run_id, original_query, final_outcome, total_attempts,
                   total_latency_ms, total_cost, mode, is_adversarial
               ) VALUES (?, ?, ?, ?, 0, 0, ?, ?)""",
            [
                ("b1", "What is RAG?", "answered", 1, "baseline", False),
                ("b2", "What was the benchmark score?", "abstained_judge_rejected", 1, "baseline", True),
                ("c1", "What was the benchmark score?", "answered_after_retry", 2, "corrected", True),
                ("c2", "Who won the adversarial contest?", "abstained_judge_rejected", 3, "corrected", True),
                ("c3", "What is a language model?", "answered", 1, "corrected", False),
            ],
        )
        connection.executemany(
            """INSERT INTO attempts (
                   run_id, attempt_number, query_used, retrieved_chunks,
                   generated_answer, judge_failure_type, judge_reason,
                   retrieve_latency_ms, generate_latency_ms, judge_latency_ms,
                   retrieve_cost, generate_cost, judge_cost, latency_ms, cost
               ) VALUES (?, ?, ?, '[]', ?, ?, ?, 10, 20, 30, 0, 0, 0, ?, 0)""",
            [
                ("b1", 0, "query used 0", "RAG retrieves relevant documents.", "none", "reason 0", 100),
                ("b2", 0, "query used 0", "No answer.", "none", "reason 0", 80),
                ("c1", 0, "query used 0", "The answer was not found in the document.", "retrieval", "reason 0", 90),
                ("c1", 1, "query used 1", "The document does not provide a benchmark score.", "none", "reason 1", 110),
                ("c2", 0, "query used 0", "No answer.", "retrieval", "reason 0", 100),
                ("c2", 1, "query used 1", "Still no answer.", "retrieval", "reason 1", 120),
                ("c2", 2, "query used 2", "No answer.", "retrieval", "reason 2", 130),
                ("c3", 0, "query used 0", "A model trained on text.", "none", "reason 0", 70),
            ],
        )
    return path


@pytest.fixture
def connection(database_path):
    database = open_readonly_connection(database_path)
    try:
        yield database
    finally:
        database.close()


def test_total_runs_and_retry_percentage(connection):
    assert get_total_runs(connection) == 5
    assert get_retry_percentage(connection) == 40.0
    assert get_retry_percentage(connection, "baseline") == 0.0
    assert get_retry_percentage(connection, "corrected") == pytest.approx(200 / 3)


def test_failure_type_breakdown_excludes_none(connection):
    assert get_failure_type_breakdown(connection) == {"retrieval": 4}
    assert get_failure_type_breakdown(connection, "baseline") == {}
    assert get_failure_type_breakdown(connection, "corrected") == {"retrieval": 4}


def test_final_outcome_split(connection):
    assert get_final_outcome_split(connection) == {
        "answered": 2,
        "answered_after_retry": 1,
        "abstained_judge_rejected": 2,
    }
    assert get_final_outcome_split(connection, "baseline") == {
        "answered": 1,
        "abstained_judge_rejected": 1,
    }


def test_decision_trace_returns_ordered_attempts(connection):
    trace = get_decision_trace(connection, "c2")

    assert trace["run"]["original_query"] == "Who won the adversarial contest?"
    assert [attempt["attempt_number"] for attempt in trace["attempts"]] == [0, 1, 2]
    assert [attempt["generated_answer"] for attempt in trace["attempts"]] == [
        "No answer.",
        "Still no answer.",
        "No answer.",
    ]
    assert get_decision_trace(connection, "missing") is None


def test_run_summaries_returns_one_per_run_and_uses_final_attempt(connection):
    summaries = get_run_summaries(connection)
    by_id = {summary["run_id"]: summary for summary in summaries}

    assert len(summaries) == 5
    assert len(by_id) == 5
    assert by_id["c1"] == {
        "run_id": "c1",
        "original_query": "What was the benchmark score?",
        "final_outcome": "answered_after_retry",
        "mode": "corrected",
        "total_attempts": 2,
        "query_used": "query used 1",
        "generated_answer": "The document does not provide a benchmark score.",
    }


def test_run_summaries_handles_run_without_attempts(tmp_path):
    path = tmp_path / "run_without_attempts.db"
    _create_database(path)
    with sqlite3.connect(path) as database:
        database.execute(
            """INSERT INTO runs (
                   run_id, original_query, final_outcome, total_attempts,
                   mode, is_adversarial
               ) VALUES ('empty', 'No trace yet?', 'answered', 0, 'baseline', 0)"""
        )

    with open_readonly_connection(path) as database:
        assert get_run_summaries(database) == [{
            "run_id": "empty",
            "original_query": "No trace yet?",
            "final_outcome": "answered",
            "mode": "baseline",
            "total_attempts": 0,
            "query_used": None,
            "generated_answer": None,
        }]


def test_latency_and_zero_placeholder_cost_by_attempt_number(connection):
    rows = get_latency_cost_by_attempt_number(connection)

    assert rows == [
        {
            "attempt_number": 0,
            "attempt_count": 5,
            "average_latency_ms": 88.0,
            "average_cost": NOT_RECORDED,
        },
        {
            "attempt_number": 1,
            "attempt_count": 2,
            "average_latency_ms": 115.0,
            "average_cost": NOT_RECORDED,
        },
        {
            "attempt_number": 2,
            "attempt_count": 1,
            "average_latency_ms": 130.0,
            "average_cost": NOT_RECORDED,
        },
    ]


def test_mode_comparison_uses_unmatched_all_recorded_runs(connection):
    result = get_mode_comparison(connection)

    assert result["population_label"] == POPULATION_LABEL
    assert result["modes"]["baseline"] == {
        "n_runs": 2,
        "adversarial_n": 1,
        "retry_percentage": 0.0,
        "answered_n": 1,
        "abstained_n": 1,
        "false_answer_rate": 0.0,
        "correct_abstention_rate": 1.0,
        "judge_approved_after_retry": None,
        "judge_approved_after_retry_n": 0,
    }
    assert result["modes"]["corrected"] == {
        "n_runs": 3,
        "adversarial_n": 2,
        "retry_percentage": pytest.approx(200 / 3),
        "answered_n": 1,
        "abstained_n": 2,
        "false_answer_rate": 0.0,
        "correct_abstention_rate": 1.0,
        "judge_approved_after_retry": 0.5,
        "judge_approved_after_retry_n": 2,
    }


def test_answered_abstained_breakdown_uses_existing_helper(connection):
    result = get_answered_abstained_breakdown(connection)

    assert result["population_label"] == POPULATION_LABEL
    assert result["all_recorded_runs"] == {
        "total": 5,
        "answered": 2,
        "abstained": 3,
    }
    assert result["baseline"] == {"total": 2, "answered": 1, "abstained": 1}
    assert result["corrected"] == {"total": 3, "answered": 1, "abstained": 2}


def test_per_stage_latency_and_cost_breakdown(connection):
    assert get_per_stage_latency_cost_breakdown(connection) == [
        {
            "stage": "retrieval",
            "attempt_count": 8,
            "average_latency_ms": 10.0,
            "average_cost": NOT_RECORDED,
        },
        {
            "stage": "generation",
            "attempt_count": 8,
            "average_latency_ms": 20.0,
            "average_cost": NOT_RECORDED,
        },
        {
            "stage": "judge",
            "attempt_count": 8,
            "average_latency_ms": 30.0,
            "average_cost": NOT_RECORDED,
        },
    ]


def test_average_retries(connection):
    assert get_average_retries(connection) == pytest.approx(0.6)
    assert get_average_retries(connection, "baseline") == 0.0
    assert get_average_retries(connection, "corrected") == 1.0


def test_unavailable_evaluation_metrics_are_not_read_from_offline_json():
    assert get_unavailable_evaluation_metrics() == {
        "Recall@K": NOT_STORED,
        "Hit Rate@K": NOT_STORED,
        "Answer Correctness": NOT_STORED,
        "Hallucination Rate": NOT_STORED,
    }


def test_empty_database_shapes_and_missing_run_handling(tmp_path):
    path = tmp_path / "empty_traces.db"
    _create_database(path)
    with open_readonly_connection(path) as database:
        assert get_total_runs(database) == 0
        assert get_retry_percentage(database) is None
        assert get_failure_type_breakdown(database) == {}
        assert get_final_outcome_split(database) == {}
        assert get_decision_trace(database, "missing") is None
        assert get_latency_cost_by_attempt_number(database) == []
        assert get_average_retries(database) is None
        assert get_answered_abstained_breakdown(database)["all_recorded_runs"] == {
            "total": 0,
            "answered": 0,
            "abstained": 0,
        }
        assert all(
            row["average_latency_ms"] is None
            and row["average_cost"] == NOT_RECORDED
            for row in get_per_stage_latency_cost_breakdown(database)
        )


def test_connection_is_read_only_and_queries_do_not_change_database(database_path):
    before = database_path.read_bytes()
    with open_readonly_connection(database_path) as database:
        with pytest.raises(sqlite3.OperationalError):
            database.execute("UPDATE runs SET final_outcome = 'changed'")
        assert get_total_runs(database) == 5
    assert database_path.read_bytes() == before


def test_connection_uses_traces_db_path_environment_variable(tmp_path, monkeypatch):
    environment_path = tmp_path / "environment_traces.db"
    _create_database(environment_path)
    with sqlite3.connect(environment_path) as connection:
        connection.execute(
            """INSERT INTO runs (
                   run_id, original_query, final_outcome, total_attempts,
                   mode, is_adversarial
               ) VALUES ('env-run', 'Environment database', 'answered', 1, 'baseline', 0)"""
        )
    monkeypatch.setenv("TRACES_DB_PATH", str(environment_path))

    with open_readonly_connection() as database:
        assert get_total_runs(database) == 1
        with pytest.raises(sqlite3.OperationalError):
            database.execute("UPDATE runs SET final_outcome = 'changed'")


def test_connection_falls_back_to_default_db_path(tmp_path, monkeypatch):
    default_path = tmp_path / "default_traces.db"
    _create_database(default_path)
    monkeypatch.delenv("TRACES_DB_PATH", raising=False)
    monkeypatch.setattr(dashboard_queries, "DB_PATH", default_path)

    with open_readonly_connection() as database:
        assert get_total_runs(database) == 0


def test_explicit_connection_path_takes_precedence_over_environment(
    database_path, tmp_path, monkeypatch
):
    environment_path = tmp_path / "environment_traces.db"
    _create_database(environment_path)
    monkeypatch.setenv("TRACES_DB_PATH", str(environment_path))

    with open_readonly_connection(database_path) as database:
        assert get_total_runs(database) == 5
