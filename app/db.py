
import sqlite3
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "traces.db"


def get_connection() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS attempts (
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
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                original_query TEXT NOT NULL,
                final_outcome TEXT NOT NULL,
                total_attempts INTEGER NOT NULL,
                total_latency_ms REAL,
                total_cost REAL,
                mode TEXT NOT NULL DEFAULT 'corrected',
                is_adversarial BOOLEAN NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Migrate existing databases without deleting stored traces.
        run_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(runs)").fetchall()
        }

        if "mode" not in run_columns:
            connection.execute("""
                ALTER TABLE runs
                ADD COLUMN mode TEXT NOT NULL DEFAULT 'corrected'
            """)

        if "is_adversarial" not in run_columns:
            connection.execute("""
                ALTER TABLE runs
                ADD COLUMN is_adversarial BOOLEAN NOT NULL DEFAULT 0
            """)
