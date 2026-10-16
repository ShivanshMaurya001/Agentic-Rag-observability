
import time
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.db import get_connection, init_db
from app.graph import graph
from app.langsmith_config import configure_langsmith
from app.trace_logger import log_attempt, log_run


app = FastAPI(title="Agentic RAG Observability API")


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)


@app.on_event("startup")
def startup() -> None:
    init_db()
    configure_langsmith()


@app.post("/query")
def run_query(request: QueryRequest) -> dict:
    run_id = str(uuid.uuid4())

    state = {
        "original_query": request.query,
        "current_query": request.query,
        "retrieved_chunks": [],
        "answer": "",
        "judge_verdict": None,
        "attempt_number": 0,
        "failure_type": [],
        "final_outcome": None,
    }

    stage_latency = {
        "retrieve": 0.0,
        "generate": 0.0,
        "judge": 0.0,
    }
    attempt_count = 0
    total_latency_ms = 0.0
    last_tick = time.perf_counter()

    try:
        updates = graph.stream(
            state,
            config={
                "run_name": "agentic-rag-query",
                "run_id": uuid.UUID(run_id),
                "metadata": {"original_query": request.query},
            },
            stream_mode="updates",
        )

        for update in updates:
            now = time.perf_counter()
            elapsed_ms = (now - last_tick) * 1000
            last_tick = now

            for node_name, node_update in update.items():
                if node_name in stage_latency:
                    stage_latency[node_name] += elapsed_ms
                    total_latency_ms += elapsed_ms

                state.update(node_update)

                if node_name == "judge":
                    verdict = state["judge_verdict"]
                    if verdict is None:
                        raise ValueError("Judge node returned no verdict.")

                    attempt_count += 1

                    log_attempt(
                        run_id=run_id,
                        attempt_number=state["attempt_number"],
                        query_used=state["current_query"],
                        retrieved_chunks=state["retrieved_chunks"],
                        generated_answer=state["answer"],
                        judge_verdict=verdict.model_dump(),
                        retrieve_latency_ms=stage_latency["retrieve"],
                        generate_latency_ms=stage_latency["generate"],
                        judge_latency_ms=stage_latency["judge"],
                    )

                    # Reset stage timers for the next attempt.
                    stage_latency = {
                        "retrieve": 0.0,
                        "generate": 0.0,
                        "judge": 0.0,
                    }

        if not state["final_outcome"]:
            raise ValueError("Graph finished without a final outcome.")

        log_run(
            run_id=run_id,
            original_query=request.query,
            final_outcome=state["final_outcome"],
            total_attempts=attempt_count,
            total_latency_ms=total_latency_ms,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Query processing or trace logging failed.",
        ) from exc

    return {
        "run_id": run_id,
        "query": request.query,
        "answer": state["answer"],
        "final_outcome": state["final_outcome"],
    }


@app.get("/traces/{run_id}")
def get_trace(run_id: str) -> dict:
    with get_connection() as connection:
        run = connection.execute(
            "SELECT * FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()

        attempts = connection.execute(
            """
            SELECT * FROM attempts
            WHERE run_id = ?
            ORDER BY id
            """,
            (run_id,),
        ).fetchall()

    if run is None:
        raise HTTPException(status_code=404, detail="Trace not found.")

    return {
        "run": dict(run),
        "attempts": [dict(attempt) for attempt in attempts],
    }


@app.get("/traces")
def list_traces(limit: int = 20, offset: int = 0) -> dict:
    if limit < 1 or limit > 100 or offset < 0:
        raise HTTPException(
            status_code=400,
            detail="Use limit 1–100 and a non-negative offset.",
        )

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT * FROM runs
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()

    return {
        "limit": limit,
        "offset": offset,
        "runs": [dict(row) for row in rows],
    }
