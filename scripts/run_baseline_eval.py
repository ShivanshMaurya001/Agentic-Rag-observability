
import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from app.db import init_db
from app.embedding_service import client
from app.graph import graph
from app.retriever import retrieve
from app.trace_logger import log_attempt, log_run
from eval.baseline_runner import run_comparison
from eval.eval_dataset import load_eval_queries
from app.config import GENERATOR_MODEL


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_PATH = BASE_DIR / "eval" / "baseline_results.json"

# Shared cache for baseline's first retrieval and generation.
INITIAL_ATTEMPTS: dict[str, dict[str, Any]] = {}


def make_run_id(mode: str, query: str) -> str:
    """Create a stable ID using the expected evaluation prefix."""
    query_hash = hashlib.sha256(
        query.encode("utf-8")
    ).hexdigest()[:16]

    return f"eval-{mode}-{query_hash}"


def generate_answer(
    query: str,
    chunks: list[dict[str, Any]],
) -> str:
    context = "\n\n".join(chunk["text"] for chunk in chunks)

    prompt = f"""
You are an assistant that answers questions about a document.
Use ONLY the information provided in the context.
If the answer is not present, say:
"The answer was not found in the document."

Context:
{context}

Question:
{query}

Answer:
""".strip()

    response = client.models.generate_content(
        model=GENERATOR_MODEL,
        contents=prompt,
    )

    if response is None or response.text is None:
        raise ValueError("Generator returned no answer.")

    return response.text.strip()


def run_baseline(
    item: dict[str, Any],
    initial_attempts: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run baseline and cache its initial retrieval and generation."""

    if initial_attempts is None:
        initial_attempts = INITIAL_ATTEMPTS

    retrieve_start = time.perf_counter()
    chunks = retrieve(item["query"])
    retrieve_latency_ms = (
        time.perf_counter() - retrieve_start
    ) * 1000

    generate_start = time.perf_counter()
    answer = generate_answer(item["query"], chunks)
    generate_latency_ms = (
        time.perf_counter() - generate_start
    ) * 1000

    attempt = {
        "attempt_number": 0,
        "query_used": item["query"],
        "retrieved_chunks": chunks,
        "generated_answer": answer,
        "judge_verdict": {},
        "retrieve_latency_ms": retrieve_latency_ms,
        "generate_latency_ms": generate_latency_ms,
        "judge_latency_ms": 0.0,
    }

    initial_attempts[item["query"]] = {
        "retrieved_chunks": chunks,
        "answer": answer,
        "retrieve_latency_ms": retrieve_latency_ms,
        "generate_latency_ms": generate_latency_ms,
    }

    return {
        "answer": answer,
        "final_outcome": "answered",
        "total_attempts": 1,
        "retrieved_chunks": chunks,
        "retrieved_chunk_ids": [
            chunk["chunk_id"] for chunk in chunks
        ],
        "attempts": [attempt],
        "is_hallucination": None,
    }


def run_corrected(
    item: dict[str, Any],
    initial_attempts: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the corrective graph, reusing baseline's first attempt."""

    if initial_attempts is None:
        initial_attempts = INITIAL_ATTEMPTS

    initial = initial_attempts.get(item["query"])

    initial_state = {
        "original_query": item["query"],
        "current_query": item["query"],
        "retrieved_chunks": (
            initial["retrieved_chunks"] if initial else []
        ),
        "answer": initial["answer"] if initial else "",
        "judge_verdict": None,
        "attempt_number": 0,
        "failure_type": [],
        "final_outcome": None,
        "attempts": [],
        "retrieve_latency_ms": (
            initial["retrieve_latency_ms"] if initial else 0.0
        ),
        "generate_latency_ms": (
            initial["generate_latency_ms"] if initial else 0.0
        ),
        "reuse_initial_attempt": initial is not None,
    }

    final_state = graph.invoke(initial_state)
    chunks = final_state.get("retrieved_chunks", [])

    return {
        "answer": final_state.get("answer", ""),
        "final_outcome": final_state.get(
            "final_outcome", "answered"
        ),
        "total_attempts": len(final_state.get("attempts", [])),
        "retrieved_chunks": chunks,
        "retrieved_chunk_ids": [
            chunk["chunk_id"] for chunk in chunks
        ],
        "attempts": final_state.get("attempts", []),
        "is_hallucination": None,
    }


def load_saved_results() -> dict[str, Any]:
    if not RESULTS_PATH.exists():
        return {
            "results": {"baseline": [], "corrected": []},
            "metrics": {},
        }

    with RESULTS_PATH.open("r", encoding="utf-8") as file:
        saved = json.load(file)

    saved.setdefault("results", {})
    saved["results"].setdefault("baseline", [])
    saved["results"].setdefault("corrected", [])
    saved.setdefault("metrics", {})

    return saved


def main() -> None:
    global INITIAL_ATTEMPTS

    parser = argparse.ArgumentParser(
        description="Run M4 baseline vs corrected evaluation."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N queries.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse saved query/mode results and run only missing pairs.",
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    init_db()

    dataset = load_eval_queries()
    if args.limit is not None:
        dataset = dataset[:args.limit]

    saved = load_saved_results() if args.resume else {
        "results": {"baseline": [], "corrected": []},
        "metrics": {},
    }

    cached: dict[tuple[str, str], dict[str, Any]] = {}

    for mode in ("baseline", "corrected"):
        for row in saved["results"].get(mode, []):
            cached[(mode, row["query"])] = row

    # Reset the in-memory cache for this evaluation run.
    INITIAL_ATTEMPTS = {}

    # Restore reusable baseline attempts from saved results.
    for row in saved["results"].get("baseline", []):
        attempts = row.get("attempts", [])
        first_attempt = attempts[0] if attempts else {}

        chunks = first_attempt.get(
            "retrieved_chunks",
            row.get("retrieved_chunks", []),
        )
        answer = first_attempt.get(
            "generated_answer",
            row.get("answer", ""),
        )

        if chunks and answer:
            INITIAL_ATTEMPTS[row["query"]] = {
                "retrieved_chunks": chunks,
                "answer": answer,
                "retrieve_latency_ms": first_attempt.get(
                    "retrieve_latency_ms", 0.0
                ),
                "generate_latency_ms": first_attempt.get(
                    "generate_latency_ms", 0.0
                ),
            }

    def cached_or_run(mode: str, actual_runner):
        def runner(item: dict[str, Any]) -> dict[str, Any]:
            previous = cached.get((mode, item["query"]))

            if previous is not None:
                return previous

            # Both production runners accept the item as their first argument.
            # Their optional cache argument defaults to INITIAL_ATTEMPTS.
            return actual_runner(item)

        return runner

    def persist_row(row: dict[str, Any]) -> None:
        mode = row["mode"]
        query = row["query"]

        # Saved rows have already been persisted.
        if (mode, query) in cached:
            return

        run_id = make_run_id(mode, query)
        attempts = row.get("attempts", [])

        # Compatibility fallback for older result shapes.
        if not attempts:
            attempts = [{
                "attempt_number": max(
                    row.get("total_attempts", 1) - 1, 0
                ),
                "query_used": query,
                "retrieved_chunks": row.get(
                    "retrieved_chunks", []
                ),
                "generated_answer": row.get("answer", ""),
                "judge_verdict": {},
                "retrieve_latency_ms": 0.0,
                "generate_latency_ms": 0.0,
                "judge_latency_ms": 0.0,
            }]

        for attempt in attempts:
            log_attempt(
                run_id=run_id,
                attempt_number=attempt["attempt_number"],
                query_used=attempt.get("query_used", query),
                retrieved_chunks=attempt.get(
                    "retrieved_chunks", []
                ),
                generated_answer=attempt.get(
                    "generated_answer", ""
                ),
                judge_verdict=attempt.get("judge_verdict") or {},
                retrieve_latency_ms=attempt.get(
                    "retrieve_latency_ms", 0.0
                ),
                generate_latency_ms=attempt.get(
                    "generate_latency_ms", 0.0
                ),
                judge_latency_ms=attempt.get(
                    "judge_latency_ms", 0.0
                ),
            )

        log_run(
            run_id=run_id,
            original_query=query,
            final_outcome=row["final_outcome"],
            total_attempts=row.get(
                "total_attempts", len(attempts)
            ),
            mode=mode,
            is_adversarial=row["is_adversarial"],
        )

        cached[(mode, query)] = row

    comparison = run_comparison(
        run_baseline=cached_or_run("baseline", run_baseline),
        run_corrected=cached_or_run("corrected", run_corrected),
        log_run=persist_row,
        dataset=dataset,
    )

    # Merge updated rows without discarding saved queries outside --limit.
    saved_rows: dict[str, list[dict[str, Any]]] = {}

    for mode in ("baseline", "corrected"):
        merged = {
            row["query"]: row
            for row in saved["results"].get(mode, [])
        }

        for row in comparison["results"][mode]:
            merged[row["query"]] = row

        saved_rows[mode] = list(merged.values())

    output = {
        "results": saved_rows,
        "metrics": comparison["metrics"],
    }

    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print(json.dumps(output["metrics"], indent=2))
    print(f"Saved evaluation results to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
