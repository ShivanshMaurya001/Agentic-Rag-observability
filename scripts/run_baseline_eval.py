
import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from app.db import init_db
from app.graph import graph
from app.llm_service import (
    generate_text,
    get_usage_summary,
    reset_usage_summary,
)
from app.retriever import retrieve
from app.trace_logger import log_evaluation_row
from eval.baseline_runner import calculate_metrics, run_comparison
from eval.eval_dataset import load_eval_queries
from app.config import GENERATOR_MODEL, JUDGE_MODEL, REWRITE_MODEL


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_PATH = BASE_DIR / "eval" / "baseline_results.json"

# Shared cache for baseline's first retrieval and generation.
INITIAL_ATTEMPTS: dict[str, dict[str, Any]] = {}
LLM_PROVIDER = "openai"


def model_metadata(mode: str) -> dict[str, Any]:
    """Return the providers and model IDs that define one mode's identity."""
    role_models = {
        "generator": GENERATOR_MODEL,
    }
    if mode == "corrected":
        role_models.update({
            "rewrite": REWRITE_MODEL,
            "judge": JUDGE_MODEL,
        })
    elif mode != "baseline":
        raise ValueError("mode must be 'baseline' or 'corrected'")

    providers = {role: LLM_PROVIDER for role in role_models}
    fingerprint_source = {
        role: {"provider": providers[role], "model": model}
        for role, model in role_models.items()
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_source,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "providers": providers,
        "models": role_models,
        "model_fingerprint": fingerprint,
    }


ACTIVE_METADATA = {
    mode: model_metadata(mode)
    for mode in ("baseline", "corrected")
}


def make_run_id(
    mode: str,
    query: str,
    fingerprint: str | None = None,
) -> str:
    """Create a stable ID separated by evaluation mode and model identity."""
    if fingerprint is None:
        fingerprint = ACTIVE_METADATA[mode]["model_fingerprint"]

    query_hash = hashlib.sha256(
        query.encode("utf-8")
    ).hexdigest()[:16]

    return f"eval-{mode}-{fingerprint}-{query_hash}"


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

    return generate_text(prompt, model=GENERATOR_MODEL).strip()


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


def load_saved_results(path: Path | None = None) -> dict[str, Any]:
    path = path or RESULTS_PATH
    if not path.exists():
        return {
            "results": {"baseline": [], "corrected": []},
            "metrics": {},
        }

    with path.open("r", encoding="utf-8") as file:
        saved = json.load(file)

    saved.setdefault("results", {})
    saved["results"].setdefault("baseline", [])
    saved["results"].setdefault("corrected", [])
    saved.setdefault("metrics", {})

    return saved


def merge_result_row(
    rows: list[dict[str, Any]],
    row: dict[str, Any],
) -> None:
    """Replace only the same query under the same explicit fingerprint."""
    fingerprint = row.get("model_fingerprint")
    for index, existing in enumerate(rows):
        if (
            existing.get("query") == row["query"]
            and existing.get("model_fingerprint") == fingerprint
        ):
            rows[index] = row
            return
    rows.append(row)


def _create_results_backup(path: Path) -> Path:
    """Copy existing results to a unique backup without overwriting one."""
    index = 0
    while True:
        suffix = f".phase2{'.' + str(index) if index else ''}.bak"
        backup_path = path.with_name(path.stem + suffix)
        created = False
        source = path.open("rb")
        try:
            descriptor = os.open(
                backup_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
            created = True
            with os.fdopen(descriptor, "wb") as target:
                shutil.copyfileobj(source, target)
            return backup_path
        except FileExistsError:
            index += 1
        except Exception:
            if created and backup_path.exists():
                backup_path.unlink()
            raise
        finally:
            source.close()


def checkpoint_results(
    path: Path,
    payload: dict[str, Any],
    backup_checked: bool = False,
) -> tuple[bool, Path | None]:
    """Back up once, then atomically replace the JSON checkpoint."""
    backup_path = None
    if not backup_checked and path.exists():
        backup_path = _create_results_backup(path)
    backup_checked = True

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(payload, temporary, indent=2, ensure_ascii=False)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    return backup_checked, backup_path


def refresh_active_metrics(
    saved: dict[str, Any],
    metadata_by_mode: dict[str, dict[str, Any]],
) -> None:
    """Summarize only rows matching each mode's active fingerprint."""
    metrics = saved.setdefault("metrics", {})
    results = saved.setdefault("results", {})
    for mode in ("baseline", "corrected"):
        active_rows = [
            row for row in results.get(mode, [])
            if row.get("model_fingerprint")
            == metadata_by_mode[mode]["model_fingerprint"]
        ]
        metrics[mode] = (
            calculate_metrics(active_rows)
            if active_rows else {}
        )


def main() -> None:
    global INITIAL_ATTEMPTS

    parser = argparse.ArgumentParser(
        description="Run M4 baseline vs corrected evaluation."
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--limit", type=int, default=None,
        help="Evaluate only the first N queries.",
    )
    selection.add_argument(
        "--query-indices", type=int, nargs="+",
        help="Evaluate only these zero-based dataset positions.",
    )
    parser.add_argument(
        "--results-path", type=Path, default=RESULTS_PATH,
        help="JSON results/checkpoint path.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse saved query/mode results and run only missing pairs.",
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    if args.query_indices is not None and (
        len(set(args.query_indices)) != len(args.query_indices)
        or any(index < 0 for index in args.query_indices)
    ):
        parser.error("--query-indices must contain unique non-negative positions")

    results_path = args.results_path
    if (
        args.query_indices is not None
        and results_path.resolve() == RESULTS_PATH.resolve()
    ):
        parser.error("--query-indices requires a separate --results-path")

    reset_usage_summary()
    init_db()

    dataset = load_eval_queries()
    if args.limit is not None:
        dataset = dataset[:args.limit]
    if args.query_indices is not None:
        if any(index >= len(dataset) for index in args.query_indices):
            parser.error("--query-indices contains a position outside the dataset")
        dataset = [dataset[index] for index in args.query_indices]

    saved = load_saved_results(results_path)

    cached: dict[tuple[str, str, str], dict[str, Any]] = {}

    for mode in ("baseline", "corrected"):
        for row in saved["results"].get(mode, []):
            fingerprint = ACTIVE_METADATA[mode]["model_fingerprint"]
            if args.resume and row.get("model_fingerprint") == fingerprint:
                cached[(mode, fingerprint, row["query"])] = row

    # Reset the in-memory cache for this evaluation run.
    INITIAL_ATTEMPTS = {}

    # Restore reusable baseline attempts from saved results.
    for row in saved["results"].get("baseline", []):
        if row.get("model_fingerprint") != ACTIVE_METADATA["baseline"]["model_fingerprint"]:
            continue
        if not args.resume:
            continue
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
            fingerprint = ACTIVE_METADATA[mode]["model_fingerprint"]
            previous = cached.get((mode, fingerprint, item["query"]))

            if previous is not None:
                return previous

            # Both production runners accept the item as their first argument.
            # Their optional cache argument defaults to INITIAL_ATTEMPTS.
            return actual_runner(item)

        return runner

    backup_checked = False

    def persist_row(row: dict[str, Any]) -> None:
        nonlocal backup_checked
        mode = row["mode"]
        query = row["query"]
        fingerprint = ACTIVE_METADATA[mode]["model_fingerprint"]
        cache_key = (mode, fingerprint, query)

        # Saved rows have already been persisted in this evaluation flow.
        if cache_key in cached:
            return

        run_id = make_run_id(mode, query, fingerprint)
        attempts = row.get("attempts", [])

        # Compatibility fallback for older result shapes.
        if not attempts:
            attempts = [
                {
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
                }
            ]

        total_latency_ms = sum(
            attempt.get("retrieve_latency_ms", 0.0)
            + attempt.get("generate_latency_ms", 0.0)
            + attempt.get("judge_latency_ms", 0.0)
            for attempt in attempts
        )

        total_cost = sum(
            attempt.get("retrieve_cost", 0.0)
            + attempt.get("generate_cost", 0.0)
            + attempt.get("judge_cost", 0.0)
            for attempt in attempts
        )

        # Persist the attempts and run summary together.
        # Existing run IDs are updated safely by log_evaluation_row().
        log_evaluation_row(
            run_id=run_id,
            original_query=query,
            final_outcome=row["final_outcome"],
            total_attempts=row.get(
                "total_attempts", len(attempts)
            ),
            attempts=attempts,
            total_latency_ms=total_latency_ms,
            total_cost=total_cost,
            mode=mode,
            is_adversarial=row["is_adversarial"],
        )

        cached[cache_key] = row
        merge_result_row(saved["results"].setdefault(mode, []), row)
        refresh_active_metrics(saved, ACTIVE_METADATA)
        backup_checked, _ = checkpoint_results(
            results_path,
            saved,
            backup_checked=backup_checked,
        )

    run_comparison(
        run_baseline=cached_or_run("baseline", run_baseline),
        run_corrected=cached_or_run("corrected", run_corrected),
        log_run=persist_row,
        dataset=dataset,
        metadata_by_mode=ACTIVE_METADATA,
    )

    refresh_active_metrics(saved, ACTIVE_METADATA)
    backup_checked, _ = checkpoint_results(
        results_path,
        saved,
        backup_checked=backup_checked,
    )

    print(json.dumps(saved["metrics"], indent=2))
    print(f"Saved evaluation results to: {results_path}")
    print(f"LLM token usage: {json.dumps(get_usage_summary(), sort_keys=True)}")


if __name__ == "__main__":
    main()
