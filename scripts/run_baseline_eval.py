
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from app.abstain import ABSTAIN_MESSAGE
from app.db import get_connection, init_db
from app.embedding_service import client
from app.graph import graph
from app.retriever import retrieve
from app.trace_logger import log_attempt, log_run
from eval.baseline_runner import run_comparison
from eval.eval_dataset import load_eval_queries


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_PATH = BASE_DIR / "eval" / "baseline_results.json"
MODEL_NAME = "gemini-3.6-flash"


def make_run_id(mode: str, query: str) -> str:
    """Create a repeatable run ID so resumed runs don't duplicate DB rows."""
    digest = hashlib.sha256(f"{mode}:{query}".encode("utf-8")).hexdigest()[:24]
    return f"eval-{mode}-{digest}"


def generate_answer(query: str, chunks: list[dict[str, Any]]) -> str:
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
        model=MODEL_NAME,
        contents=prompt,
    )
    if response is None or response.text is None:
        raise ValueError("Generator returned no answer.")
    return response.text.strip()


def run_baseline(item: dict[str, Any]) -> dict[str, Any]:
    """Single retrieve -> generate path without the corrective graph."""
    chunks = retrieve(item["query"])
    answer = generate_answer(item["query"], chunks)

    return {
        "answer": answer,
        "final_outcome": "answered",
        "total_attempts": 1,
        "retrieved_chunks": chunks,
        "retrieved_chunk_ids": [chunk["chunk_id"] for chunk in chunks],
        "is_hallucination": None,
    }


def run_corrected(item: dict[str, Any]) -> dict[str, Any]:
    """Run the existing corrective LangGraph using its required state."""
    initial_state = {
        "original_query": item["query"],
        "current_query": item["query"],
        "retrieved_chunks": [],
        "answer": "",
        "judge_verdict": None,
        "attempt_number": 0,
        "failure_type": [],
        "final_outcome": None,
    }
    final_state = graph.invoke(initial_state)
    chunks = final_state.get("retrieved_chunks", [])

    return {
        "answer": final_state.get("answer", ""),
        "final_outcome": final_state.get("final_outcome", "answered"),
        "total_attempts": final_state.get("attempt_number", 0) + 1,
        "retrieved_chunks": chunks,
        "retrieved_chunk_ids": [chunk["chunk_id"] for chunk in chunks],
        "is_hallucination": None,
    }


def load_saved_results() -> dict[str, Any]:
    if not RESULTS_PATH.exists():
        return {"results": {"baseline": [], "corrected": []}, "metrics": {}}

    with RESULTS_PATH.open("r", encoding="utf-8") as file:
        saved = json.load(file)

    saved.setdefault("results", {})
    saved["results"].setdefault("baseline", [])
    saved["results"].setdefault("corrected", [])
    saved.setdefault("metrics", {})
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Run M4 baseline vs corrected evaluation.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N queries.")
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

    active_mode = {"value": ""}

    def cached_or_run(mode: str, actual_runner):
        def runner(item: dict[str, Any]) -> dict[str, Any]:
            active_mode["value"] = mode
            previous = cached.get((mode, item["query"]))
            if previous is not None:
                return previous
            return actual_runner(item)
        return runner

    def persist_row(row: dict[str, Any]) -> None:
        mode = row["mode"]
        query = row["query"]

        # Cached results are already persisted; don't write them again.
        if (mode, query) in cached:
            return

        run_id = make_run_id(mode, query)
        chunks = row.get("retrieved_chunks", [])
        log_attempt(
            run_id=run_id,
            attempt_number=max(row.get("total_attempts", 1) - 1, 0),
            query_used=query,
            retrieved_chunks=chunks,
            generated_answer=row.get("answer", ""),
            judge_verdict={},
        )
        log_run(
            run_id=run_id,
            original_query=query,
            final_outcome=row["final_outcome"],
            total_attempts=row.get("total_attempts", 1),
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

    saved_rows: dict[str, list[dict[str, Any]]] = {}
    for mode in ("baseline", "corrected"):
        # Keep only queries in this invocation's selected dataset.
        selected_queries = {item["query"] for item in dataset}
        merged = {
            row["query"]: row
            for row in saved["results"].get(mode, [])
            if row["query"] in selected_queries
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
