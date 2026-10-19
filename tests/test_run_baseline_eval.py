
import json
import sys

from scripts.run_baseline_eval import load_saved_results, make_run_id


def test_make_run_id_is_stable():
    first = make_run_id("baseline", "What is retrieval?")
    second = make_run_id("baseline", "What is retrieval?")

    assert first == second
    assert first.startswith("eval-baseline-")


def test_load_saved_results_when_file_missing(tmp_path, monkeypatch):
    import scripts.run_baseline_eval as evaluation

    results_path = tmp_path / "baseline_results.json"
    monkeypatch.setattr(evaluation, "RESULTS_PATH", results_path)

    result = load_saved_results()

    assert result == {
        "results": {"baseline": [], "corrected": []},
        "metrics": {},
    }


def test_load_saved_results_reads_json(tmp_path, monkeypatch):
    import scripts.run_baseline_eval as evaluation

    results_path = tmp_path / "baseline_results.json"
    expected = {
        "results": {"baseline": [{"query": "q1"}], "corrected": []},
        "metrics": {},
    }
    results_path.write_text(json.dumps(expected), encoding="utf-8")
    monkeypatch.setattr(evaluation, "RESULTS_PATH", results_path)

    assert load_saved_results() == expected


def test_load_saved_results_fills_missing_sections(tmp_path, monkeypatch):
    import scripts.run_baseline_eval as evaluation

    results_path = tmp_path / "baseline_results.json"
    results_path.write_text(
        json.dumps({"results": {"baseline": [{"query": "q1"}]}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(evaluation, "RESULTS_PATH", results_path)

    result = load_saved_results()

    assert result["results"]["baseline"] == [{"query": "q1"}]
    assert result["results"]["corrected"] == []
    assert result["metrics"] == {}


def test_saved_results_can_be_reused_by_query_and_mode(
    tmp_path, monkeypatch
):
    import scripts.run_baseline_eval as evaluation

    results_path = tmp_path / "baseline_results.json"
    saved_row = {
        "query": "What is retrieval?",
        "mode": "baseline",
        "answer": "Saved answer",
        "final_outcome": "answered",
        "total_attempts": 1,
        "is_adversarial": False,
    }
    results_path.write_text(
        json.dumps(
            {
                "results": {
                    "baseline": [saved_row],
                    "corrected": [],
                },
                "metrics": {},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(evaluation, "RESULTS_PATH", results_path)

    saved = load_saved_results()
    cached = {
        (mode, row["query"]): row
        for mode in ("baseline", "corrected")
        for row in saved["results"].get(mode, [])
    }

    runner_calls = []

    def cached_or_run(mode, actual_runner):
        def runner(item):
            previous = cached.get((mode, item["query"]))
            if previous is not None:
                return previous

            runner_calls.append((mode, item["query"]))
            return actual_runner(item)

        return runner

    runner = cached_or_run(
        "baseline",
        lambda item: {"answer": "New answer"},
    )
    result = runner({"query": "What is retrieval?"})

    assert result == saved_row
    assert runner_calls == []


def test_main_resume_reuses_saved_pair_and_runs_missing_pairs(
    tmp_path, monkeypatch
):
    import scripts.run_baseline_eval as evaluation

    results_path = tmp_path / "baseline_results.json"

    saved_query = {
        "query": "Saved query",
        "is_adversarial": False,
        "expected_answer": "saved answer",
        "relevant_chunks": [],
    }
    missing_query = {
        "query": "Missing query",
        "is_adversarial": True,
        "expected_answer": "",
        "relevant_chunks": [],
    }

    saved_baseline_row = {
        "query": "Saved query",
        "mode": "baseline",
        "answer": "saved answer",
        "final_outcome": "answered",
        "total_attempts": 1,
        "is_adversarial": False,
        "retrieved_chunk_ids": [],
        "retrieved_chunks": [],
        "attempts": [],
        "answer_correct": True,
        "answer_correctness_score": 1.0,
        "relevant_chunks": [],
    }

    results_path.write_text(
        json.dumps(
            {
                "results": {
                    "baseline": [saved_baseline_row],
                    "corrected": [],
                },
                "metrics": {},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(evaluation, "RESULTS_PATH", results_path)
    monkeypatch.setattr(evaluation, "init_db", lambda: None)
    monkeypatch.setattr(
        evaluation,
        "load_eval_queries",
        lambda: [saved_query, missing_query],
    )

    # Prevent real model/graph execution. Calls are counted by query.
    baseline_calls = []
    corrected_calls = []

    def fake_baseline(item):
        baseline_calls.append(item["query"])
        return {
            "answer": "generated baseline answer",
            "final_outcome": "answered",
            "total_attempts": 1,
            "retrieved_chunks": [],
            "retrieved_chunk_ids": [],
            "attempts": [],
            "is_hallucination": None,
        }

    def fake_corrected(item):
        corrected_calls.append(item["query"])
        return {
            "answer": "generated corrected answer",
            "final_outcome": "answered",
            "total_attempts": 1,
            "retrieved_chunks": [],
            "retrieved_chunk_ids": [],
            "attempts": [],
            "is_hallucination": None,
        }

    monkeypatch.setattr(evaluation, "run_baseline", fake_baseline)
    monkeypatch.setattr(evaluation, "run_corrected", fake_corrected)

    # Prevent writes to the real SQLite database.
    logged_rows = []
    monkeypatch.setattr(
        evaluation,
        "log_evaluation_row",
        lambda **kwargs: None,
    )

    # Capture rows passed to the persistence callback.
    original_run_comparison = evaluation.run_comparison

    def fake_run_comparison(
        run_baseline,
        run_corrected,
        log_run=None,
        dataset=None,
        limit=None,
    ):
        def capture(row):
            logged_rows.append(row)
            if log_run is not None:
                log_run(row)

        return original_run_comparison(
            run_baseline=run_baseline,
            run_corrected=run_corrected,
            log_run=capture,
            dataset=dataset,
            limit=limit,
        )

    monkeypatch.setattr(
        evaluation,
        "run_comparison",
        fake_run_comparison,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_baseline_eval.py", "--resume"],
    )
    evaluation.main()

    # Saved baseline is reused; missing baseline runs once.
    assert baseline_calls == ["Missing query"]

    # Corrected results are missing for both queries.
    assert corrected_calls == ["Saved query", "Missing query"]

    saved_output = json.loads(
        results_path.read_text(encoding="utf-8")
    )
    baseline_queries = {
        row["query"] for row in saved_output["results"]["baseline"]
    }
    corrected_queries = {
        row["query"] for row in saved_output["results"]["corrected"]
    }

    assert baseline_queries == {"Saved query", "Missing query"}
    assert corrected_queries == {"Saved query", "Missing query"}

    # Only the missing result pairs should be persisted.
    assert len(logged_rows) == 4
