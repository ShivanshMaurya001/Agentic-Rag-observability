
import json
import sys
from pathlib import Path

from scripts.run_baseline_eval import (
    ACTIVE_METADATA,
    checkpoint_results,
    load_saved_results,
    make_run_id,
    merge_result_row,
    model_metadata,
    refresh_active_metrics,
)


def test_make_run_id_is_stable():
    first = make_run_id("baseline", "What is retrieval?")
    second = make_run_id("baseline", "What is retrieval?")
    openai_run = make_run_id(
        "baseline", "What is retrieval?", "openai-fingerprint"
    )
    gemini_run = make_run_id(
        "baseline", "What is retrieval?", "gemini-fingerprint"
    )

    assert first == second
    assert first.startswith("eval-baseline-")
    assert openai_run != gemini_run


def test_model_metadata_tracks_each_role():
    baseline = model_metadata("baseline")
    corrected = model_metadata("corrected")

    assert baseline["providers"] == {"generator": "openai"}
    assert set(baseline["models"]) == {"generator"}
    assert set(corrected["providers"]) == {
        "generator", "rewrite", "judge"
    }
    assert set(corrected["models"]) == {
        "generator", "rewrite", "judge"
    }
    assert baseline["model_fingerprint"] != corrected["model_fingerprint"]


def test_merge_result_row_preserves_historical_and_legacy_rows():
    historical = {"query": "q", "answer": "Gemini", "model_fingerprint": "old"}
    legacy = {"query": "q", "answer": "legacy"}
    rows = [historical, legacy]

    merge_result_row(rows, {
        "query": "q", "answer": "OpenAI", "model_fingerprint": "active"
    })

    assert rows == [
        historical,
        legacy,
        {"query": "q", "answer": "OpenAI", "model_fingerprint": "active"},
    ]


def test_checkpoint_preserves_existing_json_and_creates_unique_backup(tmp_path):
    results_path = tmp_path / "baseline_results.json"
    original = '{"results": {"baseline": [{"query": "legacy"}]}}'
    results_path.write_text(original, encoding="utf-8")
    existing_backup = tmp_path / "baseline_results.phase2.bak"
    existing_backup.write_text("keep this backup", encoding="utf-8")

    state, backup = checkpoint_results(
        results_path,
        {"results": {"baseline": [{"query": "legacy"}, {"query": "new"}]}},
    )

    assert state is True
    assert backup == tmp_path / "baseline_results.phase2.1.bak"
    assert backup.read_text(encoding="utf-8") == original
    assert existing_backup.read_text(encoding="utf-8") == "keep this backup"
    assert json.loads(results_path.read_text(encoding="utf-8"))["results"]["baseline"] == [
        {"query": "legacy"}, {"query": "new"}
    ]


def test_checkpoint_uses_atomic_replace_and_leaves_no_temp_file(
    tmp_path, monkeypatch
):
    import scripts.run_baseline_eval as evaluation

    results_path = tmp_path / "baseline_results.json"
    results_path.write_text("{}", encoding="utf-8")
    calls = []
    real_replace = evaluation.os.replace

    def record_replace(source, destination):
        source_path = Path(source)
        assert source_path.exists()
        assert Path(destination) == results_path
        assert (tmp_path / "baseline_results.phase2.bak").exists()
        calls.append(source_path)
        real_replace(source, destination)

    monkeypatch.setattr(evaluation.os, "replace", record_replace)
    checkpoint_results(results_path, {"ready": True})

    assert len(calls) == 1
    assert not calls[0].exists()
    assert not list(tmp_path.glob(".*.tmp"))
    assert json.loads(results_path.read_text(encoding="utf-8")) == {"ready": True}


def test_refresh_active_metrics_excludes_legacy_and_other_provider_rows():
    fingerprint = ACTIVE_METADATA["baseline"]["model_fingerprint"]
    active = {
        "model_fingerprint": fingerprint,
        "is_adversarial": False,
        "final_outcome": "answered",
        "answer": "right answer",
        "answer_correct": True,
        "answer_correctness_score": 1.0,
    }
    saved = {
        "results": {
            "baseline": [
                active,
                {**active, "model_fingerprint": "gemini"},
                {key: value for key, value in active.items()
                 if key != "model_fingerprint"},
            ],
            "corrected": [],
        },
        "metrics": {},
    }

    refresh_active_metrics(saved, ACTIVE_METADATA)

    assert saved["metrics"]["baseline"]["total_queries"] == 1
    assert saved["metrics"]["baseline"]["answer_correctness"] == 1.0
    assert saved["metrics"]["corrected"] == {}


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
        **evaluation.ACTIVE_METADATA["baseline"],
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
        metadata_by_mode=None,
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
            metadata_by_mode=metadata_by_mode,
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


def test_resume_does_not_reuse_other_provider_row(
    tmp_path, monkeypatch
):
    import scripts.run_baseline_eval as evaluation

    results_path = tmp_path / "baseline_results.json"
    query = {
        "query": "Provider isolation query",
        "is_adversarial": False,
        "expected_answer": "active answer",
        "relevant_chunks": [],
    }
    gemini_row = {
        "query": query["query"],
        "mode": "baseline",
        "answer": "Gemini cached answer",
        "final_outcome": "answered",
        "total_attempts": 1,
        "is_adversarial": False,
        "provider": "gemini",
        "model_fingerprint": "gemini-fingerprint",
    }
    results_path.write_text(
        json.dumps({"results": {"baseline": [gemini_row], "corrected": []}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(evaluation, "RESULTS_PATH", results_path)
    monkeypatch.setattr(evaluation, "init_db", lambda: None)
    monkeypatch.setattr(evaluation, "load_eval_queries", lambda: [query])
    baseline_calls = []
    corrected_calls = []
    monkeypatch.setattr(
        evaluation,
        "run_baseline",
        lambda item: (
            baseline_calls.append(item["query"])
            or {
                "answer": "Active OpenAI answer",
                "final_outcome": "answered",
                "total_attempts": 1,
                "retrieved_chunks": [],
                "retrieved_chunk_ids": [],
                "attempts": [],
            }
        ),
    )
    monkeypatch.setattr(
        evaluation,
        "run_corrected",
        lambda item: (
            corrected_calls.append(item["query"])
            or {
                "answer": "Active corrected answer",
                "final_outcome": "answered",
                "total_attempts": 1,
                "retrieved_chunks": [],
                "retrieved_chunk_ids": [],
                "attempts": [],
            }
        ),
    )
    monkeypatch.setattr(evaluation, "log_evaluation_row", lambda **kwargs: None)
    monkeypatch.setattr(sys, "argv", ["run_baseline_eval.py", "--resume"])

    evaluation.main()

    saved = json.loads(results_path.read_text(encoding="utf-8"))
    baseline_rows = saved["results"]["baseline"]
    assert baseline_calls == [query["query"]]
    assert corrected_calls == [query["query"]]
    assert len(baseline_rows) == 2
    assert baseline_rows[0]["answer"] == "Gemini cached answer"
    assert baseline_rows[1]["providers"] == {"generator": "openai"}


def test_main_selects_dataset_positions_and_isolates_results_path(
    tmp_path, monkeypatch
):
    import scripts.run_baseline_eval as evaluation

    main_results = tmp_path / "baseline_results.json"
    main_results.write_text("keep main results", encoding="utf-8")
    smoke_results = tmp_path / "smoke_results.json"
    monkeypatch.setattr(evaluation, "RESULTS_PATH", main_results)
    monkeypatch.setattr(evaluation, "init_db", lambda: None)
    monkeypatch.setattr(evaluation, "log_evaluation_row", lambda **kwargs: None)
    monkeypatch.setattr(
        evaluation,
        "run_baseline",
        lambda item: {
            "answer": "mock answer",
            "final_outcome": "answered",
            "total_attempts": 1,
            "retrieved_chunks": [],
            "retrieved_chunk_ids": [],
            "attempts": [],
            "is_hallucination": None,
        },
    )
    monkeypatch.setattr(
        evaluation,
        "run_corrected",
        lambda item: {
            "answer": "mock answer",
            "final_outcome": "answered",
            "total_attempts": 1,
            "retrieved_chunks": [],
            "retrieved_chunk_ids": [],
            "attempts": [],
            "is_hallucination": None,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_baseline_eval.py",
            "--query-indices", "0", "1", "7",
            "--results-path", str(smoke_results),
        ],
    )

    evaluation.main()

    saved = json.loads(smoke_results.read_text(encoding="utf-8"))
    for mode in ("baseline", "corrected"):
        rows = saved["results"][mode]
        assert [row["is_adversarial"] for row in rows] == [False, False, True]
        assert saved["metrics"][mode]["total_queries"] == 3
    assert main_results.read_text(encoding="utf-8") == "keep main results"
