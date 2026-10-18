
import json

from scripts.run_baseline_eval import load_saved_results, make_run_id


def test_make_run_id_is_stable():
    first = make_run_id("baseline", "What is retrieval?")
    second = make_run_id("baseline", "What is retrieval?")

    assert first == second
    assert first.startswith("eval-baseline-")


def test_load_saved_results_when_file_missing(tmp_path, monkeypatch):
    results_path = tmp_path / "baseline_results.json"
    monkeypatch.setattr("scripts.run_baseline_eval.RESULTS_PATH", results_path)

    result = load_saved_results()

    assert result == {
        "results": {"baseline": [], "corrected": []},
        "metrics": {},
    }


def test_load_saved_results_reads_json(tmp_path, monkeypatch):
    results_path = tmp_path / "baseline_results.json"
    expected = {
        "results": {"baseline": [{"query": "q1"}], "corrected": []},
        "metrics": {},
    }
    results_path.write_text(json.dumps(expected), encoding="utf-8")
    monkeypatch.setattr("scripts.run_baseline_eval.RESULTS_PATH", results_path)

    assert load_saved_results() == expected
