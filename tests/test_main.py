
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app


def test_query_endpoint_logs_run_and_attempt():
    fake_verdict = {
        "grounded": True,
        "retrieval_quality": 0.9,
        "answer_quality": 0.95,
        "failure_type": "none",
        "reason": "Answer is supported by the retrieved context.",
    }

    fake_chunks = [
        {
            "chunk_id": "chunk-1",
            "text": "RAG combines retrieval and generation.",
        }
    ]

    def fake_stream(*args, **kwargs):
        yield {
            "retrieve": {
                "retrieved_chunks": fake_chunks,
            }
        }
        yield {
            "generate": {
                "answer": "RAG combines retrieval and generation.",
            }
        }

        from app.judge import JudgeVerdict

        yield {
            "judge": {
                "judge_verdict": JudgeVerdict(**fake_verdict),
                "failure_type": ["none"],
            }
        }
        yield {
            "answered": {
                "final_outcome": "answered",
            }
        }

    with (
        patch("app.main.init_db"),
        patch("app.main.configure_langsmith"),
        patch("app.main.graph.stream", side_effect=fake_stream),
        TestClient(app) as client,
    ):
        response = client.post(
            "/query",
            json={"query": "What is RAG?"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["query"] == "What is RAG?"
        assert body["answer"] == "RAG combines retrieval and generation."
        assert body["final_outcome"] == "answered"

        trace_response = client.get(f"/traces/{body['run_id']}")

        assert trace_response.status_code == 200
        trace = trace_response.json()

        assert trace["run"]["original_query"] == "What is RAG?"
        assert trace["run"]["final_outcome"] == "answered"
        assert len(trace["attempts"]) == 1
        assert trace["attempts"][0]["judge_failure_type"] == "none"


def test_trace_endpoint_returns_404_for_unknown_run():
    with (
        patch("app.main.init_db"),
        patch("app.main.configure_langsmith"),
        TestClient(app) as client,
    ):
        response = client.get("/traces/nonexistent-run")

    assert response.status_code == 404
