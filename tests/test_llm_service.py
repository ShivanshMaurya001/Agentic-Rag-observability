from types import SimpleNamespace

import pytest
from pydantic import BaseModel, Field

from app import llm_service


class SampleVerdict(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    outcome: str


def test_generate_text_uses_responses_api(monkeypatch):
    calls = []
    fake_client = SimpleNamespace(
        responses=SimpleNamespace(
            create=lambda **kwargs: calls.append(kwargs)
            or SimpleNamespace(status="completed", output_text="answer")
        )
    )
    monkeypatch.setattr(llm_service, "_get_client", lambda: fake_client)

    result = llm_service.generate_text("prompt", model="gpt-6-luna")

    assert result == "answer"
    assert calls == [{"model": "gpt-6-luna", "input": "prompt"}]


def test_generate_structured_uses_pydantic_schema(monkeypatch):
    calls = []
    verdict = SampleVerdict(score=0.8, outcome="pass")
    fake_client = SimpleNamespace(
        responses=SimpleNamespace(
            parse=lambda **kwargs: calls.append(kwargs)
            or SimpleNamespace(status="completed", output_parsed=verdict)
        )
    )
    monkeypatch.setattr(llm_service, "_get_client", lambda: fake_client)

    result = llm_service.generate_structured(
        "judge prompt", model="gpt-6-luna", schema=SampleVerdict
    )

    assert result == verdict
    assert calls == [{
        "model": "gpt-6-luna",
        "input": "judge prompt",
        "text_format": SampleVerdict,
    }]


def test_generate_text_rejects_incomplete_response(monkeypatch):
    fake_client = SimpleNamespace(
        responses=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(
                status="incomplete", output_text="partial"
            )
        )
    )
    monkeypatch.setattr(llm_service, "_get_client", lambda: fake_client)

    with pytest.raises(llm_service.LLMIncompleteResponseError):
        llm_service.generate_text("prompt", model="gpt-6-luna")


def test_generate_structured_detects_refusal(monkeypatch):
    refusal = SimpleNamespace(type="refusal", refusal="not allowed")
    fake_client = SimpleNamespace(
        responses=SimpleNamespace(
            parse=lambda **kwargs: SimpleNamespace(
                status="completed",
                output_parsed=None,
                output=[SimpleNamespace(content=[refusal])],
            )
        )
    )
    monkeypatch.setattr(llm_service, "_get_client", lambda: fake_client)

    with pytest.raises(llm_service.LLMRefusalError):
        llm_service.generate_structured(
            "prompt", model="gpt-6-luna", schema=SampleVerdict
        )


def test_generate_structured_rejects_schema_mismatch(monkeypatch):
    fake_client = SimpleNamespace(
        responses=SimpleNamespace(
            parse=lambda **kwargs: SimpleNamespace(
                status="completed",
                output_parsed={"score": 2.0, "outcome": "pass"},
            )
        )
    )
    monkeypatch.setattr(llm_service, "_get_client", lambda: fake_client)

    with pytest.raises(llm_service.LLMServiceError) as error:
        llm_service.generate_structured(
            "prompt", model="gpt-6-luna", schema=SampleVerdict
        )

    assert "required schema" in str(error.value)


def test_generate_text_detects_refusal(monkeypatch):
    refusal = SimpleNamespace(type="refusal", refusal="not allowed")
    fake_client = SimpleNamespace(
        responses=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(
                status="completed",
                output_text="",
                output=[SimpleNamespace(content=[refusal])],
            )
        )
    )
    monkeypatch.setattr(llm_service, "_get_client", lambda: fake_client)

    with pytest.raises(llm_service.LLMRefusalError):
        llm_service.generate_text("prompt", model="gpt-6-luna")


def test_request_errors_are_sanitized(monkeypatch):
    class RateLimitError(RuntimeError):
        pass

    def fail_request(**kwargs):
        raise RateLimitError("credential-do-not-expose")

    fake_client = SimpleNamespace(
        responses=SimpleNamespace(create=fail_request)
    )
    monkeypatch.setattr(llm_service, "_get_client", lambda: fake_client)

    with pytest.raises(llm_service.LLMServiceError) as error:
        llm_service.generate_text("prompt", model="gpt-6-luna")

    assert "RateLimitError" in str(error.value)
    assert "credential-do-not-expose" not in str(error.value)


def test_client_configures_bounded_transport_retries(monkeypatch):
    created = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            created.update(kwargs)

    monkeypatch.setattr(llm_service, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(llm_service, "_client", None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    llm_service._get_client()

    assert created["max_retries"] == 2
    assert created["api_key"] == "test-key"


def test_usage_summary_accumulates_resets_and_marks_unavailable(monkeypatch):
    llm_service.reset_usage_summary()
    responses = iter([
        SimpleNamespace(
            status="completed",
            output_text="one",
            model="gpt-6-luna",
            usage=SimpleNamespace(
                input_tokens=10,
                output_tokens=5,
                output_tokens_details=SimpleNamespace(reasoning_tokens=0),
            ),
        ),
        SimpleNamespace(
            status="completed", output_text="two", model="gpt-6-luna",
            usage=SimpleNamespace(input_tokens=3, output_tokens=None),
        ),
        SimpleNamespace(status="completed", output_text="three"),
    ])
    fake_client = SimpleNamespace(
        responses=SimpleNamespace(create=lambda **kwargs: next(responses))
    )
    monkeypatch.setattr(llm_service, "_get_client", lambda: fake_client)

    for model in ("gpt-6-luna", "gpt-6-luna", "other-model"):
        llm_service.generate_text("prompt", model=model)

    assert llm_service.get_usage_summary() == {
        "gpt-6-luna": {
            "calls": 2,
            "usage_unavailable_calls": 1,
            "input_tokens": 13,
            "output_tokens": 5,
            "reasoning_tokens": 0,
            "reasoning_usage_unavailable_calls": 1,
        },
        "other-model": {
            "calls": 1,
            "usage_unavailable_calls": 1,
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "reasoning_usage_unavailable_calls": 1,
        },
    }

    llm_service.reset_usage_summary()
    assert llm_service.get_usage_summary() == {}
