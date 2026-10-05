import pytest
from pydantic import ValidationError

from app.judge import JudgeVerdict


def test_valid_judge_verdict():
    verdict = JudgeVerdict(
        grounded=True,
        retrieval_quality=0.9,
        answer_quality=0.8,
        failure_type="none",
        reason="The answer is supported by the retrieved context.",
    )

    assert verdict.grounded is True
    assert verdict.retrieval_quality == 0.9
    assert verdict.answer_quality == 0.8
    assert verdict.failure_type == "none"


def test_invalid_failure_type_fails():
    with pytest.raises(ValidationError):
        JudgeVerdict(
            grounded=True,
            retrieval_quality=0.9,
            answer_quality=0.8,
            failure_type="unknown",
            reason="Invalid failure type.",
        )


def test_score_above_one_fails():
    with pytest.raises(ValidationError):
        JudgeVerdict(
            grounded=True,
            retrieval_quality=1.5,
            answer_quality=0.8,
            failure_type="none",
            reason="Invalid score.",
        )


def test_missing_field_fails():
    with pytest.raises(ValidationError):
        JudgeVerdict(
            grounded=True,
            retrieval_quality=0.9,
            answer_quality=0.8,
            failure_type="none",
        )





def test_judge_answer_returns_structured_verdict(monkeypatch):
    from app import judge

    class FakeResponse:
        parsed = {
            "grounded": True,
            "retrieval_quality": 1.0,
            "answer_quality": 0.9,
            "failure_type": "none",
            "reason": "The answer is supported by the context.",
        }

    def fake_generate_content(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr(
        judge.client.models,
        "generate_content",
        fake_generate_content,
    )

    verdict = judge.judge_answer(
        query="What does self-attention do?",
        context="Self-attention determines how relevant each token is to other tokens.",
        answer="Self-attention determines token relevance.",
    )

    assert isinstance(verdict, judge.JudgeVerdict)
    assert verdict.grounded is True
    assert verdict.retrieval_quality == 1.0
    assert verdict.answer_quality == 0.9
    assert verdict.failure_type == "none"


def test_judge_answer_raises_on_missing_structured_response(monkeypatch):
    from app import judge

    class FakeResponse:
        parsed = None

    def fake_generate_content(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr(
        judge.client.models,
        "generate_content",
        fake_generate_content,
    )

    with pytest.raises(
        ValueError,
        match="Gemini judge returned no structured verdict.",
    ):
        judge.judge_answer(
            query="What does self-attention do?",
            context="Self-attention determines token relevance.",
            answer="Self-attention determines token relevance.",
        )