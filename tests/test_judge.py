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



        