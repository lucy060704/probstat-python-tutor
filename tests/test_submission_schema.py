"""Tests for the validated learner-authored submission and evidence contract."""

import pytest
from pydantic import ValidationError

from probstat_tutor.schemas import (
    MAX_ANSWER_LENGTH,
    MAX_PYTHON_CODE_LENGTH,
    MAX_REASONING_LENGTH,
    LearnerEvidence,
    LearnerSubmission,
    LearningSubmissionRequest,
    SubmissionField,
)


def test_submission_keeps_observable_text_and_defaults_optional_fields() -> None:
    submission = LearnerSubmission(answer="  8  ")

    assert submission.answer == "  8  "
    assert submission.reasoning == ""
    assert submission.python_code == ""


@pytest.mark.parametrize(
    ("payload", "expected_message"),
    [
        ({}, "答案不能为空"),
        ({"answer": "   "}, "答案不能为空"),
        ({"answer": 8}, "答案必须是文本"),
        ({"answer": "8", "reasoning": None}, "思考过程必须是文本"),
        ({"answer": "8", "python_code": ["print(8)"]}, "Python 代码必须是文本"),
        ({"answer": "8", "score": 1.0}, "学习者提交包含不支持的字段"),
        ({"answer": "a" * (MAX_ANSWER_LENGTH + 1)}, "答案不能超过"),
        (
            {"answer": "8", "reasoning": "a" * (MAX_REASONING_LENGTH + 1)},
            "思考过程不能超过",
        ),
        (
            {"answer": "8", "python_code": "a" * (MAX_PYTHON_CODE_LENGTH + 1)},
            "Python 代码不能超过",
        ),
    ],
)
def test_submission_rejects_invalid_input_with_chinese_message(
    payload: dict[str, object],
    expected_message: str,
) -> None:
    with pytest.raises(ValidationError, match=expected_message):
        LearnerSubmission.model_validate(payload)


def test_submission_is_immutable() -> None:
    submission = LearnerSubmission(answer="8")

    with pytest.raises(ValidationError):
        submission.answer = "9"


def test_structured_evidence_labels_the_exact_learner_source() -> None:
    evidence = LearnerEvidence(
        source=SubmissionField.PYTHON_CODE,
        quote='df["value"].median()',
    )

    assert evidence.model_dump() == {
        "source": SubmissionField.PYTHON_CODE,
        "quote": 'df["value"].median()',
    }


@pytest.mark.parametrize("hint_level", [-1, 5, True, "1", object()])
def test_submission_request_rejects_invalid_hint_level(hint_level: object) -> None:
    with pytest.raises(ValidationError, match="提示层级必须是 0、1、2、3 或 4"):
        LearningSubmissionRequest(
            learner_id="learner",
            session_id="session",
            question_id="question",
            submission=LearnerSubmission(answer="8"),
            hint_level=hint_level,
        )
