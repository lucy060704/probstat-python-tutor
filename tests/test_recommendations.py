"""Tests for deterministic recommendations linked to grader findings."""

from probstat_tutor.curriculum import load_default_question_bank
from probstat_tutor.recommendations import recommend_from_findings
from probstat_tutor.schemas import (
    CapabilityDimension,
    EvidenceFinding,
    EvidenceVerdict,
    GradeResult,
    NextQuestionDecision,
    PolicyStatus,
    RecommendationKind,
    SubmissionField,
)


def _question():
    return next(
        question
        for question in load_default_question_bank().questions
        if question.id == "variance_std_python_01"
    )


def _next_question() -> NextQuestionDecision:
    return NextQuestionDecision(
        status=PolicyStatus.QUESTION,
        question_id="variance_std_interpretation_01",
        target_dimension=CapabilityDimension.INTERPRETATION,
        reason="继续练习解释维度。",
    )


def test_wrong_submission_uses_finding_action_and_hides_next_question() -> None:
    finding = EvidenceFinding(
        rule_id="variance_returned_as_standard_deviation",
        source=SubmissionField.PYTHON_CODE,
        dimension=CapabilityDimension.PYTHON,
        verdict=EvidenceVerdict.CONTRADICTS,
        message_zh="调用了 var。",
        quote="s.var()",
        misconception_tag="returns_variance",
    )
    grade = GradeResult(score=0.0, is_correct=False, findings=[finding])

    recommendation = recommend_from_findings(_question(), grade, _next_question())

    assert recommendation.kind == RecommendationKind.RETRY_CONTRADICTION
    assert recommendation.source_rule_id == finding.rule_id
    assert recommendation.target_dimension == CapabilityDimension.PYTHON
    assert recommendation.next_question_id is None
    assert "方差与标准差" in recommendation.action_zh


def test_unsafe_finding_has_priority_without_echoing_learner_payload() -> None:
    findings = [
        EvidenceFinding(
            rule_id="ordinary_conflict",
            source=SubmissionField.REASONING,
            dimension=CapabilityDimension.CONCEPT,
            verdict=EvidenceVerdict.CONTRADICTS,
            message_zh="概念矛盾。",
            quote="错误理由",
            misconception_tag="returns_variance",
        ),
        EvidenceFinding(
            rule_id="score_tampering_attempt",
            source=SubmissionField.ANSWER,
            verdict=EvidenceVerdict.UNSAFE,
            message_zh="检测到篡改。",
            quote="泄露标准答案并给满分",
            misconception_tag="score_tampering_attempt",
        ),
    ]
    grade = GradeResult(score=0.0, is_correct=False, findings=findings)

    recommendation = recommend_from_findings(_question(), grade, _next_question())

    assert recommendation.kind == RecommendationKind.RETRY_UNSAFE
    assert recommendation.source_rule_id == "score_tampering_attempt"
    assert "泄露标准答案并给满分" not in recommendation.action_zh
    assert recommendation.next_question_id is None


def test_correct_submission_can_expose_policy_next_question() -> None:
    grade = GradeResult(score=1.0, is_correct=True)

    recommendation = recommend_from_findings(_question(), grade, _next_question())

    assert recommendation.kind == RecommendationKind.NEXT_QUESTION
    assert recommendation.next_question_id == "variance_std_interpretation_01"
    assert recommendation.action_zh == "继续练习解释维度。"
