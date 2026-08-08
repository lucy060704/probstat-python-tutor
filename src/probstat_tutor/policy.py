"""Pure, deterministic next-question policy for v0.1."""

from collections.abc import Sequence

from probstat_tutor.mastery import concept_mastery, weakest_dimension
from probstat_tutor.schemas import (
    CAPABILITY_LABELS_ZH,
    LearningState,
    NextQuestionDecision,
    PolicyStatus,
    Question,
)

PREREQUISITE_THRESHOLD = 0.60
DIFFICULTY_STEP = 0.15


def select_next_question(
    state: LearningState,
    questions: Sequence[Question],
) -> NextQuestionDecision:
    """Choose one eligible question using stable heuristic ranking."""

    if not questions:
        return NextQuestionDecision(
            status=PolicyStatus.COMPLETE,
            reason="题库为空，没有需要完成的题目。",
        )

    all_question_ids = {question.id for question in questions}
    if all_question_ids <= state.completed_question_ids:
        return NextQuestionDecision(
            status=PolicyStatus.COMPLETE,
            reason="当前题库中的题目已经全部完成。",
        )

    recent_question_ids = {attempt.question_id for attempt in state.history[-3:]}
    candidates = [
        question
        for question in questions
        if question.id not in state.completed_question_ids
        and question.id not in recent_question_ids
        and _prerequisites_satisfied(state, question)
    ]
    if not candidates:
        return NextQuestionDecision(
            status=PolicyStatus.BLOCKED,
            reason="暂时没有合适的下一题：请先提升前置知识，或完成三道其他题后再重试。",
        )

    target_dimension = weakest_dimension(state)
    failure_streak = _has_streak(state, expected_correct=False)
    success_streak = _has_streak(state, expected_correct=True)
    hint_level = 1 if failure_streak else 0

    def ranking_key(question: Question) -> tuple[float, float, str]:
        dimension_weight = getattr(question.dimension_weights, target_dimension.value)
        target_difficulty = getattr(
            state.mastery[question.concept_id], target_dimension.value
        )
        if failure_streak:
            target_difficulty = max(0.0, target_difficulty - DIFFICULTY_STEP)
        elif success_streak:
            target_difficulty = min(1.0, target_difficulty + DIFFICULTY_STEP)
        return (-dimension_weight, abs(question.difficulty - target_difficulty), question.id)

    selected = min(candidates, key=ranking_key)
    if failure_streak:
        reason = "最近连续两次未答对，因此降低目标难度，并建议从一级提示开始。"
    elif success_streak:
        reason = "最近连续两次答对，因此适当提高目标难度。"
    else:
        reason = (
            f"优先练习当前较弱的“{CAPABILITY_LABELS_ZH[target_dimension]}”维度，"
            "并选择难度接近当前掌握度的题目。"
        )

    return NextQuestionDecision(
        status=PolicyStatus.QUESTION,
        question_id=selected.id,
        target_dimension=target_dimension,
        recommended_hint_level=hint_level,
        reason=reason,
    )


def _prerequisites_satisfied(state: LearningState, question: Question) -> bool:
    return all(
        concept_mastery(state, prerequisite) >= PREREQUISITE_THRESHOLD
        for prerequisite in question.prerequisites
    )


def _has_streak(state: LearningState, *, expected_correct: bool) -> bool:
    return len(state.history) >= 2 and all(
        attempt.is_correct is expected_correct for attempt in state.history[-2:]
    )
