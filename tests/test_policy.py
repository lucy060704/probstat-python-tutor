"""Tests for the deterministic next-question heuristic."""

from probstat_tutor.curriculum import load_default_question_bank
from probstat_tutor.mastery import apply_attempt, create_initial_state
from probstat_tutor.policy import select_next_question
from probstat_tutor.schemas import (
    AttemptRecord,
    CapabilityDimension,
    ConceptId,
    GradeResult,
    LearningState,
    MasteryScores,
    PolicyStatus,
    Question,
)


def _questions() -> list[Question]:
    return load_default_question_bank().questions


def _history(*, correct: bool) -> tuple[AttemptRecord, AttemptRecord]:
    return tuple(
        AttemptRecord(
            question_id=f"old_{index}",
            concept_id=ConceptId.MEAN_MEDIAN,
            difficulty=0.5,
            score=1.0 if correct else 0.0,
            adjusted_evidence=1.0 if correct else 0.0,
            is_correct=correct,
            hint_level=0,
        )
        for index in range(2)
    )


def test_new_learner_receives_foundational_question() -> None:
    decision = select_next_question(create_initial_state(), _questions())

    assert decision.status == PolicyStatus.QUESTION
    assert decision.question_id == "mean_median_concept_01"
    assert decision.target_dimension == CapabilityDimension.CONCEPT
    assert decision.recommended_hint_level == 0


def test_weakest_dimension_is_prioritized() -> None:
    state = create_initial_state()
    mastery = dict(state.mastery)
    mastery[ConceptId.MEAN_MEDIAN] = MasteryScores(
        concept=0.8, calculation=0.8, python=0.1, interpretation=0.8
    )
    state = state.model_copy(update={"mastery": mastery})

    decision = select_next_question(state, _questions())

    assert decision.target_dimension == CapabilityDimension.PYTHON
    assert decision.question_id == "data_quality_python_01"


def test_two_failures_choose_lower_difficulty_and_level_one_hint() -> None:
    base = _questions()[0]
    low = base.model_copy(update={"id": "low", "difficulty": 0.35})
    high = base.model_copy(update={"id": "high", "difficulty": 0.65})
    state = LearningState(history=_history(correct=False))

    decision = select_next_question(state, [low, high])

    assert decision.question_id == "low"
    assert decision.recommended_hint_level == 1
    assert "连续两次" in decision.reason


def test_two_successes_choose_higher_difficulty() -> None:
    base = _questions()[0]
    low = base.model_copy(update={"id": "low", "difficulty": 0.35})
    high = base.model_copy(update={"id": "high", "difficulty": 0.65})
    state = LearningState(history=_history(correct=True))

    decision = select_next_question(state, [low, high])

    assert decision.question_id == "high"
    assert decision.recommended_hint_level == 0
    assert "提高" in decision.reason


def test_insufficient_prerequisite_returns_blocked_status() -> None:
    questions = [
        question
        for question in _questions()
        if question.concept_id in {ConceptId.MEAN_MEDIAN, ConceptId.VARIANCE_STD}
    ]
    foundational_ids = {
        question.id for question in questions if question.concept_id == ConceptId.MEAN_MEDIAN
    }
    state = LearningState(completed_question_ids=frozenset(foundational_ids))

    decision = select_next_question(state, questions)

    assert decision.status == PolicyStatus.BLOCKED
    assert decision.question_id is None
    assert "前置知识" in decision.reason


def test_prerequisite_at_exact_threshold_is_allowed() -> None:
    questions = [
        question
        for question in _questions()
        if question.concept_id in {ConceptId.MEAN_MEDIAN, ConceptId.VARIANCE_STD}
    ]
    mastery = create_initial_state().mastery.copy()
    mastery[ConceptId.MEAN_MEDIAN] = MasteryScores(
        concept=0.6, calculation=0.6, python=0.6, interpretation=0.6
    )
    foundational_ids = {
        question.id for question in questions if question.concept_id == ConceptId.MEAN_MEDIAN
    }
    state = LearningState(
        mastery=mastery,
        completed_question_ids=frozenset(foundational_ids),
    )

    decision = select_next_question(state, questions)

    assert decision.status == PolicyStatus.QUESTION
    assert decision.question_id == "variance_std_concept_01"


def test_recent_three_questions_are_not_repeated() -> None:
    questions = [
        question
        for question in _questions()
        if question.concept_id == ConceptId.MEAN_MEDIAN
    ]
    state = create_initial_state()
    grade = GradeResult(score=0.0, is_correct=False)
    for question in questions[:3]:
        state = apply_attempt(state, question, grade, hint_level=0)

    decision = select_next_question(state, questions)

    assert decision.status == PolicyStatus.BLOCKED
    assert decision.question_id is None


def test_all_questions_completed_returns_complete_status() -> None:
    questions = _questions()
    state = LearningState(
        completed_question_ids=frozenset(question.id for question in questions)
    )

    decision = select_next_question(state, questions)

    assert decision.status == PolicyStatus.COMPLETE
    assert decision.question_id is None
    assert "全部完成" in decision.reason
