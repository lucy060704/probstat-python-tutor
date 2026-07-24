"""Tests for the pure v0.1 mastery update functions."""

import pytest

from probstat_tutor.curriculum import load_default_question_bank
from probstat_tutor.mastery import HINT_CONFIDENCE, apply_attempt, create_initial_state
from probstat_tutor.schemas import CapabilityDimension, ConceptId, GradeResult, Question


def _question(question_id: str) -> Question:
    bank = load_default_question_bank()
    return next(question for question in bank.questions if question.id == question_id)


def _grade(*, correct: bool) -> GradeResult:
    return GradeResult(score=1.0 if correct else 0.0, is_correct=correct)


def test_new_learner_starts_neutral_in_every_cell() -> None:
    state = create_initial_state()

    assert state.history == ()
    assert state.completed_question_ids == frozenset()
    for concept in ConceptId:
        for dimension in CapabilityDimension:
            assert getattr(state.mastery[concept], dimension.value) == 0.5


def test_attempt_uses_formula_and_does_not_mutate_input() -> None:
    state = create_initial_state()
    question = _question("mean_median_concept_01")

    updated = apply_attempt(state, question, _grade(correct=True), hint_level=0)

    assert state.mastery[ConceptId.MEAN_MEDIAN].concept == 0.5
    assert updated.mastery[ConceptId.MEAN_MEDIAN].concept == pytest.approx(0.605)
    assert updated.mastery[ConceptId.MEAN_MEDIAN].calculation == 0.5
    assert question.id in updated.completed_question_ids


def test_higher_hint_level_reduces_adjusted_evidence() -> None:
    question = _question("mean_median_concept_01")
    resulting_scores: list[float] = []

    for hint_level in HINT_CONFIDENCE:
        updated = apply_attempt(
            create_initial_state(), question, _grade(correct=True), hint_level=hint_level
        )
        resulting_scores.append(updated.mastery[ConceptId.MEAN_MEDIAN].concept)

    assert resulting_scores == sorted(resulting_scores, reverse=True)
    assert len(set(resulting_scores)) == 4


def test_two_correct_attempts_raise_mastery_and_record_streak() -> None:
    state = create_initial_state()
    question = _question("mean_median_python_01")

    once = apply_attempt(state, question, _grade(correct=True), hint_level=0)
    twice = apply_attempt(once, question, _grade(correct=True), hint_level=0)

    assert twice.mastery[ConceptId.MEAN_MEDIAN].python > once.mastery[
        ConceptId.MEAN_MEDIAN
    ].python
    assert [attempt.is_correct for attempt in twice.history[-2:]] == [True, True]


def test_two_wrong_attempts_lower_mastery_and_do_not_mark_complete() -> None:
    state = create_initial_state()
    question = _question("mean_median_interpretation_01")

    once = apply_attempt(state, question, _grade(correct=False), hint_level=0)
    twice = apply_attempt(once, question, _grade(correct=False), hint_level=0)

    assert twice.mastery[ConceptId.MEAN_MEDIAN].interpretation < once.mastery[
        ConceptId.MEAN_MEDIAN
    ].interpretation
    assert question.id not in twice.completed_question_ids
    assert [attempt.is_correct for attempt in twice.history[-2:]] == [False, False]


def test_invalid_hint_level_has_clear_error() -> None:
    with pytest.raises(ValueError, match="0、1、2 或 3"):
        apply_attempt(
            create_initial_state(),
            _question("mean_median_concept_01"),
            _grade(correct=True),
            hint_level=4,
        )
