"""Pure functions for the v0.1 heuristic mastery model."""

from probstat_tutor.schemas import (
    AttemptRecord,
    CapabilityDimension,
    ConceptId,
    GradeResult,
    LearningState,
    MasteryScores,
    Question,
)

HINT_CONFIDENCE: dict[int, float] = {
    0: 1.00,
    1: 0.90,
    2: 0.75,
    3: 0.60,
}


def create_initial_state() -> LearningState:
    """Create a neutral state with 0.5 in every concept-dimension cell."""

    return LearningState()


def apply_attempt(
    state: LearningState,
    question: Question,
    grade: GradeResult,
    *,
    hint_level: int,
) -> LearningState:
    """Return a new state after one graded attempt; never mutate the input state."""

    if hint_level not in HINT_CONFIDENCE:
        raise ValueError("hint_level 必须是 0、1、2 或 3。")

    adjusted_evidence = _clip(grade.score * HINT_CONFIDENCE[hint_level])
    old_scores = state.mastery[question.concept_id]
    new_values: dict[str, float] = {}

    for dimension in CapabilityDimension:
        old = getattr(old_scores, dimension.value)
        weight = getattr(question.dimension_weights, dimension.value)
        weighted_evidence = old + weight * (adjusted_evidence - old)
        new_values[dimension.value] = _clip(0.7 * old + 0.3 * weighted_evidence)

    updated_mastery = dict(state.mastery)
    updated_mastery[question.concept_id] = MasteryScores(**new_values)
    attempt = AttemptRecord(
        question_id=question.id,
        concept_id=question.concept_id,
        difficulty=question.difficulty,
        score=grade.score,
        adjusted_evidence=adjusted_evidence,
        is_correct=grade.is_correct,
        hint_level=hint_level,
    )
    completed = state.completed_question_ids
    if grade.is_correct:
        completed = completed | {question.id}

    return LearningState(
        mastery=updated_mastery,
        history=(*state.history, attempt),
        completed_question_ids=completed,
    )


def concept_mastery(state: LearningState, concept: ConceptId) -> float:
    """Return the four-dimensional mean used for prerequisite checks."""

    scores = state.mastery[concept]
    values = [getattr(scores, dimension.value) for dimension in CapabilityDimension]
    return sum(values) / len(values)


def weakest_dimension(state: LearningState) -> CapabilityDimension:
    """Return the globally weakest dimension, with enum order as a stable tie-breaker."""

    return min(
        CapabilityDimension,
        key=lambda dimension: sum(
            getattr(state.mastery[concept], dimension.value) for concept in ConceptId
        )
        / len(ConceptId),
    )


def _clip(value: float) -> float:
    return min(1.0, max(0.0, value))
