"""Privacy and small-cell suppression tests for the teacher aggregate."""

from pathlib import Path

from probstat_tutor.analytics import (
    TeacherDashboardStatus,
    build_teacher_dashboard,
)
from probstat_tutor.schemas import AttemptRecord, ConceptId, LearningState
from probstat_tutor.storage import LearningStateStore


def _attempt(*, correct: bool, hint_level: int = 0) -> AttemptRecord:
    return AttemptRecord(
        question_id="data_quality_concept_01",
        concept_id=ConceptId.DATA_QUALITY,
        difficulty=0.25,
        score=1.0 if correct else 0.0,
        adjusted_evidence=1.0 if correct else 0.0,
        is_correct=correct,
        hint_level=hint_level,
    )


def test_no_data_and_small_cohort_outcomes_are_suppressed() -> None:
    empty = build_teacher_dashboard(())
    small = build_teacher_dashboard(
        (
            LearningState(history=(_attempt(correct=True),)),
            LearningState(history=(_attempt(correct=False, hint_level=1),)),
        )
    )

    assert empty.status == TeacherDashboardStatus.NO_DATA
    assert empty.profile_count == 0
    assert empty.attempted_profile_count == 0
    assert empty.total_attempt_count == 0
    assert small.status == TeacherDashboardStatus.SUPPRESSED
    assert small.profile_count is None
    assert small.attempted_profile_count is None
    assert small.total_attempt_count is None
    assert small.overall_correct_rate is None
    concept = next(
        item for item in small.concept_summaries if item.concept_id == ConceptId.DATA_QUALITY
    )
    assert concept.attempted_profile_count is None
    assert concept.attempt_count is None
    assert concept.correct_rate is None
    assert concept.average_mastery is None


def test_three_profiles_unlock_only_their_aggregate_cell() -> None:
    dashboard = build_teacher_dashboard(
        tuple(
            LearningState(history=(_attempt(correct=correct, hint_level=index),))
            for index, correct in enumerate((True, False, True))
        )
    )

    assert dashboard.status == TeacherDashboardStatus.READY
    assert dashboard.profile_count == 3
    assert dashboard.attempted_profile_count == 3
    assert dashboard.total_attempt_count == 3
    assert dashboard.overall_correct_rate == 2 / 3
    data_quality = next(
        item
        for item in dashboard.concept_summaries
        if item.concept_id == ConceptId.DATA_QUALITY
    )
    assert data_quality.correct_rate == 2 / 3
    assert data_quality.average_hint_level == 1.0
    assert all(
        item.suppressed
        for item in dashboard.concept_summaries
        if item.concept_id != ConceptId.DATA_QUALITY
    )


def test_storage_teacher_surface_returns_states_without_ids_or_receipts(
    tmp_path: Path,
) -> None:
    store = LearningStateStore(tmp_path / "learning.sqlite3")
    store.save("real-looking-id@example.test", LearningState(history=(_attempt(correct=True),)))

    serialized = repr(store.load_anonymized_states())

    assert "real-looking-id" not in serialized
    assert "learner_id" not in serialized
    assert "report_json" not in serialized


def test_teacher_dto_schema_forbids_private_fields() -> None:
    fields = set(build_teacher_dashboard(()).model_dump())

    assert fields.isdisjoint(
        {"learner_id", "session_id", "answer", "reasoning", "python_code", "receipt"}
    )


def test_serialized_small_cohort_contains_no_exact_counts() -> None:
    dashboard = build_teacher_dashboard(
        (
            LearningState(history=(_attempt(correct=True),)),
            LearningState(history=(_attempt(correct=False),)),
        )
    )
    payload = dashboard.model_dump(mode="json")

    assert payload["profile_count"] is None
    assert payload["attempted_profile_count"] is None
    assert payload["total_attempt_count"] is None
    assert all(
        summary["attempted_profile_count"] is None
        and summary["attempt_count"] is None
        for summary in payload["concept_summaries"]
    )
