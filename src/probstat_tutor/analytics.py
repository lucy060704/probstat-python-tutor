"""Privacy-preserving teacher aggregates derived only from learner state."""

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from probstat_tutor.schemas import ConceptId, LearningState, MasteryScores

DEFAULT_MINIMUM_COHORT_SIZE = 3


class TeacherDashboardStatus(StrEnum):
    """Whether anonymous aggregate metrics are safe to display."""

    NO_DATA = "no_data"
    SUPPRESSED = "suppressed"
    READY = "ready"


class TeacherConceptSummary(BaseModel):
    """One concept aggregate; small cells suppress exact counts and outcomes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    concept_id: ConceptId
    attempted_profile_count: int | None = Field(default=None, ge=0)
    attempt_count: int | None = Field(default=None, ge=0)
    correct_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    average_hint_level: float | None = Field(default=None, ge=0.0, le=4.0)
    average_mastery: MasteryScores | None = None
    suppressed: bool

    @model_validator(mode="after")
    def suppression_contract_is_consistent(self) -> Self:
        private_values = (
            self.attempted_profile_count,
            self.attempt_count,
            self.correct_rate,
            self.average_hint_level,
            self.average_mastery,
        )
        if self.suppressed and any(value is not None for value in private_values):
            raise ValueError("小样本知识点不能包含精确计数或结果")
        if not self.suppressed and (
            self.attempted_profile_count is None
            or self.attempted_profile_count < DEFAULT_MINIMUM_COHORT_SIZE
            or self.attempt_count is None
            or self.correct_rate is None
            or self.average_hint_level is None
            or self.average_mastery is None
        ):
            raise ValueError("可见知识点必须包含达到最小群组的完整聚合")
        return self


class TeacherDashboard(BaseModel):
    """Anonymous summary DTO that cannot contain learner IDs or raw answers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: TeacherDashboardStatus
    minimum_cohort_size: int = Field(ge=3)
    profile_count: int | None = Field(default=None, ge=0)
    attempted_profile_count: int | None = Field(default=None, ge=0)
    total_attempt_count: int | None = Field(default=None, ge=0)
    overall_correct_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    concept_summaries: tuple[TeacherConceptSummary, ...]
    privacy_notice_zh: str

    @model_validator(mode="after")
    def status_matches_visible_counts(self) -> Self:
        counts = (
            self.profile_count,
            self.attempted_profile_count,
            self.total_attempt_count,
        )
        if self.status == TeacherDashboardStatus.NO_DATA:
            if counts != (0, 0, 0) or self.overall_correct_rate is not None:
                raise ValueError("无数据状态只能显示零计数，不能显示结果")
        elif self.status == TeacherDashboardStatus.SUPPRESSED:
            if any(value is not None for value in counts) or self.overall_correct_rate is not None:
                raise ValueError("小样本总体不能包含精确计数或结果")
        elif (
            any(value is None for value in counts)
            or self.profile_count < self.minimum_cohort_size  # type: ignore[operator]
            or self.attempted_profile_count < self.minimum_cohort_size  # type: ignore[operator]
            or self.total_attempt_count < self.attempted_profile_count  # type: ignore[operator]
            or self.overall_correct_rate is None
        ):
            raise ValueError("可见总体必须包含达到最小群组的完整聚合")
        return self


def build_teacher_dashboard(
    states: tuple[LearningState, ...],
    *,
    minimum_cohort_size: int = DEFAULT_MINIMUM_COHORT_SIZE,
) -> TeacherDashboard:
    """Aggregate validated states while suppressing every cell smaller than k."""

    if minimum_cohort_size < 3:
        raise ValueError("匿名汇总最小群组必须至少为 3")

    attempted_states = tuple(state for state in states if state.history)
    attempts = tuple(attempt for state in states for attempt in state.history)
    overall_visible = len(attempted_states) >= minimum_cohort_size
    status = (
        TeacherDashboardStatus.NO_DATA
        if not states
        else TeacherDashboardStatus.READY
        if overall_visible
        else TeacherDashboardStatus.SUPPRESSED
    )
    overall_correct_rate = (
        sum(attempt.is_correct for attempt in attempts) / len(attempts)
        if overall_visible and attempts
        else None
    )

    summaries = tuple(
        _build_concept_summary(states, concept_id, minimum_cohort_size)
        for concept_id in ConceptId
    )
    no_data = status == TeacherDashboardStatus.NO_DATA
    return TeacherDashboard(
        status=status,
        minimum_cohort_size=minimum_cohort_size,
        profile_count=0 if no_data else len(states) if overall_visible else None,
        attempted_profile_count=(
            0 if no_data else len(attempted_states) if overall_visible else None
        ),
        total_attempt_count=0 if no_data else len(attempts) if overall_visible else None,
        overall_correct_rate=overall_correct_rate,
        concept_summaries=summaries,
        privacy_notice_zh=(
            "本页只读取匿名学习状态，不读取作答回执或原始答案；少于 3 人的结果单元格自动隐藏。"
        ),
    )


def _build_concept_summary(
    states: tuple[LearningState, ...],
    concept_id: ConceptId,
    minimum_cohort_size: int,
) -> TeacherConceptSummary:
    attempted_states = tuple(
        state
        for state in states
        if any(attempt.concept_id == concept_id for attempt in state.history)
    )
    attempts = tuple(
        attempt
        for state in attempted_states
        for attempt in state.history
        if attempt.concept_id == concept_id
    )
    suppressed = len(attempted_states) < minimum_cohort_size
    if suppressed:
        return TeacherConceptSummary(
            concept_id=concept_id,
            suppressed=True,
        )

    mastery = MasteryScores(
        **{
            dimension: sum(
                getattr(state.mastery[concept_id], dimension) for state in attempted_states
            )
            / len(attempted_states)
            for dimension in ("concept", "calculation", "python", "interpretation")
        }
    )
    return TeacherConceptSummary(
        concept_id=concept_id,
        attempted_profile_count=len(attempted_states),
        attempt_count=len(attempts),
        correct_rate=sum(attempt.is_correct for attempt in attempts) / len(attempts),
        average_hint_level=sum(attempt.hint_level for attempt in attempts) / len(attempts),
        average_mastery=mastery,
        suppressed=False,
    )
