"""Validated data models shared by the curriculum and later milestones."""

from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GradeResult(BaseModel):
    """Unified, deterministic result returned by every grader."""

    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0.0, le=1.0)
    is_correct: bool
    evidence: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    misconception_candidates: list[str] = Field(default_factory=list)


class ConceptId(StrEnum):
    """Knowledge points included in the v0.1 curriculum."""

    MEAN_MEDIAN = "mean_median"
    VARIANCE_STD = "variance_std"
    SAMPLING_STANDARD_ERROR = "sampling_standard_error"
    CONFIDENCE_INTERVAL = "confidence_interval"


class QuestionType(StrEnum):
    """The learning dimension primarily observed by a question."""

    CONCEPT = "concept"
    PYTHON = "python"
    INTERPRETATION = "interpretation"


class CapabilityDimension(StrEnum):
    """The four separately tracked learner capabilities."""

    CONCEPT = "concept"
    CALCULATION = "calculation"
    PYTHON = "python"
    INTERPRETATION = "interpretation"


class DimensionWeights(BaseModel):
    """Relative contribution of one question to the four-dimensional profile."""

    model_config = ConfigDict(extra="forbid")

    concept: float = Field(ge=0.0, le=1.0)
    calculation: float = Field(ge=0.0, le=1.0)
    python: float = Field(ge=0.0, le=1.0)
    interpretation: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> Self:
        total = self.concept + self.calculation + self.python + self.interpretation
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"dimension_weights 之和必须为 1，当前为 {total:g}")
        return self


class MasteryScores(BaseModel):
    """Mastery scores for one knowledge point."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    concept: float = Field(default=0.5, ge=0.0, le=1.0)
    calculation: float = Field(default=0.5, ge=0.0, le=1.0)
    python: float = Field(default=0.5, ge=0.0, le=1.0)
    interpretation: float = Field(default=0.5, ge=0.0, le=1.0)


class AttemptRecord(BaseModel):
    """Minimal deterministic evidence needed by the recommendation policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    question_id: str
    concept_id: ConceptId
    difficulty: float = Field(ge=0.0, le=1.0)
    score: float = Field(ge=0.0, le=1.0)
    adjusted_evidence: float = Field(ge=0.0, le=1.0)
    is_correct: bool
    hint_level: int = Field(ge=0, le=3)


def _initial_mastery() -> dict[ConceptId, MasteryScores]:
    return {concept: MasteryScores() for concept in ConceptId}


class LearningState(BaseModel):
    """In-memory v0.1 state, indexed by knowledge point and capability."""

    model_config = ConfigDict(extra="forbid")

    mastery: dict[ConceptId, MasteryScores] = Field(default_factory=_initial_mastery)
    history: tuple[AttemptRecord, ...] = ()
    completed_question_ids: frozenset[str] = frozenset()

    @model_validator(mode="after")
    def contains_every_concept(self) -> Self:
        missing = set(ConceptId) - set(self.mastery)
        if missing:
            names = ", ".join(sorted(concept.value for concept in missing))
            raise ValueError(f"学习状态缺少知识点：{names}")
        return self


class PolicyStatus(StrEnum):
    """Possible outcomes of next-question selection."""

    QUESTION = "question"
    COMPLETE = "complete"
    BLOCKED = "blocked"


class NextQuestionDecision(BaseModel):
    """A deterministic next-step decision with a beginner-readable reason."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: PolicyStatus
    question_id: str | None = None
    target_dimension: CapabilityDimension | None = None
    recommended_hint_level: int = Field(default=0, ge=0, le=3)
    reason: str


class DiagnosticReport(BaseModel):
    """Structured tutor output whose factual fields come from deterministic tools."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    overall_correctness: float = Field(ge=0.0, le=1.0)
    dimension_scores: MasteryScores
    evidence: list[str] = Field(min_length=1)
    misconception_tags: list[str]
    feedback: str
    hint_level: int = Field(ge=0, le=3)
    recommended_action: str
    next_question_id: str | None
    uncertainty: str


class Question(BaseModel):
    """One validated curriculum question."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, pattern=r"^[a-z0-9_]+$")
    title: str = Field(min_length=1)
    concept_id: ConceptId
    prerequisites: list[ConceptId]
    difficulty: float = Field(ge=0.0, le=1.0)
    question_type: QuestionType
    prompt: str = Field(min_length=1)
    dataset: dict[str, Any]
    expected_answer: Any
    numeric_tolerance: float | None = Field(default=None, ge=0.0)
    rubric: list[str] = Field(min_length=1)
    misconception_tags: list[str]
    dimension_weights: DimensionWeights


class QuestionBank(BaseModel):
    """The complete, internally consistent v0.1 question bank."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(ge=1)
    questions: list[Question] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_bank_integrity(self) -> Self:
        question_ids = [question.id for question in self.questions]
        if len(question_ids) != len(set(question_ids)):
            raise ValueError("题目 ID 不能重复")

        available_concepts = {question.concept_id for question in self.questions}
        for question in self.questions:
            missing = set(question.prerequisites) - available_concepts
            if missing:
                names = ", ".join(sorted(concept.value for concept in missing))
                raise ValueError(f"题目 {question.id} 引用了不存在的前置知识点: {names}")

        for concept in ConceptId:
            concept_questions = [
                question for question in self.questions if question.concept_id == concept
            ]
            if len(concept_questions) != 3:
                raise ValueError(f"知识点 {concept.value} 必须正好包含三道题")
            actual_types = {question.question_type for question in concept_questions}
            if actual_types != set(QuestionType):
                raise ValueError(f"知识点 {concept.value} 必须各有一道概念、Python 和解释题")

        return self
