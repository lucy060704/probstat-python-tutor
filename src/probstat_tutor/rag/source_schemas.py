"""Structured contracts for reviewed RAG course source documents."""

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from probstat_tutor.rag.schemas import (
    AllowedUsage,
    AnswerLeakageRisk,
    KnowledgeSource,
    SourceLanguage,
    SourceLicense,
)
from probstat_tutor.schemas import ConceptId

SUPPORTED_SOURCE_SCHEMA_VERSION = "1.0"

FORBIDDEN_SOURCE_KEYS = {
    "dataset",
    "dimension_weights",
    "expected_answer",
    "misconception_tags",
    "numeric_tolerance",
    "question_id",
    "rubric",
}


class FormulaExplanation(BaseModel):
    """A formula's meaning and assumptions without a question-specific solution."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=120)
    expression: str = Field(min_length=1, max_length=300)
    symbols: dict[str, str] = Field(min_length=1)
    meaning: str = Field(min_length=1, max_length=1200)
    assumptions: list[str] = Field(min_length=1)
    cautions: list[str] = Field(min_length=1)


class PythonConnection(BaseModel):
    """A safe API-level connection that does not execute learner code."""

    model_config = ConfigDict(extra="forbid")

    library: str = Field(min_length=1, max_length=80)
    api: str = Field(min_length=1, max_length=120)
    purpose: str = Field(min_length=1, max_length=800)
    input_expectation: str = Field(min_length=1, max_length=800)
    interpretation_caution: str = Field(min_length=1, max_length=1000)


class MisconceptionExplanation(BaseModel):
    """General misconception guidance, not a grader label or trigger rule."""

    model_config = ConfigDict(extra="forbid")

    misconception: str = Field(min_length=1, max_length=500)
    why_incorrect: str = Field(min_length=1, max_length=1200)
    better_question: str = Field(min_length=1, max_length=500)


class RagSourceDocument(BaseModel):
    """One beginner-facing course source tied to exactly one curriculum concept."""

    model_config = ConfigDict(extra="forbid")

    source_schema_version: str
    source_id: str = Field(
        min_length=3,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_]*$",
    )
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    title: str = Field(min_length=1, max_length=200)
    language: SourceLanguage
    concept_id: ConceptId
    learning_objectives: list[str] = Field(min_length=1)
    prerequisite_knowledge: list[str] = Field(min_length=1)
    concept_explanation: list[str] = Field(min_length=1)
    formula_explanation: list[FormulaExplanation] = Field(min_length=1)
    python_connection: list[PythonConnection] = Field(min_length=1)
    data_interpretation_guidance: list[str] = Field(min_length=1)
    common_misconceptions: list[MisconceptionExplanation] = Field(min_length=1)
    reflective_questions: list[str] = Field(min_length=1)
    summary: list[str] = Field(min_length=1)

    @field_validator("source_schema_version")
    @classmethod
    def source_schema_version_is_supported(cls, value: str) -> str:
        if value != SUPPORTED_SOURCE_SCHEMA_VERSION:
            raise ValueError(
                f"暂不支持 source_schema_version={value!r}，"
                f"当前仅支持 {SUPPORTED_SOURCE_SCHEMA_VERSION}"
            )
        return value

    @model_validator(mode="after")
    def text_lists_do_not_contain_blank_items(self) -> Self:
        for field_name in (
            "learning_objectives",
            "prerequisite_knowledge",
            "concept_explanation",
            "data_interpretation_guidance",
            "reflective_questions",
            "summary",
        ):
            values = getattr(self, field_name)
            if any(not value.strip() for value in values):
                raise ValueError(f"{field_name} 不能包含空白条目")
        return self


class EligibilityRejectionCode(StrEnum):
    """Stable reasons why a valid source cannot enter future chunking."""

    RETRIEVAL_NOT_ALLOWED = "retrieval_not_allowed"
    LICENSE_PERMISSION_REQUIRED = "license_permission_required"
    ANSWER_LEAKAGE_RISK_HIGH = "answer_leakage_risk_high"
    ANSWER_LEAKAGE_RISK_PROHIBITED = "answer_leakage_risk_prohibited"


class ChunkingEligibility(BaseModel):
    """Policy decision separated from file integrity and schema validation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    eligible_for_chunking: bool
    rejection_reasons: list[str]
    rejection_codes: list[EligibilityRejectionCode]

    @model_validator(mode="after")
    def decision_is_internally_consistent(self) -> Self:
        if self.eligible_for_chunking and (
            self.rejection_reasons or self.rejection_codes
        ):
            raise ValueError("允许切片时不能同时包含拒绝原因")
        if not self.eligible_for_chunking and (
            not self.rejection_reasons or not self.rejection_codes
        ):
            raise ValueError("不允许切片时必须提供明确的拒绝原因和代码")
        if len(self.rejection_reasons) != len(self.rejection_codes):
            raise ValueError("rejection_reasons 和 rejection_codes 必须一一对应")
        return self


class LoadedRagSource(BaseModel):
    """One safely loaded, integrity-checked source and its policy decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_entry: KnowledgeSource
    document: RagSourceDocument
    relative_path: str
    content_checksum: str
    eligibility: ChunkingEligibility


def assess_chunking_eligibility(source: KnowledgeSource) -> ChunkingEligibility:
    """Decide whether an already validated source may enter future chunking."""

    reasons: list[str] = []
    codes: list[EligibilityRejectionCode] = []

    if AllowedUsage.RETRIEVAL not in source.allowed_usage:
        codes.append(EligibilityRejectionCode.RETRIEVAL_NOT_ALLOWED)
        reasons.append("allowed_usage 未包含 retrieval，不能进入教学检索切片流程")
    if source.license == SourceLicense.PERMISSION_REQUIRED:
        codes.append(EligibilityRejectionCode.LICENSE_PERMISSION_REQUIRED)
        reasons.append("资料授权仍为 permission-required，完成授权前不能切片")
    if source.answer_leakage_risk == AnswerLeakageRisk.HIGH:
        codes.append(EligibilityRejectionCode.ANSWER_LEAKAGE_RISK_HIGH)
        reasons.append("答案泄露风险为 high，超过当前允许进入切片的最高等级 medium")
    elif source.answer_leakage_risk == AnswerLeakageRisk.PROHIBITED:
        codes.append(EligibilityRejectionCode.ANSWER_LEAKAGE_RISK_PROHIBITED)
        reasons.append("答案泄露风险为 prohibited，只允许审查，不能进入切片")

    return ChunkingEligibility(
        eligible_for_chunking=not codes,
        rejection_reasons=reasons,
        rejection_codes=codes,
    )
