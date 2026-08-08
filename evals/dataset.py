"""Schemas and validators for the physically separated v0.2 evaluation sets."""

import hashlib
import json
from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from probstat_tutor.curriculum import load_default_question_bank
from probstat_tutor.schemas import CapabilityDimension, ConceptId, Question


class SourceType(StrEnum):
    """How an evaluation case was created."""

    EXPERT_CREATED = "expert_created"
    STUDENT_SIMULATED = "student_simulated"
    ERROR_PATTERN = "error_pattern"
    ADVERSARIAL = "adversarial"


class DifficultyLevel(StrEnum):
    """Human-labelled difficulty of diagnosing the response, not question difficulty."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class CoverageGroup(StrEnum):
    """The four milestone-level case groups with frozen target counts."""

    EXISTING_ERROR_VARIANT = "existing_error_variant"
    NEW_MISCONCEPTION = "new_misconception"
    BOUNDARY_CASE = "boundary_case"
    ADVERSARIAL_INPUT = "adversarial_input"


class TagOrigin(StrEnum):
    """Whether an expected tag is from the question bank or only from evaluation."""

    QUESTION_BANK = "question_bank"
    EVALUATION_EXTENSION = "evaluation_extension"


class V2EvalCase(BaseModel):
    """One v0.2 case with input, human labels, provenance, and review evidence."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    split: Literal["development", "blind"]
    categories: list[str] = Field(min_length=1)
    coverage_group: CoverageGroup
    question_id: str = Field(min_length=1)
    concept_id: ConceptId
    capability_dimension: CapabilityDimension
    source_type: SourceType
    difficulty_level: DifficultyLevel
    answer: str
    reasoning: str = ""
    python_code: str = ""
    hint_level: int = Field(default=0, ge=0, le=3)
    expected_correct: bool
    expected_misconception_tags: list[str]
    expected_action: Literal[
        "retry_with_guidance",
        "next_question",
        "complete",
    ]
    forbidden_hint_tokens: list[str] = Field(default_factory=list)
    rubric_item_indices: list[int] = Field(default_factory=list)
    observable_evidence: list[str] = Field(min_length=1)
    tag_origins: dict[str, TagOrigin] = Field(default_factory=dict)
    variant_family: str = Field(min_length=1)

    @model_validator(mode="after")
    def label_metadata_is_complete(self) -> Self:
        expected_tags = set(self.expected_misconception_tags)
        if len(expected_tags) != len(self.expected_misconception_tags):
            raise ValueError("expected_misconception_tags 不能重复")
        if set(self.tag_origins) != expected_tags:
            raise ValueError("tag_origins 必须为每个 expected_misconception_tag 标明来源")
        if self.expected_correct and self.expected_action == "retry_with_guidance":
            raise ValueError("正确案例不能要求 retry_with_guidance")
        if not self.expected_correct and self.expected_action != "retry_with_guidance":
            raise ValueError("错误案例必须要求 retry_with_guidance")
        return self

    def to_legacy_payload(self) -> dict[str, object]:
        """Return only fields understood by the unchanged v0.1 evaluation engine."""

        return {
            "id": self.id,
            "categories": self.categories,
            "question_id": self.question_id,
            "answer": self.answer,
            "reasoning": self.reasoning,
            "python_code": self.python_code,
            "hint_level": self.hint_level,
            "expected_correct": self.expected_correct,
            "expected_misconception_tags": self.expected_misconception_tags,
            "expected_action": self.expected_action,
            "forbidden_hint_tokens": self.forbidden_hint_tokens,
        }


class V2DatasetDistribution(BaseModel):
    """Counts used to check the approved 48-case design."""

    model_config = ConfigDict(extra="forbid")

    by_concept_id: dict[str, int]
    by_capability_dimension: dict[str, int]
    by_coverage_group: dict[str, int]
    by_source_type: dict[str, int]
    by_difficulty_level: dict[str, int]
    by_category: dict[str, int]


def load_v2_cases(
    path: str | Path,
    *,
    expected_split: Literal["development", "blind"],
    questions: list[Question] | None = None,
) -> list[V2EvalCase]:
    """Load one v0.2 JSONL split and validate every reference against the question bank."""

    case_path = Path(path)
    cases: list[V2EvalCase] = []
    for line_number, line in enumerate(
        case_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            case = V2EvalCase.model_validate_json(line)
        except Exception as error:
            raise ValueError(
                f"v0.2 评测案例第 {line_number} 行无效：{error}"
            ) from error
        if case.split != expected_split:
            raise ValueError(
                f"案例 {case.id} 的 split={case.split}，应为 {expected_split}"
            )
        cases.append(case)

    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{expected_split} 数据集内的案例 ID 不能重复")

    question_list = questions or load_default_question_bank().questions
    questions_by_id = {question.id: question for question in question_list}
    for case in cases:
        _validate_question_references(case, questions_by_id)
    return cases


def validate_combined_v2_cases(
    development_cases: list[V2EvalCase],
    blind_cases: list[V2EvalCase],
    *,
    legacy_ids: set[str] | None = None,
) -> None:
    """Validate global IDs, input uniqueness, and split-independent variant families."""

    all_cases = [*development_cases, *blind_cases]
    all_ids = [case.id for case in all_cases]
    if legacy_ids:
        all_ids.extend(legacy_ids)
    duplicates = sorted(
        case_id for case_id, count in Counter(all_ids).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"三个数据集存在重复 ID：{', '.join(duplicates)}")

    hashes = [normalized_input_sha256(case) for case in all_cases]
    duplicate_hashes = sorted(
        value for value, count in Counter(hashes).items() if count > 1
    )
    if duplicate_hashes:
        raise ValueError(
            "开发集与盲测集存在完全重复的学习者输入："
            f"{', '.join(duplicate_hashes)}"
        )

    family_splits: dict[str, set[str]] = {}
    for case in all_cases:
        family_splits.setdefault(case.variant_family, set()).add(case.split)
    crossed = sorted(
        family for family, splits in family_splits.items() if len(splits) > 1
    )
    if crossed:
        raise ValueError(
            "variant_family 不能同时出现在开发集和盲测集："
            f"{', '.join(crossed)}"
        )


def normalized_input_sha256(case: V2EvalCase) -> str:
    """Hash learner-visible input only, excluding expected labels and runtime data."""

    payload = {
        "question_id": case.question_id,
        "answer": case.answer,
        "reasoning": case.reasoning,
        "python_code": case.python_code,
        "hint_level": case.hint_level,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def analyze_v2_distribution(cases: list[V2EvalCase]) -> V2DatasetDistribution:
    """Return explicit multi-label and single-label distribution counts."""

    return V2DatasetDistribution(
        by_concept_id=_enum_counts(cases, "concept_id", ConceptId),
        by_capability_dimension=_enum_counts(
            cases,
            "capability_dimension",
            CapabilityDimension,
        ),
        by_coverage_group=_enum_counts(cases, "coverage_group", CoverageGroup),
        by_source_type=_enum_counts(cases, "source_type", SourceType),
        by_difficulty_level=_enum_counts(
            cases,
            "difficulty_level",
            DifficultyLevel,
        ),
        by_category=dict(
            sorted(Counter(category for case in cases for category in case.categories).items())
        ),
    )


def _validate_question_references(
    case: V2EvalCase,
    questions_by_id: dict[str, Question],
) -> None:
    try:
        question = questions_by_id[case.question_id]
    except KeyError as error:
        raise ValueError(
            f"案例 {case.id} 引用了不存在的题目：{case.question_id}"
        ) from error

    if case.concept_id != question.concept_id:
        raise ValueError(
            f"案例 {case.id} 的 concept_id 与题目 {case.question_id} 不一致"
        )
    dimension_weight = getattr(
        question.dimension_weights,
        case.capability_dimension.value,
    )
    if dimension_weight <= 0:
        raise ValueError(
            f"案例 {case.id} 的能力维度 {case.capability_dimension.value} "
            f"在题目 {case.question_id} 中没有观察权重"
        )

    invalid_rubric_indices = sorted(
        index
        for index in case.rubric_item_indices
        if index < 0 or index >= len(question.rubric)
    )
    if invalid_rubric_indices:
        raise ValueError(
            f"案例 {case.id} 引用了不存在的 rubric 下标："
            f"{invalid_rubric_indices}"
        )

    question_tags = set(question.misconception_tags)
    invalid_bank_tags = sorted(
        tag
        for tag, origin in case.tag_origins.items()
        if origin == TagOrigin.QUESTION_BANK and tag not in question_tags
    )
    if invalid_bank_tags:
        raise ValueError(
            f"案例 {case.id} 把非题库标签错误标记为 question_bank："
            f"{', '.join(invalid_bank_tags)}"
        )


def _enum_counts(
    cases: list[V2EvalCase],
    field_name: str,
    enum_type: type[StrEnum],
) -> dict[str, int]:
    counter = Counter(getattr(case, field_name).value for case in cases)
    return {
        member.value: counter[member.value]
        for member in enum_type
        if counter[member.value] > 0
    }
