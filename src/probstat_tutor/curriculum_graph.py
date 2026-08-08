"""Validated eight-unit curriculum graph and 22-chapter directory mapping."""

from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from probstat_tutor.config import get_settings
from probstat_tutor.curriculum import load_default_question_bank
from probstat_tutor.schemas import (
    CapabilityDimension,
    ContentReviewStatus,
    DeepUnitId,
    QuestionBank,
    QuestionType,
)


class UnitContentStatus(StrEnum):
    """Construction and human-review state of one deep unit."""

    PLANNED = "planned"
    DRAFT = "draft"
    PENDING_TEACHER_REVIEW = "pending_teacher_review"
    APPROVED = "approved"


class TextbookId(StrEnum):
    """The two school-selected books represented in the directory map."""

    PROBABILITY_STATISTICS = "probability_statistics"
    PYTHON_DATA_ANALYSIS = "python_data_analysis"


class CoverageLevel(StrEnum):
    """A directory link is not automatically a claim of deep coverage."""

    DIRECTORY_ONLY = "directory_only"
    SUPPORTING = "supporting"
    DEEP = "deep"


class VerificationStatus(StrEnum):
    """Whether a human has checked a chapter mapping against the issued edition."""

    PENDING_TEACHER_REVIEW = "pending_teacher_review"
    VERIFIED = "verified"


class KnowledgeRelation(StrEnum):
    """Supported directed relationships between knowledge nodes."""

    PREREQUISITE = "prerequisite"
    SUPPORTS = "supports"
    APPLIES_TO = "applies_to"
    INTERPRETS = "interprets"


class DeepUnit(BaseModel):
    """One of the eight deep diagnostic units."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    unit_id: DeepUnitId
    order: int = Field(ge=1, le=8)
    title_zh: str = Field(min_length=1, max_length=120)
    summary_zh: str = Field(min_length=1, max_length=500)
    learning_objectives: tuple[str, ...] = Field(min_length=1)
    prerequisite_unit_ids: tuple[DeepUnitId, ...] = ()
    knowledge_node_ids: tuple[str, ...] = Field(min_length=1)
    question_ids: tuple[str, ...] = ()
    source_ids: tuple[str, ...] = ()
    content_status: UnitContentStatus
    minimum_question_count: int = Field(default=3, ge=3)
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")


class KnowledgeNode(BaseModel):
    """A teachable, observable node inside one deep unit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    unit_id: DeepUnitId
    title_zh: str = Field(min_length=1, max_length=120)
    learning_objective: str = Field(min_length=1, max_length=500)
    capability_dimensions: tuple[CapabilityDimension, ...] = Field(min_length=1)
    key_terms: tuple[str, ...] = Field(min_length=1)
    observable_success: str = Field(min_length=1, max_length=500)
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")


class KnowledgeEdge(BaseModel):
    """One directed, explainable relationship in the knowledge graph."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_node_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    target_node_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    relation: KnowledgeRelation
    rationale_zh: str = Field(min_length=1, max_length=500)


class TextbookChapterMapping(BaseModel):
    """A directory-level chapter link with an explicit claim boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    textbook_id: TextbookId
    edition: str = Field(min_length=1, max_length=120)
    chapter_number: int = Field(ge=1, le=30)
    chapter_title: str = Field(min_length=1, max_length=160)
    coverage_level: CoverageLevel
    unit_ids: tuple[DeepUnitId, ...] = Field(min_length=1)
    knowledge_node_ids: tuple[str, ...] = ()
    mapping_rationale_zh: str = Field(min_length=1, max_length=500)
    verification_status: VerificationStatus


class CurriculumCatalog(BaseModel):
    """Eight-unit graph plus the exact 8+14 chapter directory skeleton."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str
    coverage_level_semantics: Literal["target_only_not_implementation_status"]
    units: tuple[DeepUnit, ...] = Field(min_length=8, max_length=8)
    knowledge_nodes: tuple[KnowledgeNode, ...] = Field(min_length=8)
    edges: tuple[KnowledgeEdge, ...] = Field(min_length=1)
    chapter_mappings: tuple[TextbookChapterMapping, ...] = Field(
        min_length=22,
        max_length=22,
    )

    @model_validator(mode="after")
    def graph_is_complete_and_consistent(self) -> Self:
        self._validate_units()
        self._validate_nodes_and_edges()
        self._validate_chapter_directory()
        return self

    def _validate_units(self) -> None:
        unit_ids = [unit.unit_id for unit in self.units]
        if set(unit_ids) != set(DeepUnitId) or len(unit_ids) != len(set(unit_ids)):
            raise ValueError("课程图谱必须不重不漏地包含 8 个深度单元")
        if {unit.order for unit in self.units} != set(range(1, 9)):
            raise ValueError("8 个深度单元的 order 必须不重不漏地覆盖 1–8")
        for unit in self.units:
            if unit.unit_id in unit.prerequisite_unit_ids:
                raise ValueError(f"单元 {unit.unit_id.value} 不能把自身设为前置")
        _assert_acyclic(
            {
                unit.unit_id.value: tuple(item.value for item in unit.prerequisite_unit_ids)
                for unit in self.units
            },
            "深度单元前置图不能成环",
        )

    def _validate_nodes_and_edges(self) -> None:
        node_ids = [node.node_id for node in self.knowledge_nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("知识节点 ID 不能重复")
        nodes_by_id = {node.node_id: node for node in self.knowledge_nodes}
        for unit in self.units:
            declared = set(unit.knowledge_node_ids)
            actual = {
                node.node_id for node in self.knowledge_nodes if node.unit_id == unit.unit_id
            }
            if declared != actual:
                raise ValueError(f"单元 {unit.unit_id.value} 的知识节点清单与节点定义不一致")
        edge_keys: set[tuple[str, str, KnowledgeRelation]] = set()
        for edge in self.edges:
            if edge.source_node_id not in nodes_by_id or edge.target_node_id not in nodes_by_id:
                raise ValueError("知识边引用了不存在的节点")
            if edge.source_node_id == edge.target_node_id:
                raise ValueError("知识节点不能指向自身")
            key = (edge.source_node_id, edge.target_node_id, edge.relation)
            if key in edge_keys:
                raise ValueError("知识边不能重复")
            edge_keys.add(key)
        prerequisite_graph = {node_id: [] for node_id in node_ids}
        for edge in self.edges:
            if edge.relation == KnowledgeRelation.PREREQUISITE:
                prerequisite_graph[edge.target_node_id].append(edge.source_node_id)
        _assert_acyclic(
            {key: tuple(value) for key, value in prerequisite_graph.items()},
            "知识节点前置图不能成环",
        )

    def _validate_chapter_directory(self) -> None:
        expected = {
            TextbookId.PROBABILITY_STATISTICS: set(range(1, 9)),
            TextbookId.PYTHON_DATA_ANALYSIS: set(range(1, 15)),
        }
        actual: dict[TextbookId, list[int]] = {book: [] for book in TextbookId}
        node_ids = {node.node_id for node in self.knowledge_nodes}
        mapped_units: set[DeepUnitId] = set()
        for mapping in self.chapter_mappings:
            actual[mapping.textbook_id].append(mapping.chapter_number)
            mapped_units.update(mapping.unit_ids)
            if not set(mapping.knowledge_node_ids) <= node_ids:
                raise ValueError("章节映射引用了不存在的知识节点")
        for textbook_id, chapter_numbers in actual.items():
            if len(chapter_numbers) != len(set(chapter_numbers)):
                raise ValueError(f"教材 {textbook_id.value} 的章节号不能重复")
            if set(chapter_numbers) != expected[textbook_id]:
                raise ValueError(f"教材 {textbook_id.value} 的章节目录不完整")
        if mapped_units != set(DeepUnitId):
            raise ValueError("22 章目录映射必须至少关联到全部 8 个深度单元")


class CurriculumGraphLoadError(ValueError):
    """A beginner-readable curriculum catalog loading error."""


def load_curriculum_catalog(
    path: str | Path,
    *,
    question_bank: QuestionBank | None = None,
) -> CurriculumCatalog:
    """Load the catalog and optionally validate its published question links."""

    catalog_path = Path(path)
    try:
        raw_data = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise CurriculumGraphLoadError(f"课程图谱文件不存在：{catalog_path}") from error
    except OSError as error:
        raise CurriculumGraphLoadError(f"无法读取课程图谱 {catalog_path}：{error}") from error
    except yaml.YAMLError as error:
        raise CurriculumGraphLoadError(f"课程图谱 YAML 格式错误：{error}") from error
    if not isinstance(raw_data, dict):
        raise CurriculumGraphLoadError("课程图谱顶层必须是 YAML 对象")
    try:
        catalog = CurriculumCatalog.model_validate(raw_data)
    except ValidationError as error:
        raise CurriculumGraphLoadError(f"课程图谱未通过 Pydantic 验证：\n{error}") from error
    if question_bank is not None:
        validate_catalog_question_readiness(catalog, question_bank)
    return catalog


def load_default_curriculum_catalog() -> CurriculumCatalog:
    """Load and cross-check the configured local curriculum artifacts."""

    settings = get_settings()
    return load_curriculum_catalog(
        settings.curriculum_catalog_path,
        question_bank=load_default_question_bank(),
    )


def validate_catalog_question_readiness(
    catalog: CurriculumCatalog,
    question_bank: QuestionBank,
) -> None:
    """Cross-check declared questions and enforce readiness only for ready units."""

    questions_by_unit = {
        unit_id: tuple(
            question for question in question_bank.questions if question.unit_id == unit_id
        )
        for unit_id in DeepUnitId
    }
    for unit in catalog.units:
        questions = questions_by_unit[unit.unit_id]
        actual_ids = {question.id for question in questions}
        if actual_ids != set(unit.question_ids):
            raise CurriculumGraphLoadError(
                f"单元 {unit.unit_id.value} 的 question_ids 与题库不一致"
            )
        if unit.content_status not in {
            UnitContentStatus.PENDING_TEACHER_REVIEW,
            UnitContentStatus.APPROVED,
        }:
            continue
        if len(questions) < unit.minimum_question_count:
            raise CurriculumGraphLoadError(
                f"待审或已审单元 {unit.unit_id.value} 的题目少于最低数量"
            )
        if not set(QuestionType) <= {question.question_type for question in questions}:
            raise CurriculumGraphLoadError(
                f"待审或已审单元 {unit.unit_id.value} 未覆盖三类题型"
            )
        for question in questions:
            if question.hints is None or not question.knowledge_node_ids:
                raise CurriculumGraphLoadError(
                    f"待审题目 {question.id} 缺少四级提示或知识节点"
                )
            if question.review_status not in {
                ContentReviewStatus.PENDING_TEACHER_REVIEW,
                ContentReviewStatus.APPROVED,
            }:
                raise CurriculumGraphLoadError(f"待审题目 {question.id} 的审核状态不正确")
            if not set(question.knowledge_node_ids) <= set(unit.knowledge_node_ids):
                raise CurriculumGraphLoadError(f"题目 {question.id} 引用了单元外知识节点")
            if question.question_type == QuestionType.PYTHON and not question.python_code_required:
                raise CurriculumGraphLoadError(f"待审 Python 题 {question.id} 必须提交代码")


def _assert_acyclic(graph: dict[str, tuple[str, ...]], message: str) -> None:
    visited: set[str] = set()
    active: set[str] = set()

    def visit(node: str) -> None:
        if node in active:
            raise ValueError(message)
        if node in visited:
            return
        active.add(node)
        for prerequisite in graph[node]:
            visit(prerequisite)
        active.remove(node)
        visited.add(node)

    for node in graph:
        visit(node)
