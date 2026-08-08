"""Evaluate the deterministic local RAG without using a model or learner answers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from probstat_tutor.config import PROJECT_ROOT
from probstat_tutor.curriculum import load_default_question_bank
from probstat_tutor.rag import (
    LocalRagIndex,
    RagChunk,
    RagQuery,
    RagSearchHit,
    RagSearchResult,
    RagSection,
    RetrievalPurpose,
    build_local_rag_index,
    infer_minimum_disclosure_level,
)
from probstat_tutor.schemas import ConceptId, DeepUnitId, KnowledgeContextStatus

DEVELOPMENT_DATASET_PATH = PROJECT_ROOT / "evals" / "rag" / "development.jsonl"
DEVELOPMENT_MANIFEST_PATH = (
    PROJECT_ROOT / "evals" / "rag" / "development_manifest.json"
)
RETRIEVAL_CONTRACT_VERSION = "g3.2-v1"
REPLAY_COUNT = 3
MINIMUM_DEVELOPMENT_CASES = 64
MINIMUM_DEVELOPMENT_MATCHED_CASES_PER_UNIT = 7
MINIMUM_HOLDOUT_CASES = 48
MINIMUM_HOLDOUT_MATCHED_CASES_PER_UNIT = 5
MINIMUM_NO_MATCH_CASES = 8
MINIMUM_QUERY_ONLY_MATCHED_CASES = 8
MINIMUM_POSITIVE_CASES_PER_SOURCE = 2
MINIMUM_PROTECTED_LEVEL_ONE_CASES = 8
MAXIMUM_NEAR_DUPLICATE_JACCARD = 0.80

UNIT_CONCEPTS: dict[DeepUnitId, frozenset[ConceptId]] = {
    DeepUnitId.DATA_QUALITY: frozenset({ConceptId.DATA_QUALITY}),
    DeepUnitId.DESCRIPTIVE_STATISTICS: frozenset(
        {ConceptId.MEAN_MEDIAN, ConceptId.VARIANCE_STD}
    ),
    DeepUnitId.PROBABILITY_SIMULATION: frozenset(
        {ConceptId.PROBABILITY_SIMULATION}
    ),
    DeepUnitId.COMMON_DISTRIBUTIONS: frozenset({ConceptId.COMMON_DISTRIBUTIONS}),
    DeepUnitId.JOINT_CORRELATION: frozenset({ConceptId.JOINT_CORRELATION}),
    DeepUnitId.SAMPLING_INFERENCE: frozenset({ConceptId.SAMPLING_STANDARD_ERROR}),
    DeepUnitId.ESTIMATION_CONFIDENCE_INTERVAL: frozenset(
        {ConceptId.CONFIDENCE_INTERVAL}
    ),
    DeepUnitId.HYPOTHESIS_TESTING: frozenset({ConceptId.HYPOTHESIS_TESTING}),
}


class RagEvalTarget(BaseModel):
    """One source with an explicit, non-Cartesian set of labelled sections."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    sections: tuple[RagSection, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def sections_are_unique(self) -> Self:
        if len(self.sections) != len(set(self.sections)):
            raise ValueError("target sections 不能重复")
        return self


class RagEvalCase(BaseModel):
    """One human-labelled retrieval query without a formal-question answer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    id: str = Field(pattern=r"^rag_(?:dev|holdout)_[a-z0-9_]+$")
    split: Literal["development", "holdout"]
    origin: Literal["team_authored", "independent_judge"]
    query_family: str = Field(pattern=r"^[a-z][a-z0-9_]{2,79}$")
    query_style: Literal[
        "learner_paraphrase",
        "formal_term",
        "formula_request",
        "python_request",
        "summary_request",
        "out_of_domain",
    ] = "learner_paraphrase"
    metadata_mode: Literal["metadata_assisted", "query_only"] | None = None
    query_text: str = Field(min_length=2, max_length=500)
    purpose: RetrievalPurpose = RetrievalPurpose.KNOWLEDGE_SEARCH
    unit_id: DeepUnitId | None = None
    concept_id: ConceptId | None = None
    gold_concept_ids: tuple[ConceptId, ...] = ()
    knowledge_node_ids: tuple[str, ...] = ()
    disclosure_level: int = Field(ge=1, le=4)
    top_k: Literal[3] = 3
    maximum_context_chars: Literal[3000] = 3_000
    expected_status: Literal["matched", "no_match"]
    target_match_mode: Literal["any_of"] = "any_of"
    required_targets: tuple[RagEvalTarget, ...] = ()
    acceptable_targets: tuple[RagEvalTarget, ...] = ()
    forbidden_sections: tuple[RagSection, ...] = ()
    forbidden_output_fragments: tuple[str, ...] = ()
    query_author_role: str = Field(default="student_team", min_length=3, max_length=80)
    label_reviewer_role: str = Field(
        default="content_design_reviewer",
        min_length=3,
        max_length=80,
    )
    label_basis: Literal["manual_source_section_review"] = (
        "manual_source_section_review"
    )
    rationale_zh: str = Field(min_length=8, max_length=500)

    @model_validator(mode="after")
    def labels_are_consistent(self) -> Self:
        if len(self.knowledge_node_ids) != len(set(self.knowledge_node_ids)):
            raise ValueError("knowledge_node_ids 不能重复")
        required_sources = [target.source_id for target in self.required_targets]
        if len(required_sources) != len(set(required_sources)):
            raise ValueError("required_targets 的 source_id 不能重复")
        acceptable_sources = [target.source_id for target in self.acceptable_targets]
        if len(acceptable_sources) != len(set(acceptable_sources)):
            raise ValueError("acceptable_targets 的 source_id 不能重复")
        if len(self.gold_concept_ids) != len(set(self.gold_concept_ids)):
            raise ValueError("gold_concept_ids 不能重复")
        if len(self.forbidden_sections) != len(set(self.forbidden_sections)):
            raise ValueError("forbidden_sections 不能重复")
        if len(self.forbidden_output_fragments) != len(
            set(self.forbidden_output_fragments)
        ):
            raise ValueError("forbidden_output_fragments 不能重复")
        if any(not _normalize(fragment) for fragment in self.forbidden_output_fragments):
            raise ValueError("forbidden_output_fragments 不能包含空白片段")
        if self.query_author_role == self.label_reviewer_role:
            raise ValueError("查询撰写与标签复核必须使用不同角色")
        inferred_mode = (
            "metadata_assisted"
            if self.concept_id is not None or self.knowledge_node_ids
            else "query_only"
        )
        if self.metadata_mode is not None and self.metadata_mode != inferred_mode:
            raise ValueError("metadata_mode 与查询中实际使用的 concept/node 不一致")
        if self.expected_status == "matched":
            if self.unit_id is None:
                raise ValueError("matched 案例必须标注 unit_id")
            if inferred_mode == "metadata_assisted" and (
                self.concept_id is None or not self.knowledge_node_ids
            ):
                raise ValueError("metadata_assisted 案例必须传入 concept 和知识节点")
            if inferred_mode == "query_only" and not self.gold_concept_ids:
                raise ValueError("query_only matched 案例必须标注 gold_concept_ids")
            if not self.required_targets:
                raise ValueError("matched 案例必须标注 required_targets")
            if not self.acceptable_targets:
                raise ValueError("matched 案例必须标注 acceptable_targets")
            if not self.required_target_pairs.issubset(self.acceptable_target_pairs):
                raise ValueError("严格目标必须属于可接受目标")
            forbidden_pairs = {
                (target.source_id, section)
                for target in self.required_targets
                for section in self.forbidden_sections
            }
            if self.required_target_pairs & forbidden_pairs:
                raise ValueError("严格目标章节不能同时被标为禁止")
        elif any(
            (
                self.unit_id is not None,
                self.concept_id is not None,
                bool(self.knowledge_node_ids),
                bool(self.required_targets),
                bool(self.gold_concept_ids),
                bool(self.acceptable_targets),
            )
        ):
            raise ValueError("no_match 案例不能靠 concept、节点或相关来源标签强制过滤")
        if self.expected_status == "no_match" and self.query_style != "out_of_domain":
            raise ValueError("no_match 案例的 query_style 必须是 out_of_domain")
        if self.expected_status == "matched" and self.query_style == "out_of_domain":
            raise ValueError("matched 案例不能使用 out_of_domain 类型")
        return self

    @property
    def effective_metadata_mode(self) -> Literal["metadata_assisted", "query_only"]:
        if self.concept_id is not None or self.knowledge_node_ids:
            return "metadata_assisted"
        return "query_only"

    @property
    def effective_gold_concept_ids(self) -> tuple[ConceptId, ...]:
        if self.gold_concept_ids:
            return self.gold_concept_ids
        return (self.concept_id,) if self.concept_id is not None else ()

    @property
    def required_target_pairs(self) -> frozenset[tuple[str, RagSection]]:
        return frozenset(
            (source_id, section)
            for target in self.required_targets
            for source_id in (target.source_id,)
            for section in target.sections
        )

    @property
    def acceptable_target_pairs(self) -> frozenset[tuple[str, RagSection]]:
        return frozenset(
            (source_id, section)
            for target in self.acceptable_targets
            for source_id in (target.source_id,)
            for section in target.sections
        )


class RatioMetric(BaseModel):
    """One count-backed metric with an explicit denominator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    numerator: int = Field(ge=0)
    denominator: int = Field(ge=0)
    value: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def value_matches_counts(self) -> Self:
        expected = self.numerator / self.denominator if self.denominator else 0.0
        if not math.isclose(self.value, expected, abs_tol=1e-12):
            raise ValueError("比例指标 value 必须等于 numerator / denominator")
        return self


class RagEvalObservation(BaseModel):
    """One measured retrieval result; failures remain visible."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case: RagEvalCase
    result: RagSearchResult | None = None
    latency_ms: float = Field(ge=0.0)
    replay_stable: bool
    failure_zh: str | None = None

    @model_validator(mode="after")
    def result_and_failure_are_exclusive(self) -> Self:
        if (self.result is None) == (self.failure_zh is None):
            raise ValueError("result 与 failure_zh 必须且只能存在一个")
        return self


class RagEvalSummary(BaseModel):
    """Separately reported G3.2 retrieval, citation, safety, and latency metrics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    retrieval_contract_version: Literal["g3.2-v1"] = RETRIEVAL_CONTRACT_VERSION
    split: Literal["development", "holdout"]
    dataset_fingerprint: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    index_fingerprint: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    retriever_fingerprint: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    evaluator_fingerprint: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    case_count: int = Field(ge=1)
    matched_case_count: int = Field(ge=0)
    no_match_case_count: int = Field(ge=0)
    by_unit: dict[str, int]
    by_query_style: dict[str, int]
    by_metadata_mode: dict[str, int]
    by_purpose: dict[str, int]
    by_disclosure_level: dict[str, int]
    positive_case_coverage_by_source: dict[str, int]
    protected_level_one_case_count: int = Field(ge=0)
    protected_level_one_by_unit: dict[str, int]
    split_independence_checked: bool
    target_recall_at_1: RatioMetric
    target_recall_at_3: RatioMetric
    per_unit_target_recall_at_3: dict[str, RatioMetric]
    query_only_target_recall_at_3: RatioMetric
    source_recall_at_1: RatioMetric
    source_recall_at_3: RatioMetric
    mean_reciprocal_rank: float = Field(ge=0.0, le=1.0)
    relevant_source_precision: RatioMetric
    citation_correctness: RatioMetric
    citation_integrity_accuracy: RatioMetric
    no_result_accuracy: RatioMetric
    disclosure_violation_rate: RatioMetric
    level_one_leakage_rate: RatioMetric
    replay_stability_accuracy: RatioMetric
    evaluation_failure_rate: RatioMetric
    average_latency_ms: float = Field(ge=0.0)
    p95_latency_ms: float = Field(ge=0.0)
    gate_checks: dict[str, bool]
    all_gates_pass: bool

    @model_validator(mode="after")
    def total_and_gate_state_are_consistent(self) -> Self:
        if self.matched_case_count + self.no_match_case_count != self.case_count:
            raise ValueError("matched/no_match 案例数之和必须等于 case_count")
        if self.all_gates_pass != all(self.gate_checks.values()):
            raise ValueError("all_gates_pass 必须等于全部 gate_checks 的合取")
        return self


class RagSourceFreeze(BaseModel):
    """One source entry frozen with an evaluation dataset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    checksum: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class RagEvalFreezeManifest(BaseModel):
    """Fingerprints that must match before a frozen dataset may run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    retrieval_contract_version: Literal["g3.2-v1"]
    split: Literal["development", "holdout"]
    frozen_on: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    case_count: int = Field(ge=1)
    matched_case_count: int = Field(ge=0)
    no_match_case_count: int = Field(ge=0)
    cases_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    index_fingerprint: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    retriever_fingerprint: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    evaluator_fingerprint: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    label_policy_version: str = Field(min_length=3, max_length=100)
    distribution: dict[str, object]
    quality_gates: dict[str, object]
    development_baseline: dict[str, object] | None = None
    source_snapshot: tuple[RagSourceFreeze, ...] = Field(min_length=1)
    holdout_policy: dict[str, object]

    @model_validator(mode="after")
    def counts_and_sources_are_consistent(self) -> Self:
        if self.matched_case_count + self.no_match_case_count != self.case_count:
            raise ValueError("冻结清单的 matched/no_match 计数不一致")
        source_ids = [source.source_id for source in self.source_snapshot]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("冻结清单的 source_id 不能重复")
        return self


def load_rag_eval_cases(
    path: Path,
    *,
    expected_split: Literal["development", "holdout"],
    index: LocalRagIndex,
) -> tuple[RagEvalCase, ...]:
    """Load, validate, de-duplicate, and boundary-audit one JSONL dataset."""

    cases: list[RagEvalCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            case = RagEvalCase.model_validate_json(line)
        except ValidationError as error:
            raise ValueError(f"RAG 评测集第 {line_number} 行无效：{error}") from error
        if case.split != expected_split:
            raise ValueError(
                f"案例 {case.id} 的 split={case.split}，应为 {expected_split}"
            )
        cases.append(case)
    if not cases:
        raise ValueError("RAG 评测集不能为空")

    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("RAG 评测案例 ID 不能重复")
    query_hashes = [_query_contract_sha256(case) for case in cases]
    if len(query_hashes) != len(set(query_hashes)):
        raise ValueError("RAG 评测查询合同规范化后不能重复")
    text_hashes = [_normalized_query_sha256(case.query_text) for case in cases]
    if len(text_hashes) != len(set(text_hashes)):
        raise ValueError("RAG 评测查询文本规范化后不能重复")

    source_ids = set(index.source_ids)
    chunks_by_source: dict[str, list[RagChunk]] = {}
    for chunk in index.chunks:
        chunks_by_source.setdefault(chunk.source_id, []).append(chunk)
    question_bank = load_default_question_bank().questions
    protected_prompts = {_normalize(question.prompt) for question in question_bank}
    protected_ids = {_normalize(question.id) for question in question_bank}
    chunk_texts = {_normalize(chunk.text) for chunk in index.chunks}
    chunk_ids = {_normalize(chunk.chunk_id) for chunk in index.chunks}

    for case in cases:
        normalized_query = _normalize(case.query_text)
        expected_origin = (
            "team_authored" if expected_split == "development" else "independent_judge"
        )
        if case.origin != expected_origin:
            raise ValueError(
                f"案例 {case.id} 的 origin={case.origin}，应为 {expected_origin}"
            )
        if normalized_query in protected_prompts:
            raise ValueError(f"案例 {case.id} 复制了正式题完整题干")
        if any(question_id in normalized_query for question_id in protected_ids):
            raise ValueError(f"案例 {case.id} 包含正式题内部 ID")
        if normalized_query in chunk_texts:
            raise ValueError(f"案例 {case.id} 直接复制了知识卡完整切片")
        if any(source_id in normalized_query for source_id in source_ids):
            raise ValueError(f"案例 {case.id} 查询中不能包含 source_id")
        if any(chunk_id in normalized_query for chunk_id in chunk_ids):
            raise ValueError(f"案例 {case.id} 查询中不能包含 chunk_id")
        if any(
            forbidden in normalized_query
            for forbidden in ("sha256:", "data/rag/", ".yaml")
        ):
            raise ValueError(f"案例 {case.id} 查询中不能包含指纹或文件路径")
        for chunk_text in chunk_texts:
            if _is_near_duplicate(normalized_query, chunk_text):
                raise ValueError(f"案例 {case.id} 与知识卡切片过度相似")
        required_source_ids = {
            target.source_id for target in case.required_targets
        }
        acceptable_source_ids = {
            target.source_id for target in case.acceptable_targets
        }
        unknown_sources = (required_source_ids | acceptable_source_ids) - source_ids
        if unknown_sources:
            raise ValueError(
                f"案例 {case.id} 引用了未知来源：{', '.join(sorted(unknown_sources))}"
            )
        if case.unit_id is not None and not set(case.effective_gold_concept_ids).issubset(
            UNIT_CONCEPTS[case.unit_id]
        ):
            raise ValueError(f"案例 {case.id} 的 concept 与 unit_id 不一致")
        target_source_ids = required_source_ids | acceptable_source_ids
        for source_id in target_source_ids:
            source_chunks = chunks_by_source[source_id]
            if not any(
                chunk.concept_id in case.effective_gold_concept_ids
                for chunk in source_chunks
            ):
                raise ValueError(f"案例 {case.id} 的 concept 与相关来源不一致")
        if case.required_targets and not _has_disclosure_safe_target(
            case.required_targets,
            chunks_by_source,
            disclosure_level=case.disclosure_level,
        ):
            raise ValueError(
                f"案例 {case.id} 没有任何在当前披露级别可用的严格目标"
            )
        if case.acceptable_targets and not _has_disclosure_safe_target(
            case.acceptable_targets,
            chunks_by_source,
            disclosure_level=case.disclosure_level,
        ):
            raise ValueError(
                f"案例 {case.id} 没有任何在当前披露级别可用的可接受目标"
            )
        if case.knowledge_node_ids and not any(
            set(case.knowledge_node_ids) & set(chunk.knowledge_node_ids)
            for source_id in required_source_ids
            for chunk in chunks_by_source[source_id]
        ):
            raise ValueError(f"案例 {case.id} 的知识节点与相关来源没有交集")
    return tuple(cases)


def load_rag_eval_freeze_manifest(path: Path) -> RagEvalFreezeManifest:
    """Load a strict freeze manifest without silently accepting unknown fields."""

    try:
        return RagEvalFreezeManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as error:
        raise ValueError(f"RAG 评测冻结清单无效：{error}") from error


def validate_frozen_dataset(
    cases: tuple[RagEvalCase, ...],
    *,
    index: LocalRagIndex,
    manifest: RagEvalFreezeManifest,
) -> None:
    """Reject stale labels or code before calculating any quality metric."""

    matched_count = sum(case.expected_status == "matched" for case in cases)
    no_match_count = len(cases) - matched_count
    expected_retriever_fingerprint = _file_sha256(
        PROJECT_ROOT / "src" / "probstat_tutor" / "rag" / "retrieval.py"
    )
    expected_evaluator_fingerprint = _file_sha256(Path(__file__))
    actual_sources = {
        (chunk.source_id, chunk.source_version, chunk.source_checksum)
        for chunk in index.chunks
    }
    frozen_sources = {
        (source.source_id, source.version, source.checksum)
        for source in manifest.source_snapshot
    }
    checks = {
        "split": manifest.split == cases[0].split,
        "contract": manifest.retrieval_contract_version == RETRIEVAL_CONTRACT_VERSION,
        "case_count": manifest.case_count == len(cases),
        "matched_case_count": manifest.matched_case_count == matched_count,
        "no_match_case_count": manifest.no_match_case_count == no_match_count,
        "dataset_fingerprint": manifest.cases_sha256 == dataset_fingerprint(cases),
        "index_fingerprint": manifest.index_fingerprint == index.index_fingerprint,
        "retriever_fingerprint": (
            manifest.retriever_fingerprint == expected_retriever_fingerprint
        ),
        "evaluator_fingerprint": (
            manifest.evaluator_fingerprint == expected_evaluator_fingerprint
        ),
        "source_snapshot": frozen_sources == actual_sources,
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    if failed_checks:
        raise ValueError(
            "RAG 评测冻结已失效，必须重新审核标签并冻结："
            + "、".join(failed_checks)
        )


def validate_split_independence(
    development_cases: tuple[RagEvalCase, ...],
    holdout_cases: tuple[RagEvalCase, ...],
) -> None:
    """Check cross-split family, exact-query, contract, and near-duplicate isolation."""

    if any(case.split != "development" for case in development_cases) or any(
        case.split != "holdout" for case in holdout_cases
    ):
        raise ValueError("跨 split 独立性检查收到了错误的数据集类型")
    development_families = {case.query_family for case in development_cases}
    holdout_families = {case.query_family for case in holdout_cases}
    if development_families & holdout_families:
        raise ValueError("holdout 与 development 存在重复 query_family")
    development_text_hashes = {
        _normalized_query_sha256(case.query_text) for case in development_cases
    }
    holdout_text_hashes = {
        _normalized_query_sha256(case.query_text) for case in holdout_cases
    }
    if development_text_hashes & holdout_text_hashes:
        raise ValueError("holdout 与 development 存在重复规范化查询")
    development_contract_hashes = {
        _query_contract_sha256(case) for case in development_cases
    }
    holdout_contract_hashes = {_query_contract_sha256(case) for case in holdout_cases}
    if development_contract_hashes & holdout_contract_hashes:
        raise ValueError("holdout 与 development 存在重复查询合同")
    if any(
        _is_near_duplicate(
            _normalize(development.query_text),
            _normalize(holdout.query_text),
        )
        for development in development_cases
        for holdout in holdout_cases
    ):
        raise ValueError("holdout 与 development 存在高度近重复查询")


def dataset_fingerprint(cases: tuple[RagEvalCase, ...]) -> str:
    """Return a stable fingerprint over the complete labelled dataset."""

    payload = [
        case.model_dump(mode="json") for case in sorted(cases, key=lambda item: item.id)
    ]
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"sha256:{hashlib.sha256(serialized.encode('utf-8')).hexdigest()}"


def _has_disclosure_safe_target(
    targets: tuple[RagEvalTarget, ...],
    chunks_by_source: dict[str, list[RagChunk]],
    *,
    disclosure_level: int,
) -> bool:
    return any(
        chunk.section in target.sections and chunk.min_hint_level <= disclosure_level
        for target in targets
        for chunk in chunks_by_source[target.source_id]
    )


def evaluate_rag_cases(
    cases: tuple[RagEvalCase, ...],
    *,
    index: LocalRagIndex,
    split_independence_checked: bool = False,
) -> tuple[tuple[RagEvalObservation, ...], RagEvalSummary]:
    """Run deterministic retrieval and calculate all metrics without a model."""

    observations: list[RagEvalObservation] = []
    for case in cases:
        query = RagQuery(
            text=case.query_text,
            concept_id=case.concept_id,
            knowledge_node_ids=case.knowledge_node_ids,
            purpose=case.purpose,
            disclosure_level=case.disclosure_level,
            top_k=case.top_k,
            maximum_context_chars=case.maximum_context_chars,
        )
        replay_results: list[RagSearchResult] = []
        replay_latencies: list[float] = []
        failed = False
        for _replay in range(REPLAY_COUNT):
            started = time.perf_counter()
            try:
                replay_results.append(index.search(query))
            except Exception:
                failed = True
            finally:
                replay_latencies.append((time.perf_counter() - started) * 1_000)
            if failed:
                break

        if failed or len(replay_results) != REPLAY_COUNT:
            observations.append(
                RagEvalObservation(
                    case=case,
                    latency_ms=statistics.fmean(replay_latencies),
                    replay_stable=False,
                    failure_zh="检索评测流程失败；未记录内部异常内容。",
                )
            )
            continue

        signatures = tuple(_result_signature(result) for result in replay_results)
        observations.append(
            RagEvalObservation(
                case=case,
                result=replay_results[0],
                latency_ms=statistics.fmean(replay_latencies),
                replay_stable=len(set(signatures)) == 1,
            )
        )

    frozen_observations = tuple(observations)
    summary = _summarize(
        frozen_observations,
        index=index,
        split_independence_checked=split_independence_checked,
    )
    return frozen_observations, summary


def _summarize(
    observations: tuple[RagEvalObservation, ...],
    *,
    index: LocalRagIndex,
    split_independence_checked: bool,
) -> RagEvalSummary:
    matched_observations = [
        item for item in observations if item.case.expected_status == "matched"
    ]
    no_match_observations = [
        item for item in observations if item.case.expected_status == "no_match"
    ]
    target_recall_one = 0
    target_recall_three = 0
    source_recall_one = 0
    source_recall_three = 0
    reciprocal_ranks: list[float] = []
    relevant_hits = 0
    total_matched_hits = 0
    unit_target_successes: Counter[str] = Counter()
    query_only_successes = 0
    query_only_matched = 0
    citation_correct = 0
    citation_total = 0
    citation_integral = 0
    no_result_correct = 0
    disclosure_violations = 0
    level_one_leakages = 0
    level_one_cases = 0
    stable_replays = 0
    failures = 0
    chunks_by_id = {chunk.chunk_id: chunk for chunk in index.chunks}

    for observation in observations:
        stable_replays += observation.replay_stable
        result = observation.result
        if result is None:
            failures += 1
            if observation.case.disclosure_level == 1:
                level_one_cases += 1
            continue
        case_disclosure_violation = _case_has_disclosure_violation(
            observation.case,
            result,
        )
        disclosure_violations += case_disclosure_violation
        if observation.case.disclosure_level == 1:
            level_one_cases += 1
            level_one_leakages += case_disclosure_violation
        if observation.case.expected_status == "no_match":
            no_result_correct += (
                result.status == KnowledgeContextStatus.NO_MATCH and not result.hits
            )
        else:
            case = observation.case
            required_pairs = case.required_target_pairs
            acceptable_pairs = case.acceptable_target_pairs
            relevant_sources = {
                target.source_id for target in case.required_targets
            }
            ranked_pairs = [_hit_pair(hit) for hit in result.hits]
            target_ranks = [
                rank
                for rank, pair in enumerate(ranked_pairs, start=1)
                if pair in required_pairs
            ]
            source_ranks = [
                rank
                for rank, pair in enumerate(ranked_pairs, start=1)
                if pair[0] in relevant_sources
            ]
            target_recall_one += bool(target_ranks and target_ranks[0] == 1)
            target_recall_three += bool(target_ranks and target_ranks[0] <= 3)
            source_recall_one += bool(source_ranks and source_ranks[0] == 1)
            source_recall_three += bool(source_ranks and source_ranks[0] <= 3)
            reciprocal_ranks.append(1.0 / target_ranks[0] if target_ranks else 0.0)
            if target_ranks and target_ranks[0] <= 3 and case.unit_id is not None:
                unit_target_successes[case.unit_id.value] += 1
            if case.effective_metadata_mode == "query_only":
                query_only_matched += 1
                query_only_successes += bool(target_ranks and target_ranks[0] <= 3)
            ranked_sources = [pair[0] for pair in ranked_pairs]
            relevant_hits += sum(source_id in relevant_sources for source_id in ranked_sources)
            total_matched_hits += len(ranked_sources)
            for hit in result.hits:
                citation_total += 1
                technically_integral = _citation_is_integral(hit, chunks_by_id)
                citation_integral += technically_integral
                citation_correct += (
                    technically_integral
                    and _hit_pair(hit) in acceptable_pairs
                    and not _hit_has_disclosure_violation(case, hit)
                )

    latencies = [item.latency_ms for item in observations]
    unit_counts = Counter(
        item.case.unit_id.value
        for item in matched_observations
        if item.case.unit_id is not None
    )
    query_style_counts = Counter(item.case.query_style for item in observations)
    metadata_mode_counts = Counter(
        item.case.effective_metadata_mode for item in observations
    )
    purpose_counts = Counter(item.case.purpose.value for item in observations)
    disclosure_counts = Counter(
        str(item.case.disclosure_level) for item in observations
    )
    source_case_counts: Counter[str] = Counter()
    for item in matched_observations:
        required_sources = {
            target.source_id for target in item.case.required_targets
        }
        if len(required_sources) == 1:
            source_case_counts.update(required_sources)
    protected_level_one_cases = [
        item
        for item in matched_observations
        if item.case.disclosure_level == 1
        and (item.case.forbidden_sections or item.case.forbidden_output_fragments)
    ]
    protected_level_one_by_unit = Counter(
        item.case.unit_id.value
        for item in protected_level_one_cases
        if item.case.unit_id is not None
    )
    target_recall_at_1 = _ratio(target_recall_one, len(matched_observations))
    target_recall_at_3 = _ratio(target_recall_three, len(matched_observations))
    source_recall_at_1 = _ratio(source_recall_one, len(matched_observations))
    source_recall_at_3 = _ratio(source_recall_three, len(matched_observations))
    relevant_source_precision = _ratio(relevant_hits, total_matched_hits)
    citation_correctness = _ratio(citation_correct, citation_total)
    citation_accuracy = _ratio(citation_integral, citation_total)
    no_result_accuracy = _ratio(no_result_correct, len(no_match_observations))
    disclosure_rate = _ratio(disclosure_violations, len(observations))
    level_one_rate = _ratio(level_one_leakages, level_one_cases)
    replay_accuracy = _ratio(stable_replays, len(observations))
    failure_rate = _ratio(failures, len(observations))
    mrr = statistics.fmean(reciprocal_ranks) if reciprocal_ranks else 0.0
    p95 = _percentile(latencies, 0.95)
    split = observations[0].case.split
    minimum_cases = (
        MINIMUM_DEVELOPMENT_CASES
        if split == "development"
        else MINIMUM_HOLDOUT_CASES
    )
    minimum_matched_per_unit = (
        MINIMUM_DEVELOPMENT_MATCHED_CASES_PER_UNIT
        if split == "development"
        else MINIMUM_HOLDOUT_MATCHED_CASES_PER_UNIT
    )
    per_unit_target_recall = {
        unit.value: _ratio(unit_target_successes[unit.value], unit_counts[unit.value])
        for unit in DeepUnitId
    }
    query_only_target_recall = _ratio(query_only_successes, query_only_matched)
    covered_concepts = {
        concept
        for item in matched_observations
        for concept in item.case.effective_gold_concept_ids
    }

    gate_checks = _build_gate_checks(
        split=split,
        case_count=len(observations),
        minimum_cases=minimum_cases,
        no_match_count=len(no_match_observations),
        unit_counts=unit_counts,
        minimum_matched_per_unit=minimum_matched_per_unit,
        covered_concepts=covered_concepts,
        source_case_counts=source_case_counts,
        source_ids=index.source_ids,
        query_only_matched=query_only_matched,
        purpose_counts=purpose_counts,
        protected_level_one_count=len(protected_level_one_cases),
        protected_level_one_by_unit=protected_level_one_by_unit,
        target_recall_at_1=target_recall_at_1,
        target_recall_at_3=target_recall_at_3,
        per_unit_target_recall=per_unit_target_recall,
        query_only_target_recall=query_only_target_recall,
        mean_reciprocal_rank=mrr,
        citation_correctness=citation_correctness,
        citation_integrity=citation_accuracy,
        no_result_accuracy=no_result_accuracy,
        disclosure_rate=disclosure_rate,
        level_one_rate=level_one_rate,
        replay_accuracy=replay_accuracy,
        failure_rate=failure_rate,
        split_independence_checked=split_independence_checked,
    )
    return RagEvalSummary(
        split=split,
        dataset_fingerprint=dataset_fingerprint(
            tuple(item.case for item in observations)
        ),
        index_fingerprint=index.index_fingerprint,
        retriever_fingerprint=_file_sha256(
            PROJECT_ROOT / "src" / "probstat_tutor" / "rag" / "retrieval.py"
        ),
        evaluator_fingerprint=_file_sha256(Path(__file__)),
        case_count=len(observations),
        matched_case_count=len(matched_observations),
        no_match_case_count=len(no_match_observations),
        by_unit=dict(sorted(unit_counts.items())),
        by_query_style=dict(sorted(query_style_counts.items())),
        by_metadata_mode=dict(sorted(metadata_mode_counts.items())),
        by_purpose=dict(sorted(purpose_counts.items())),
        by_disclosure_level=dict(sorted(disclosure_counts.items())),
        positive_case_coverage_by_source={
            source_id: source_case_counts[source_id]
            for source_id in index.source_ids
        },
        protected_level_one_case_count=len(protected_level_one_cases),
        protected_level_one_by_unit={
            unit.value: protected_level_one_by_unit[unit.value]
            for unit in DeepUnitId
        },
        split_independence_checked=split_independence_checked,
        target_recall_at_1=target_recall_at_1,
        target_recall_at_3=target_recall_at_3,
        per_unit_target_recall_at_3=per_unit_target_recall,
        query_only_target_recall_at_3=query_only_target_recall,
        source_recall_at_1=source_recall_at_1,
        source_recall_at_3=source_recall_at_3,
        mean_reciprocal_rank=mrr,
        relevant_source_precision=relevant_source_precision,
        citation_correctness=citation_correctness,
        citation_integrity_accuracy=citation_accuracy,
        no_result_accuracy=no_result_accuracy,
        disclosure_violation_rate=disclosure_rate,
        level_one_leakage_rate=level_one_rate,
        replay_stability_accuracy=replay_accuracy,
        evaluation_failure_rate=failure_rate,
        average_latency_ms=statistics.fmean(latencies),
        p95_latency_ms=p95,
        gate_checks=gate_checks,
        all_gates_pass=all(gate_checks.values()),
    )


def _build_gate_checks(
    *,
    split: Literal["development", "holdout"],
    case_count: int,
    minimum_cases: int,
    no_match_count: int,
    unit_counts: Counter[str],
    minimum_matched_per_unit: int,
    covered_concepts: set[ConceptId],
    source_case_counts: Counter[str],
    source_ids: tuple[str, ...],
    query_only_matched: int,
    purpose_counts: Counter[str],
    protected_level_one_count: int,
    protected_level_one_by_unit: Counter[str],
    target_recall_at_1: RatioMetric,
    target_recall_at_3: RatioMetric,
    per_unit_target_recall: dict[str, RatioMetric],
    query_only_target_recall: RatioMetric,
    mean_reciprocal_rank: float,
    citation_correctness: RatioMetric,
    citation_integrity: RatioMetric,
    no_result_accuracy: RatioMetric,
    disclosure_rate: RatioMetric,
    level_one_rate: RatioMetric,
    replay_accuracy: RatioMetric,
    failure_rate: RatioMetric,
    split_independence_checked: bool,
) -> dict[str, bool]:
    checks = {
        "minimum_case_count": case_count >= minimum_cases,
        "minimum_no_match_cases": no_match_count >= MINIMUM_NO_MATCH_CASES,
        "unit_coverage": all(
            unit_counts[unit.value] >= minimum_matched_per_unit
            for unit in DeepUnitId
        ),
        "concept_coverage": covered_concepts == set(ConceptId),
        "source_positive_coverage": all(
            source_case_counts[source_id] >= MINIMUM_POSITIVE_CASES_PER_SOURCE
            for source_id in source_ids
        ),
        "minimum_query_only_matched_cases": (
            query_only_matched >= MINIMUM_QUERY_ONLY_MATCHED_CASES
        ),
        "purpose_coverage": all(
            purpose_counts[purpose.value] >= 8 for purpose in RetrievalPurpose
        ),
        "protected_level_one_coverage": (
            protected_level_one_count >= MINIMUM_PROTECTED_LEVEL_ONE_CASES
            and all(
                protected_level_one_by_unit[unit.value] >= 1 for unit in DeepUnitId
            )
        ),
        "target_recall_at_1_ge_0_75": target_recall_at_1.value >= 0.75,
        "target_recall_at_3_ge_0_90": target_recall_at_3.value >= 0.90,
        "per_unit_target_recall_at_3_ge_0_80": all(
            metric.value >= 0.80 for metric in per_unit_target_recall.values()
        ),
        "query_only_target_recall_at_3_ge_0_75": (
            query_only_target_recall.value >= 0.75
        ),
        "mean_reciprocal_rank_ge_0_80": mean_reciprocal_rank >= 0.80,
        "citation_correctness_ge_0_95": citation_correctness.value >= 0.95,
        "citation_integrity_eq_1": citation_integrity.value == 1.0,
        "no_result_accuracy_eq_1": no_result_accuracy.value == 1.0,
        "disclosure_violation_rate_eq_0": disclosure_rate.value == 0.0,
        "level_one_leakage_rate_eq_0": level_one_rate.value == 0.0,
        "replay_stability_eq_1": replay_accuracy.value == 1.0,
        "evaluation_failure_rate_eq_0": failure_rate.value == 0.0,
    }
    if split == "holdout":
        checks["split_independence_checked"] = split_independence_checked
    return checks


def _citation_is_integral(
    hit: RagSearchHit,
    chunks_by_id: dict[str, RagChunk],
) -> bool:
    citation = hit.citation
    chunk = chunks_by_id.get(citation.chunk_id)
    if chunk is None:
        return False
    expected_quote = chunk.text if len(chunk.text) <= 600 else f"{chunk.text[:599]}…"
    return all(
        (
            citation.citation_id == f"R{hit.rank}",
            citation.source_id == chunk.source_id,
            citation.source_version == chunk.source_version,
            citation.source_title == chunk.source_title,
            citation.section == chunk.section.value,
            citation.content_checksum == chunk.content_checksum,
            citation.source_checksum == chunk.source_checksum,
            citation.min_hint_level == chunk.min_hint_level,
            citation.review_status == chunk.review_status,
            hit.content == chunk.text,
            citation.quote == expected_quote,
        )
    )


def _hit_pair(hit: RagSearchHit) -> tuple[str, RagSection]:
    return hit.citation.source_id, RagSection(hit.citation.section)


def _hit_has_disclosure_violation(case: RagEvalCase, hit: RagSearchHit) -> bool:
    normalized_content = _normalize(hit.content)
    return any(
        (
            hit.citation.min_hint_level > case.disclosure_level,
            infer_minimum_disclosure_level(hit.content) > case.disclosure_level,
            RagSection(hit.citation.section) in case.forbidden_sections,
            any(
                _normalize(fragment) in normalized_content
                for fragment in case.forbidden_output_fragments
            ),
        )
    )


def _case_has_disclosure_violation(
    case: RagEvalCase,
    result: RagSearchResult,
) -> bool:
    return any(_hit_has_disclosure_violation(case, hit) for hit in result.hits)


def _result_signature(result: RagSearchResult) -> str:
    """Hash every deterministic result field while intentionally excluding time."""

    serialized = json.dumps(
        result.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _ratio(numerator: int, denominator: int) -> RatioMetric:
    return RatioMetric(
        numerator=numerator,
        denominator=denominator,
        value=numerator / denominator if denominator else 0.0,
    )


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = max(0, math.ceil(len(ordered) * fraction) - 1)
    return ordered[position]


def _normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split()).strip()


def _normalized_query_sha256(value: str) -> str:
    return hashlib.sha256(_normalize(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _query_contract_sha256(case: RagEvalCase) -> str:
    payload = {
        "query_text": _normalize(case.query_text),
        "concept_id": case.concept_id,
        "knowledge_node_ids": sorted(case.knowledge_node_ids),
        "purpose": case.purpose,
        "disclosure_level": case.disclosure_level,
        "top_k": case.top_k,
        "maximum_context_chars": case.maximum_context_chars,
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _is_near_duplicate(left: str, right: str) -> bool:
    left_compact = "".join(character for character in left if character.isalnum())
    right_compact = "".join(character for character in right if character.isalnum())
    if len(left_compact) < 8 or len(right_compact) < 8:
        return False
    if left_compact in right_compact or right_compact in left_compact:
        return True
    left_grams = _character_ngrams(left_compact, width=4)
    right_grams = _character_ngrams(right_compact, width=4)
    union = left_grams | right_grams
    if not union:
        return False
    return len(left_grams & right_grams) / len(union) >= MAXIMUM_NEAR_DUPLICATE_JACCARD


def _character_ngrams(value: str, *, width: int) -> frozenset[str]:
    return frozenset(
        value[index : index + width]
        for index in range(max(0, len(value) - width + 1))
    )


def run_dataset(
    path: Path = DEVELOPMENT_DATASET_PATH,
    *,
    expected_split: Literal["development", "holdout"] = "development",
    freeze_manifest_path: Path | None = None,
) -> RagEvalSummary:
    """Validate frozen code/data, then run one local dataset and return aggregates."""

    index = build_local_rag_index(PROJECT_ROOT)
    cases = load_rag_eval_cases(path, expected_split=expected_split, index=index)
    manifest_path = freeze_manifest_path
    if manifest_path is None:
        if expected_split == "holdout":
            raise ValueError("holdout 运行必须显式提供独立评委冻结清单")
        manifest_path = DEVELOPMENT_MANIFEST_PATH
    manifest = load_rag_eval_freeze_manifest(manifest_path)
    validate_frozen_dataset(cases, index=index, manifest=manifest)

    split_independence_checked = False
    if expected_split == "holdout":
        development_cases = load_rag_eval_cases(
            DEVELOPMENT_DATASET_PATH,
            expected_split="development",
            index=index,
        )
        development_manifest = load_rag_eval_freeze_manifest(
            DEVELOPMENT_MANIFEST_PATH
        )
        validate_frozen_dataset(
            development_cases,
            index=index,
            manifest=development_manifest,
        )
        validate_split_independence(development_cases, cases)
        split_independence_checked = True

    _observations, summary = evaluate_rag_cases(
        cases,
        index=index,
        split_independence_checked=split_independence_checked,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="运行离线本地 RAG 检索评测")
    parser.add_argument("--dataset", type=Path, default=DEVELOPMENT_DATASET_PATH)
    parser.add_argument(
        "--split",
        choices=("development", "holdout"),
        default="development",
    )
    parser.add_argument(
        "--freeze-manifest",
        type=Path,
        help="冻结指纹清单；development 默认使用仓库清单",
    )
    args = parser.parse_args()
    summary = run_dataset(
        args.dataset,
        expected_split=args.split,
        freeze_manifest_path=args.freeze_manifest,
    )
    print(summary.model_dump_json(indent=2))
    if not summary.all_gates_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
