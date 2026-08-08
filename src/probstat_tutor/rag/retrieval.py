"""Deterministic, dependency-free local retrieval over reviewed RAG sources."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from probstat_tutor.rag.loader import find_source_instruction_attacks, load_rag_source
from probstat_tutor.rag.schemas import (
    AllowedUsage,
    AnswerLeakageRisk,
    SourceLicense,
    SourceType,
    load_rag_manifest,
)
from probstat_tutor.rag.source_schemas import LoadedRagSource
from probstat_tutor.schemas import (
    ConceptId,
    ContentReviewStatus,
    KnowledgeCitation,
    KnowledgeContextStatus,
)

MAX_QUERY_LENGTH = 1_000
MAX_RETRIEVAL_RESULTS = 5
DEFAULT_TOP_K = 3
DEFAULT_MAX_CONTEXT_CHARS = 3_000
MAX_CONTEXT_CHARS = 5_000
MAX_HITS_PER_SOURCE = 2
MAX_NODE_SCOPED_HITS_PER_SOURCE = 3
MIN_RETRIEVAL_SCORE = 0.75
# Unscoped Chinese queries can share accidental two-character windows with an
# unrelated course card. The development negatives showed a safe lexical margin
# for that path, while metadata-scoped and explicit ASCII-token queries retain
# the low base threshold used by local API and context-budget tests.
MIN_UNSCOPED_CHINESE_RETRIEVAL_SCORE = 10.0
EXPLICIT_SECTION_INTENT_BONUS = 50.0
SPECIFIC_SECTION_INTENT_BONUS = 12.0
WEAK_SECTION_INTENT_BONUS = 3.0
NO_MATCH_MESSAGE_ZH = (
    "当前原创知识库没有找到足够相关且允许检索的内容。"
    "系统不会编造出处；请改用更具体的统计术语，或继续使用确定性提示。"
)
INDEX_UNAVAILABLE_MESSAGE_ZH = (
    "本地知识索引暂时不可用。本次仍使用确定性判题与分级提示；"
    "系统不会编造来源或中断学习状态更新。"
)
_TOKEN_PATTERN = re.compile(
    r"[a-z][a-z0-9_.-]*|\d+(?:\.\d+)?%?|[\u0370-\u03ff]+|[\u4e00-\u9fff]+"
)
_COMPACT_SEQUENCE_PATTERN = re.compile(r"[a-z0-9\u0370-\u03ff\u4e00-\u9fff]+")
_SINGLE_TOKEN_STOPWORDS = frozenset(
    {"的", "了", "和", "与", "及", "在", "是", "为", "对", "将", "把", "或", "而"}
)
_CALL_EXPRESSION_PATTERN = re.compile(
    r"(?<![\w.])[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*"
    r"\s*\([^()\n]{0,300}\)"
)
_SUBSCRIPT_CALL_CHAIN_PATTERN = re.compile(
    r"(?<![\w.])[A-Za-z_][A-Za-z0-9_]*"
    r"(?:\s*\[[^\]\n]{1,120}\]|\.[A-Za-z_][A-Za-z0-9_]*)+\s*\("
)
_SUBSCRIPT_EXPRESSION_PATTERN = re.compile(
    r"(?<![\w.])[A-Za-z_][A-Za-z0-9_]*\s*\[[^\]\n]{1,120}\]"
)
_QUALIFIED_IDENTIFIER_PATTERN = re.compile(
    r"(?<![\w.])[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*"
)
_SNAKE_CASE_IDENTIFIER_PATTERN = re.compile(
    r"(?<![\w.])[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+(?![\w.])"
)
_INLINE_CODE_IDENTIFIER_PATTERN = re.compile(
    r"`\s*[A-Za-z_][A-Za-z0-9_.]*\s*`"
)
_MATH_MARKER_PATTERN = re.compile(r"[=≤≥≈≠∑√∩∪²³×÷±∞]|->|=>")
_BINARY_OPERATOR_PATTERN = re.compile(
    r"(?:[A-Za-z0-9\u0370-\u03ff)\]])\s*[+*/^]\s*"
    r"(?:[A-Za-z0-9\u0370-\u03ff(√])"
)
_MINUS_EXPRESSION_PATTERN = re.compile(
    r"(?:[A-Za-z\u0370-\u03ff]\s*-\s*\d|\d\s*-\s*[A-Za-z\u0370-\u03ff])"
)
_LATEX_EXPRESSION_PATTERN = re.compile(
    r"\\(?:frac|sqrt|sum|prod|int|left|right|begin|end)\b"
)
_FORMULA_MEANING_PATTERN = re.compile(
    r"(?:公式|表达式|列式|分子|分母|平方根|开平方|临界值|标准化)"
    r"|\b(?:formula|expression|numerator|denominator|square\s+root|critical\s+value)\b"
)
_COURSE_SPECIFIC_ANCHORS = (
    "均值",
    "中位数",
    "方差",
    "标准差",
    "四分位",
    "分位数",
    "条件概率",
    "联合概率",
    "独立事件",
    "随机模拟",
    "蒙特卡洛",
    "伯努利",
    "二项分布",
    "正态分布",
    "概率密度",
    "分布函数",
    "协方差",
    "相关系数",
    "抽样分布",
    "中心极限定理",
    "标准误",
    "点估计",
    "置信区间",
    "原假设",
    "备择假设",
    "假设检验",
    "p值",
    "p 值",
    "显著性水平",
    "一类错误",
    "二类错误",
    "统计功效",
    "数据清洗",
    "缺失值",
    "异常值",
    "重复记录",
    "arithmetic mean",
    "sample mean",
    "pandas mean",
    "pandas groupby",
    "groupby mean",
    "median",
    "variance",
    "standard deviation",
    "conditional probability",
    "joint probability",
    "monte carlo",
    "bernoulli",
    "binomial",
    "binom",
    "normal distribution",
    "probability density",
    "cdf",
    "pmf",
    "covariance",
    "correlation coefficient",
    "sampling distribution",
    "central limit theorem",
    "standard error",
    "point estimate",
    "confidence interval",
    "null hypothesis",
    "alternative hypothesis",
    "hypothesis test",
    "p-value",
    "significance level",
    "type i error",
    "type ii error",
    "statistical power",
    "data cleaning",
    "missing value",
    "outlier",
    "duplicate record",
    "quantile",
)
_COURSE_GENERAL_ANCHORS = (
    "概率",
    "统计",
    "样本",
    "总体",
    "估计",
    "区间",
    "检验",
    "分布",
    "随机",
    "相关",
    "独立",
    "偏差",
    "频率",
    "比例",
    "数据质量",
    "实验",
    "效应",
    "置信",
    "显著",
    "probability",
    "statistics",
    "sample",
    "population",
    "estimate",
    "interval",
    "distribution",
    "random",
    "association",
    "independent",
    "bias",
    "frequency",
    "proportion",
    "experiment",
    "effect",
    "mean",
)


class RagSection(StrEnum):
    """Stable sections from the structured course-source contract."""

    LEARNING_OBJECTIVES = "learning_objectives"
    PREREQUISITE_KNOWLEDGE = "prerequisite_knowledge"
    CONCEPT_EXPLANATION = "concept_explanation"
    FORMULA_MEANING = "formula_meaning"
    FORMULA_EXPRESSION = "formula_expression"
    PYTHON_CONNECTION = "python_connection"
    DATA_INTERPRETATION_GUIDANCE = "data_interpretation_guidance"
    COMMON_MISCONCEPTIONS = "common_misconceptions"
    REFLECTIVE_QUESTIONS = "reflective_questions"
    SUMMARY = "summary"


def infer_minimum_disclosure_level(text: str) -> int:
    """Raise formula, code, and callable expressions to at least level three."""

    normalized = unicodedata.normalize("NFKC", text)
    expression_text = re.sub(r"\bA\s*/\s*B\b", "AB", normalized, flags=re.IGNORECASE)
    if (
        _CALL_EXPRESSION_PATTERN.search(expression_text)
        or _SUBSCRIPT_CALL_CHAIN_PATTERN.search(expression_text)
        or _SUBSCRIPT_EXPRESSION_PATTERN.search(expression_text)
        or _QUALIFIED_IDENTIFIER_PATTERN.search(expression_text)
        or _SNAKE_CASE_IDENTIFIER_PATTERN.search(expression_text)
        or _INLINE_CODE_IDENTIFIER_PATTERN.search(expression_text)
        or _MATH_MARKER_PATTERN.search(expression_text)
        or _BINARY_OPERATOR_PATTERN.search(expression_text)
        or _MINUS_EXPRESSION_PATTERN.search(expression_text)
        or _LATEX_EXPRESSION_PATTERN.search(expression_text)
    ):
        return 3
    if _FORMULA_MEANING_PATTERN.search(normalized.casefold()):
        return 2
    return 1


class RetrievalPurpose(StrEnum):
    """Why the application is requesting local course context."""

    DIAGNOSTIC = "diagnostic"
    HINT = "hint"
    KNOWLEDGE_SEARCH = "knowledge_search"


class RagIndexBuildErrorCode(StrEnum):
    """Stable failures added by indexing after source loading succeeds."""

    SOURCE_NOT_ELIGIBLE = "source_not_eligible"
    REVIEW_STATUS_INVALID = "review_status_invalid"
    CHUNK_SCHEMA_INVALID = "chunk_schema_invalid"
    NO_CHUNKS = "no_chunks"


class RagIndexBuildError(ValueError):
    """Beginner-readable indexing failure with a machine-readable code."""

    def __init__(self, code: RagIndexBuildErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


class RagChunk(BaseModel):
    """One stable, independently citable piece of a reviewed source."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str = Field(pattern=r"^[a-z][a-z0-9_.:@-]*$")
    source_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    source_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    source_title: str = Field(min_length=1, max_length=200)
    concept_id: ConceptId
    knowledge_node_ids: tuple[str, ...] = ()
    section: RagSection
    position: int = Field(ge=0)
    subposition: int = Field(ge=0)
    text: str = Field(min_length=1, max_length=5_000)
    content_checksum: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    source_checksum: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    source_type: SourceType
    license: SourceLicense
    answer_leakage_risk: AnswerLeakageRisk
    allowed_to_quote: bool
    relative_path: str = Field(pattern=r"^data/rag/sources/[a-z0-9_]+\.yaml$")
    min_hint_level: int = Field(ge=1, le=4)
    review_status: ContentReviewStatus

    @field_validator("knowledge_node_ids")
    @classmethod
    def chunk_nodes_are_valid(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("切片 knowledge_node_ids 不能重复")
        if any(not re.fullmatch(r"[a-z][a-z0-9_]*", node_id) for node_id in value):
            raise ValueError("切片 knowledge_node_ids 格式不正确")
        return value


class ExcludedRagSource(BaseModel):
    """One valid manifest source excluded by retrieval policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    rejection_codes: tuple[str, ...] = Field(min_length=1)
    rejection_reasons: tuple[str, ...] = Field(min_length=1)


class RagQuery(BaseModel):
    """Bounded local retrieval request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1, max_length=MAX_QUERY_LENGTH)
    concept_id: ConceptId | None = None
    knowledge_node_ids: tuple[str, ...] = ()
    purpose: RetrievalPurpose = RetrievalPurpose.KNOWLEDGE_SEARCH
    disclosure_level: int = Field(default=1, ge=1, le=4)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_RETRIEVAL_RESULTS)
    maximum_context_chars: int = Field(
        default=DEFAULT_MAX_CONTEXT_CHARS,
        ge=500,
        le=MAX_CONTEXT_CHARS,
    )

    @field_validator("text")
    @classmethod
    def query_contains_searchable_text(cls, value: str) -> str:
        normalized = _normalize_text(value)
        if not normalized or not any(_is_strong_term(term) for term in _tokenize(value)):
            raise ValueError("检索问题不能只包含空白或标点")
        return value.strip()

    @field_validator("knowledge_node_ids")
    @classmethod
    def knowledge_nodes_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("knowledge_node_ids 不能重复")
        return value


class RagSearchHit(BaseModel):
    """One scored chunk and its exact auditable citation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: int = Field(ge=1, le=MAX_RETRIEVAL_RESULTS)
    score: float = Field(gt=0.0)
    matched_terms: tuple[str, ...] = Field(min_length=1)
    content: str = Field(min_length=1, max_length=5_000)
    citation: KnowledgeCitation

    @model_validator(mode="after")
    def citation_matches_hit(self) -> Self:
        expected_checksum = hashlib.sha256(self.content.encode("utf-8")).hexdigest()
        if self.citation.citation_id != f"R{self.rank}":
            raise ValueError("引用编号必须与命中排名一致")
        if self.citation.content_checksum != f"sha256:{expected_checksum}":
            raise ValueError("引用内容 checksum 与命中文本不一致")
        quote = self.citation.quote
        if quote is not None and quote.rstrip("…") not in self.content:
            raise ValueError("引用摘录必须来自命中文本")
        return self


class RagSearchResult(BaseModel):
    """Matched chunks or an explicit no-result degradation, never invented text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: RagQuery
    status: KnowledgeContextStatus
    hits: tuple[RagSearchHit, ...] = ()
    message_zh: str = Field(min_length=1)
    query_fingerprint: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    index_fingerprint: str | None = Field(
        default=None,
        pattern=r"^sha256:[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def status_and_hits_are_consistent(self) -> Self:
        if self.status == KnowledgeContextStatus.MATCHED and not self.hits:
            raise ValueError("matched 检索结果必须包含至少一个命中")
        if self.status != KnowledgeContextStatus.MATCHED and self.hits:
            raise ValueError("非 matched 检索结果不能包含命中")
        if self.status == KnowledgeContextStatus.MATCHED and self.index_fingerprint is None:
            raise ValueError("matched 检索结果必须包含索引指纹")
        if self.status == KnowledgeContextStatus.NO_MATCH and self.index_fingerprint is None:
            raise ValueError("no_match 检索结果必须包含索引指纹")
        if self.hits:
            expected_ranks = tuple(range(1, len(self.hits) + 1))
            if tuple(hit.rank for hit in self.hits) != expected_ranks:
                raise ValueError("检索命中排名必须从 1 连续递增")
            if len(self.hits) > self.query.top_k:
                raise ValueError("检索命中数不能超过 query.top_k")
            if len(self.render_context_for_model()) > self.query.maximum_context_chars:
                raise ValueError("渲染后的检索上下文超过查询预算")
            per_source = Counter(hit.citation.source_id for hit in self.hits)
            maximum_per_source = (
                min(self.query.top_k, MAX_NODE_SCOPED_HITS_PER_SOURCE)
                if self.query.knowledge_node_ids
                else MAX_HITS_PER_SOURCE
            )
            if max(per_source.values()) > maximum_per_source:
                raise ValueError("同一来源命中数超过上限")
            if any(
                hit.citation.min_hint_level > self.query.disclosure_level
                for hit in self.hits
            ):
                raise ValueError("检索命中超过当前提示披露层级")
        return self

    @property
    def citations(self) -> tuple[KnowledgeCitation, ...]:
        """Return citations in the same deterministic order as hits."""

        return tuple(hit.citation for hit in self.hits)

    def render_context_for_model(self) -> str:
        """Render only retrieved chunks with citation markers for the optional model."""

        if self.status != KnowledgeContextStatus.MATCHED:
            return self.message_zh
        blocks = [_render_hit_block(hit) for hit in self.hits]
        return "\n\n".join(blocks)

    @classmethod
    def index_unavailable(cls, query: RagQuery, message_zh: str | None = None) -> Self:
        """Return a safe degradation result without exposing internal failure details."""

        return cls(
            query=query,
            status=KnowledgeContextStatus.INDEX_UNAVAILABLE,
            hits=(),
            message_zh=message_zh or INDEX_UNAVAILABLE_MESSAGE_ZH,
            query_fingerprint=_fingerprint_query(query),
            index_fingerprint=None,
        )


class LocalRagIndex:
    """Immutable in-memory BM25-style index built only from validated sources."""

    def __init__(
        self,
        chunks: Iterable[RagChunk],
        *,
        excluded_sources: Iterable[ExcludedRagSource] = (),
    ) -> None:
        try:
            validated_chunks = tuple(
                RagChunk.model_validate(chunk.model_dump(mode="python"))
                for chunk in chunks
            )
        except ValidationError as error:
            raise RagIndexBuildError(
                RagIndexBuildErrorCode.CHUNK_SCHEMA_INVALID,
                "本地索引收到不符合 RagChunk 合同的切片",
            ) from error
        ordered_chunks = tuple(
            sorted(
                validated_chunks,
                key=lambda chunk: (
                    chunk.source_id,
                    chunk.section.value,
                    chunk.position,
                    chunk.subposition,
                    chunk.chunk_id,
                ),
            )
        )
        if not ordered_chunks:
            raise RagIndexBuildError(
                RagIndexBuildErrorCode.NO_CHUNKS,
                "没有可进入本地检索索引的课程切片",
            )
        invalid_chunks = [
            chunk.chunk_id
            for chunk in ordered_chunks
            if chunk.source_type != SourceType.PROJECT_AUTHORED
            or chunk.license != SourceLicense.PROJECT_OWNED
            or chunk.answer_leakage_risk != AnswerLeakageRisk.LOW
            or not chunk.allowed_to_quote
            or Path(chunk.relative_path).suffix.casefold() != ".yaml"
            or bool(find_source_instruction_attacks(_normalize_text(chunk.source_title)))
            or infer_minimum_disclosure_level(chunk.source_title) != 1
            or bool(find_source_instruction_attacks(_normalize_text(chunk.text)))
            or chunk.min_hint_level < infer_minimum_disclosure_level(chunk.text)
            or chunk.content_checksum != _checksum_text(chunk.text)
            or not chunk.chunk_id.startswith(
                f"{chunk.source_id}@{chunk.source_version}:"
            )
            or not chunk.chunk_id.endswith(_checksum_text(chunk.text)[7:19])
        ]
        if invalid_chunks:
            raise RagIndexBuildError(
                RagIndexBuildErrorCode.SOURCE_NOT_ELIGIBLE,
                "本地索引包含未通过原创、授权、泄露风险或引用政策的切片",
            )
        if len({chunk.chunk_id for chunk in ordered_chunks}) != len(ordered_chunks):
            raise RagIndexBuildError(
                RagIndexBuildErrorCode.NO_CHUNKS,
                "RAG 切片 chunk_id 重复，无法建立可审计索引",
            )

        self._chunks = ordered_chunks
        self._excluded_sources = tuple(
            sorted(excluded_sources, key=lambda source: source.source_id)
        )
        self._term_frequencies = tuple(_chunk_term_frequencies(chunk) for chunk in ordered_chunks)
        self._document_lengths = tuple(sum(counter.values()) for counter in self._term_frequencies)
        self._average_document_length = sum(self._document_lengths) / len(ordered_chunks)

        postings: dict[str, set[int]] = defaultdict(set)
        for index, frequencies in enumerate(self._term_frequencies):
            for term in frequencies:
                postings[term].add(index)
        self._postings = {term: frozenset(indices) for term, indices in postings.items()}
        self._index_fingerprint = _fingerprint_index(
            self._chunks,
            self._excluded_sources,
        )

    @property
    def chunks(self) -> tuple[RagChunk, ...]:
        return self._chunks

    @property
    def excluded_sources(self) -> tuple[ExcludedRagSource, ...]:
        return self._excluded_sources

    @property
    def index_fingerprint(self) -> str:
        return self._index_fingerprint

    @property
    def source_ids(self) -> tuple[str, ...]:
        return tuple(sorted({chunk.source_id for chunk in self._chunks}))

    def search(self, query: RagQuery | str) -> RagSearchResult:
        """Retrieve deterministic lexical matches, optionally within one concept."""

        request = query if isinstance(query, RagQuery) else RagQuery(text=query)
        if (
            request.concept_id is None
            and not request.knowledge_node_ids
            and not _has_course_domain_anchor(request.text)
        ):
            return self._no_match(request)
        query_terms = Counter(_tokenize(request.text))
        strong_query_terms = {term for term in query_terms if _is_strong_term(term)}
        if not strong_query_terms:
            return self._no_match(request)
        required_term_matches = min(2, len(strong_query_terms))

        candidate_indices: set[int] = set()
        for term in query_terms:
            candidate_indices.update(self._postings.get(term, ()))

        ranked: list[tuple[float, RagChunk, tuple[str, ...], bool]] = []
        for index in candidate_indices:
            chunk = self._chunks[index]
            if request.concept_id is not None and chunk.concept_id != request.concept_id:
                continue
            if chunk.min_hint_level > request.disclosure_level:
                continue
            frequencies = self._term_frequencies[index]
            matched_terms = tuple(
                sorted(
                    term
                    for term in query_terms
                    if term in frequencies and _is_strong_term(term)
                )
            )
            if len(matched_terms) < required_term_matches:
                continue
            score = self._score(index, query_terms, request)
            if score < _minimum_retrieval_score(request):
                continue
            node_overlap = bool(
                set(request.knowledge_node_ids) & set(chunk.knowledge_node_ids)
            )
            ranked.append((score, chunk, matched_terms, node_overlap))

        overlapping_source_ids = {
            chunk.source_id for _score, chunk, _terms, overlaps in ranked if overlaps
        }
        if overlapping_source_ids:
            ranked = [item for item in ranked if item[3]]

        ranked.sort(
            key=lambda item: (
                -item[0],
                item[1].source_id,
                item[1].section.value,
                item[1].position,
                item[1].subposition,
                item[1].chunk_id,
            )
        )
        ranked_positions = {
            item[1].chunk_id: position for position, item in enumerate(ranked)
        }
        reserved = []
        for section_group in _requested_section_intent_groups(request.text):
            candidate = next(
                (item for item in ranked if item[1].section in section_group),
                None,
            )
            if candidate is not None and candidate[1].chunk_id not in {
                item[1].chunk_id for item in reserved
            }:
                reserved.append(candidate)
        reserved.sort(key=lambda item: ranked_positions[item[1].chunk_id])
        reserved = reserved[: request.top_k]
        reserved_chunk_ids = {item[1].chunk_id for item in reserved}
        selection_candidates = [
            *reserved,
            *(item for item in ranked if item[1].chunk_id not in reserved_chunk_ids),
        ]

        selected: list[tuple[float, RagChunk, tuple[str, ...], bool]] = []
        per_source: Counter[str] = Counter()
        context_chars = 0
        maximum_per_source = (
            min(request.top_k, MAX_NODE_SCOPED_HITS_PER_SOURCE)
            if len(overlapping_source_ids) == 1
            else MAX_HITS_PER_SOURCE
        )
        for item in selection_candidates:
            score, chunk, _matched_terms, _node_overlap = item
            if per_source[chunk.source_id] >= maximum_per_source:
                continue
            rank = len(selected) + 1
            rendered_block_chars = len(_render_chunk_block(chunk, rank=rank))
            separator_chars = 2 if selected else 0
            if (
                context_chars + separator_chars + rendered_block_chars
                > request.maximum_context_chars
            ):
                continue
            selected.append(item)
            per_source[chunk.source_id] += 1
            context_chars += separator_chars + rendered_block_chars
            if len(selected) == request.top_k:
                break

        selected.sort(key=lambda item: ranked_positions[item[1].chunk_id])

        hits = tuple(
            _make_hit(chunk, score, matched_terms, rank=rank)
            for rank, (score, chunk, matched_terms, _node_overlap) in enumerate(
                selected,
                start=1,
            )
        )
        if not hits:
            return self._no_match(request)
        return RagSearchResult(
            query=request,
            status=KnowledgeContextStatus.MATCHED,
            hits=hits,
            message_zh=f"已从本地原创知识库检索到 {len(hits)} 条可引用依据。",
            query_fingerprint=_fingerprint_query(request),
            index_fingerprint=self.index_fingerprint,
        )

    def _score(
        self,
        index: int,
        query_terms: Counter[str],
        query: RagQuery,
    ) -> float:
        frequencies = self._term_frequencies[index]
        document_length = self._document_lengths[index]
        total_documents = len(self._chunks)
        k1 = 1.2
        b = 0.75
        score = 0.0
        for term, query_frequency in query_terms.items():
            term_frequency = frequencies.get(term, 0)
            if not term_frequency or not _is_strong_term(term):
                continue
            document_frequency = len(self._postings[term])
            inverse_document_frequency = math.log(
                1 + (total_documents - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            denominator = term_frequency + k1 * (
                1 - b + b * document_length / self._average_document_length
            )
            score += (
                inverse_document_frequency
                * (term_frequency * (k1 + 1) / denominator)
                * min(query_frequency, 2)
            )
        chunk = self._chunks[index]
        if query.knowledge_node_ids and set(query.knowledge_node_ids) & set(
            chunk.knowledge_node_ids
        ):
            score += 1.25
        score += _explicit_section_intent_bonus(query.text, chunk.section)
        return round(score, 12)

    def _no_match(self, query: RagQuery) -> RagSearchResult:
        return RagSearchResult(
            query=query,
            status=KnowledgeContextStatus.NO_MATCH,
            hits=(),
            message_zh=NO_MATCH_MESSAGE_ZH,
            query_fingerprint=_fingerprint_query(query),
            index_fingerprint=self.index_fingerprint,
        )


def build_local_rag_index(
    project_root: Path,
    manifest_path: Path | None = None,
) -> LocalRagIndex:
    """Build an index atomically from manifest-registered, integrity-checked sources."""

    root = Path(project_root).resolve(strict=True)
    manifest_file = manifest_path or root / "data" / "rag" / "manifest.yaml"
    manifest = load_rag_manifest(manifest_file)
    chunks: list[RagChunk] = []
    excluded_sources: list[ExcludedRagSource] = []

    for entry in sorted(manifest.sources, key=lambda source: source.source_id):
        loaded = load_rag_source(entry, root)
        rejection_codes, rejection_reasons = _index_policy_rejections(loaded)
        if rejection_codes:
            excluded_sources.append(
                ExcludedRagSource(
                    source_id=entry.source_id,
                    rejection_codes=rejection_codes,
                    rejection_reasons=rejection_reasons,
                )
            )
            continue
        chunks.extend(chunk_loaded_source(loaded))

    return LocalRagIndex(chunks, excluded_sources=excluded_sources)


def _index_policy_rejections(
    source: LoadedRagSource,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    entry = source.manifest_entry
    codes = [code.value for code in source.eligibility.rejection_codes]
    reasons = list(source.eligibility.rejection_reasons)
    policy_checks = (
        (
            entry.source_type != SourceType.PROJECT_AUTHORED,
            "source_not_project_authored",
            "比赛本地索引当前只接受团队原创资料",
        ),
        (
            entry.license != SourceLicense.PROJECT_OWNED,
            "license_not_project_owned",
            "比赛本地索引当前只接受 project-owned 授权",
        ),
        (
            entry.answer_leakage_risk != AnswerLeakageRisk.LOW,
            "answer_leakage_risk_not_low",
            "比赛本地索引只允许 low 答案泄露风险",
        ),
        (
            AllowedUsage.RETRIEVAL not in entry.allowed_usage,
            "retrieval_not_allowed",
            "allowed_usage 未允许 retrieval",
        ),
        (
            AllowedUsage.QUOTATION not in entry.allowed_usage,
            "quotation_not_allowed",
            "allowed_usage 未允许 quotation，无法生成可审计引用",
        ),
        (
            Path(entry.file_path).suffix.casefold() != ".yaml",
            "source_extension_not_yaml",
            "本地索引只接受 manifest 登记的 YAML 知识卡",
        ),
    )
    for rejected, code, reason in policy_checks:
        if rejected and code not in codes:
            codes.append(code)
            reasons.append(reason)
    return tuple(codes), tuple(reasons)


def chunk_loaded_source(source: LoadedRagSource) -> tuple[RagChunk, ...]:
    """Convert one eligible structured document into stable field-level chunks."""

    rejection_codes, _rejection_reasons = _index_policy_rejections(source)
    if rejection_codes:
        raise RagIndexBuildError(
            RagIndexBuildErrorCode.SOURCE_NOT_ELIGIBLE,
            f"资料 {source.manifest_entry.source_id} 未通过切片资格检查",
        )

    review_value = source.manifest_entry.metadata.get("content_status", "draft")
    try:
        review_status = ContentReviewStatus(str(review_value))
    except ValueError as error:
        raise RagIndexBuildError(
            RagIndexBuildErrorCode.REVIEW_STATUS_INVALID,
            f"资料 {source.manifest_entry.source_id} 的 content_status 无效",
        ) from error

    document = source.document
    section_texts: list[tuple[RagSection, int, int, str, int]] = []
    section_texts.extend(
        _plain_list_chunks(
            RagSection.LEARNING_OBJECTIVES,
            document.learning_objectives,
            min_hint_level=1,
        )
    )
    section_texts.extend(
        _plain_list_chunks(
            RagSection.PREREQUISITE_KNOWLEDGE,
            document.prerequisite_knowledge,
            min_hint_level=1,
        )
    )
    section_texts.extend(
        _plain_list_chunks(
            RagSection.CONCEPT_EXPLANATION,
            document.concept_explanation,
            min_hint_level=2,
        )
    )
    for position, formula in enumerate(document.formula_explanation):
        symbols = "；".join(
            f"{symbol}：{meaning}" for symbol, meaning in sorted(formula.symbols.items())
        )
        assumptions = "；".join(formula.assumptions)
        cautions = "；".join(formula.cautions)
        section_texts.append(
            (
                RagSection.FORMULA_MEANING,
                position,
                0,
                (
                    f"{formula.name}。含义：{formula.meaning}。"
                    f"假设：{assumptions}。注意：{cautions}。"
                ),
                2,
            )
        )
        section_texts.append(
            (
                RagSection.FORMULA_EXPRESSION,
                position,
                1,
                f"{formula.name}。表达式：{formula.expression}。符号：{symbols}。",
                3,
            )
        )
    for position, connection in enumerate(document.python_connection):
        section_texts.append(
            (
                RagSection.PYTHON_CONNECTION,
                position,
                0,
                (
                    f"{connection.library} / {connection.api}。用途：{connection.purpose}。"
                    f"输入要求：{connection.input_expectation}。"
                    f"解释提醒：{connection.interpretation_caution}。"
                ),
                3,
            )
        )
    section_texts.extend(
        _plain_list_chunks(
            RagSection.DATA_INTERPRETATION_GUIDANCE,
            document.data_interpretation_guidance,
            min_hint_level=2,
        )
    )
    for position, misconception in enumerate(document.common_misconceptions):
        section_texts.append(
            (
                RagSection.COMMON_MISCONCEPTIONS,
                position,
                0,
                (
                    f"常见误区：{misconception.misconception}。"
                    f"为什么不对：{misconception.why_incorrect}。"
                    f"可继续思考：{misconception.better_question}。"
                ),
                2,
            )
        )
    section_texts.extend(
        _plain_list_chunks(
            RagSection.REFLECTIVE_QUESTIONS,
            document.reflective_questions,
            min_hint_level=1,
        )
    )
    section_texts.extend(
        _plain_list_chunks(
            RagSection.SUMMARY,
            document.summary,
            min_hint_level=4,
        )
    )

    try:
        return tuple(
            _make_chunk(
                source,
                section=section,
                position=position,
                subposition=subposition,
                text=text,
                min_hint_level=min_hint_level,
                review_status=review_status,
            )
            for section, position, subposition, text, min_hint_level in section_texts
        )
    except ValidationError as error:
        raise RagIndexBuildError(
            RagIndexBuildErrorCode.CHUNK_SCHEMA_INVALID,
            f"资料 {source.manifest_entry.source_id} 无法生成合规的本地检索切片",
        ) from error


def _plain_list_chunks(
    section: RagSection,
    values: Iterable[str],
    *,
    min_hint_level: int,
) -> list[tuple[RagSection, int, int, str, int]]:
    return [
        (section, position, 0, value.strip(), min_hint_level)
        for position, value in enumerate(values)
    ]


def _make_chunk(
    source: LoadedRagSource,
    *,
    section: RagSection,
    position: int,
    subposition: int,
    text: str,
    min_hint_level: int,
    review_status: ContentReviewStatus,
) -> RagChunk:
    normalized_text = " ".join(text.split())
    effective_hint_level = max(
        min_hint_level,
        infer_minimum_disclosure_level(normalized_text),
    )
    digest = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
    source_entry = source.manifest_entry
    chunk_id = (
        f"{source_entry.source_id}@{source_entry.version}:"
        f"{section.value}:{position}:{subposition}:{digest[:12]}"
    )
    knowledge_node_values = source_entry.metadata.get("knowledge_node_ids", [])
    knowledge_node_ids = (
        tuple(str(value) for value in knowledge_node_values)
        if isinstance(knowledge_node_values, list)
        else ()
    )
    return RagChunk(
        chunk_id=chunk_id,
        source_id=source_entry.source_id,
        source_version=source_entry.version,
        source_title=source_entry.title,
        concept_id=source.document.concept_id,
        knowledge_node_ids=knowledge_node_ids,
        section=section,
        position=position,
        subposition=subposition,
        text=normalized_text,
        content_checksum=f"sha256:{digest}",
        source_checksum=source.content_checksum,
        source_type=source_entry.source_type,
        license=source_entry.license,
        answer_leakage_risk=source_entry.answer_leakage_risk,
        allowed_to_quote=AllowedUsage.QUOTATION in source_entry.allowed_usage,
        relative_path=source.relative_path,
        min_hint_level=effective_hint_level,
        review_status=review_status,
    )


def _chunk_term_frequencies(chunk: RagChunk) -> Counter[str]:
    frequencies = Counter(_tokenize(chunk.text))
    frequencies.update({term: 2 for term in _tokenize(chunk.source_title)})
    frequencies.update({term: 2 for term in _tokenize(chunk.concept_id.value)})
    return frequencies


def _explicit_section_intent_bonus(text: str, section: RagSection) -> float:
    """Prefer the requested content form without weakening lexical relevance."""

    normalized = _normalize_text(text)
    summary_cues = ("总结", "概括", "复习", "综合", "summary", "review")
    python_cues = (
        "python",
        "pandas",
        "numpy",
        "scipy",
        "series",
        "dataframe",
        "groupby",
        "api",
        "代码",
        "函数",
        "调用",
    )
    formula_cues = ("公式", "表达式", "列式", "怎样计算", "如何计算")
    explicit_intents = {
        RagSection.COMMON_MISCONCEPTIONS: (
            "常见误区",
            "常见错误",
            "错误理解",
            "为什么不等于",
            "不能说明",
            "不代表",
            "容易混淆",
        ),
        RagSection.DATA_INTERPRETATION_GUIDANCE: (
            "解释结果",
            "如何解读",
            "报告结论",
            "因果边界",
            "模型边界",
        ),
        RagSection.PREREQUISITE_KNOWLEDGE: (
            "前置知识",
            "先学什么",
            "需要先学",
            "学习之前",
            "学习前",
        ),
        RagSection.REFLECTIVE_QUESTIONS: (
            "反思",
            "自检",
            "进一步思考",
            "还需检查",
            "继续思考",
        ),
        RagSection.CONCEPT_EXPLANATION: (
            "概念含义",
            "基本概念",
            "定义是什么",
            "是什么",
        ),
    }
    weak_intents = {
        RagSection.COMMON_MISCONCEPTIONS: ("误区", "为什么不", "混淆"),
        RagSection.DATA_INTERPRETATION_GUIDANCE: ("解释", "解读", "影响"),
        RagSection.PREREQUISITE_KNOWLEDGE: ("前置", "先学", "之前"),
        RagSection.REFLECTIVE_QUESTIONS: ("思考",),
        RagSection.CONCEPT_EXPLANATION: ("概念", "含义", "为什么"),
    }

    bonus = 0.0
    if section == RagSection.SUMMARY and any(cue in normalized for cue in summary_cues):
        bonus += EXPLICIT_SECTION_INTENT_BONUS
    if section == RagSection.PYTHON_CONNECTION and any(
        cue in normalized for cue in python_cues
    ):
        bonus += EXPLICIT_SECTION_INTENT_BONUS
    if section in {RagSection.FORMULA_MEANING, RagSection.FORMULA_EXPRESSION} and any(
        cue in normalized for cue in formula_cues
    ):
        bonus += EXPLICIT_SECTION_INTENT_BONUS
    if any(cue in normalized for cue in explicit_intents.get(section, ())):
        bonus += SPECIFIC_SECTION_INTENT_BONUS
    elif any(cue in normalized for cue in weak_intents.get(section, ())):
        bonus += WEAK_SECTION_INTENT_BONUS
    return bonus


def _requested_section_intent_groups(text: str) -> tuple[frozenset[RagSection], ...]:
    """Return explicit section needs that should each receive Top-k coverage."""

    scored_sections = {
        section
        for section in RagSection
        if _explicit_section_intent_bonus(text, section)
        >= SPECIFIC_SECTION_INTENT_BONUS
    }
    groups: list[frozenset[RagSection]] = []
    formula_sections = {
        RagSection.FORMULA_MEANING,
        RagSection.FORMULA_EXPRESSION,
    }
    if scored_sections & formula_sections:
        groups.append(frozenset(formula_sections))
        scored_sections -= formula_sections
    groups.extend(
        frozenset((section,))
        for section in sorted(scored_sections, key=lambda item: item.value)
    )
    return tuple(groups)


def _has_course_domain_anchor(text: str) -> bool:
    """Require clear probability/statistics evidence for query-only retrieval."""

    normalized = _normalize_text(text)
    if any(_contains_anchor(normalized, anchor) for anchor in _COURSE_SPECIFIC_ANCHORS):
        return True
    general_matches = {
        anchor
        for anchor in _COURSE_GENERAL_ANCHORS
        if _contains_anchor(normalized, anchor)
    }
    return len(general_matches) >= 2


def _contains_anchor(normalized_text: str, anchor: str) -> bool:
    if re.search(r"[a-z]", anchor):
        return bool(
            re.search(
                rf"(?<![a-z0-9]){re.escape(anchor)}(?![a-z0-9])",
                normalized_text,
            )
        )
    return anchor in normalized_text


def _minimum_retrieval_score(query: RagQuery) -> float:
    if query.concept_id is not None or query.knowledge_node_ids:
        return MIN_RETRIEVAL_SCORE
    normalized = _normalize_text(query.text)
    if any(
        re.search(r"[a-z]", anchor) and _contains_anchor(normalized, anchor)
        for anchor in _COURSE_SPECIFIC_ANCHORS
    ):
        return MIN_RETRIEVAL_SCORE
    return MIN_UNSCOPED_CHINESE_RETRIEVAL_SCORE


def _tokenize(value: str) -> tuple[str, ...]:
    normalized = _normalize_text(value)
    if not normalized:
        return ()
    terms: list[str] = []
    for token in _TOKEN_PATTERN.findall(normalized):
        if token in _SINGLE_TOKEN_STOPWORDS:
            continue
        terms.append(token)
        if "_" in token:
            terms.extend(part for part in token.split("_") if part)

    compact = re.sub(r"\s+", "", normalized)
    for sequence in _COMPACT_SEQUENCE_PATTERN.findall(compact):
        if len(sequence) == 1:
            if sequence not in _SINGLE_TOKEN_STOPWORDS:
                terms.append(sequence)
            continue
        terms.extend(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return tuple(terms)


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split()).strip()


def _is_strong_term(term: str) -> bool:
    return len(term) >= 2 and term not in _SINGLE_TOKEN_STOPWORDS


def _make_hit(
    chunk: RagChunk,
    score: float,
    matched_terms: tuple[str, ...],
    *,
    rank: int,
) -> RagSearchHit:
    quote = _excerpt(chunk.text) if chunk.allowed_to_quote else None
    citation = KnowledgeCitation(
        citation_id=f"R{rank}",
        source_id=chunk.source_id,
        source_version=chunk.source_version,
        source_title=chunk.source_title,
        chunk_id=chunk.chunk_id,
        section=chunk.section.value,
        quote=quote,
        content_checksum=chunk.content_checksum,
        source_checksum=chunk.source_checksum,
        min_hint_level=chunk.min_hint_level,
        review_status=chunk.review_status,
    )
    return RagSearchHit(
        rank=rank,
        score=score,
        matched_terms=matched_terms,
        content=chunk.text,
        citation=citation,
    )


def _render_chunk_block(chunk: RagChunk, *, rank: int) -> str:
    return f"[R{rank}] {chunk.source_title} / {chunk.section.value}\n{chunk.text}"


def _render_hit_block(hit: RagSearchHit) -> str:
    citation = hit.citation
    return (
        f"[{citation.citation_id}] {citation.source_title} / "
        f"{citation.section}\n{hit.content}"
    )


def _excerpt(text: str) -> str:
    if len(text) <= 600:
        return text
    return f"{text[:599]}…"


def _checksum_text(text: str) -> str:
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def _fingerprint_index(
    chunks: tuple[RagChunk, ...],
    excluded_sources: tuple[ExcludedRagSource, ...],
) -> str:
    payload = {
        "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
        "excluded_sources": [source.model_dump(mode="json") for source in excluded_sources],
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(serialized.encode('utf-8')).hexdigest()}"


def _fingerprint_query(query: RagQuery) -> str:
    serialized = query.model_dump_json(exclude_none=False)
    return f"sha256:{hashlib.sha256(serialized.encode('utf-8')).hexdigest()}"
