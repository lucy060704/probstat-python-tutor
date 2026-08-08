"""Validated data models shared by the curriculum and later milestones."""

from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_ANSWER_LENGTH = 2_000
MAX_REASONING_LENGTH = 5_000
MAX_PYTHON_CODE_LENGTH = 10_000


class SubmissionField(StrEnum):
    """Learner-authored fields that may be cited as observable evidence."""

    ANSWER = "answer"
    REASONING = "reasoning"
    PYTHON_CODE = "python_code"


class EvidenceVerdict(StrEnum):
    """Deterministic interpretation of one observable submission fragment."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    INSUFFICIENT = "insufficient"
    IRRELEVANT = "irrelevant"
    UNSAFE = "unsafe"


class PythonStructureKind(StrEnum):
    """Supported result skeletons for the current fixed Python questions."""

    DIRECT_MEDIAN_CALL = "direct_median_call"
    DIRECT_STD_CALL = "direct_std_call"
    DIRECT_MISSING_COUNT_CHAIN = "direct_missing_count_chain"
    STANDARD_ERROR_FORMULA = "standard_error_formula"
    CONFIDENCE_INTERVAL_FORMULA = "confidence_interval_formula"
    SEEDED_BINOMIAL_PROPORTION = "seeded_binomial_proportion"
    DIRECT_BINOMIAL_PMF = "direct_binomial_pmf"
    DIRECT_GROUPBY_NAMED_AGG = "direct_groupby_named_agg"
    DIRECT_WELCH_TTEST_PVALUE = "direct_welch_ttest_pvalue"


class PythonMismatchKind(StrEnum):
    """Exact result structures that prove one supported Python misconception."""

    DIRECT_MEDIAN_ATTRIBUTE_REFERENCE = "direct_median_attribute_reference"
    SCALAR_ILOC_SELECTION = "scalar_iloc_selection"
    DIRECT_VAR_CALL = "direct_var_call"
    NUMPY_STD_MISSING_DDOF = "numpy_std_missing_ddof"
    VARIANCE_STANDARD_ERROR_FORMULA = "variance_standard_error_formula"
    POPULATION_STD_STANDARD_ERROR_FORMULA = "population_std_standard_error_formula"
    LINEAR_N_STANDARD_ERROR_FORMULA = "linear_n_standard_error_formula"
    RAW_STD_AS_STANDARD_ERROR = "raw_std_as_standard_error"
    ADDS_CONFIDENCE_MARGIN_BOTH_SIDES = "adds_confidence_margin_both_sides"
    REVERSED_CONFIDENCE_INTERVAL_ENDPOINTS = (
        "reversed_confidence_interval_endpoints"
    )
    OMITS_CONFIDENCE_CRITICAL_VALUE = "omits_confidence_critical_value"
    USES_STANDARD_DEVIATION_AS_CI_SCALE = "uses_standard_deviation_as_ci_scale"
    MISSING_METHOD_NOT_CALLED = "missing_method_not_called"
    NON_MISSING_ROW_COUNT = "non_missing_row_count"
    UNSEEDED_BINOMIAL_PROPORTION = "unseeded_binomial_proportion"
    BINOMIAL_CDF_FOR_EXACT_PROBABILITY = "binomial_cdf_for_exact_probability"
    GROUPBY_SIZE_FOR_VALID_COUNT = "groupby_size_for_valid_count"
    POOLED_TTEST_FOR_WELCH_QUESTION = "pooled_ttest_for_welch_question"
    ONE_SIDED_TTEST_FOR_TWO_SIDED_QUESTION = (
        "one_sided_ttest_for_two_sided_question"
    )
    TTEST_SAME_GROUP_TWICE = "ttest_same_group_twice"
    TTEST_RESULT_WITHOUT_PVALUE = "ttest_result_without_pvalue"


class LearnerSubmission(BaseModel):
    """Validated learner-authored text; Python code is never executed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    answer: str
    reasoning: str = ""
    python_code: str = ""

    @model_validator(mode="before")
    @classmethod
    def payload_uses_supported_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        supported = {"answer", "reasoning", "python_code"}
        unsupported = sorted(set(data) - supported)
        if unsupported:
            raise ValueError(f"学习者提交包含不支持的字段：{', '.join(unsupported)}")
        if "answer" not in data:
            raise ValueError("答案不能为空")
        return data

    @field_validator("answer", mode="before")
    @classmethod
    def answer_is_text(cls, value: object) -> object:
        if not isinstance(value, str):
            raise ValueError("答案必须是文本")
        return value

    @field_validator("reasoning", mode="before")
    @classmethod
    def reasoning_is_text(cls, value: object) -> object:
        if not isinstance(value, str):
            raise ValueError("思考过程必须是文本")
        return value

    @field_validator("python_code", mode="before")
    @classmethod
    def python_code_is_text(cls, value: object) -> object:
        if not isinstance(value, str):
            raise ValueError("Python 代码必须是文本")
        return value

    @field_validator("answer")
    @classmethod
    def answer_is_present_and_bounded(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("答案不能为空")
        if len(value) > MAX_ANSWER_LENGTH:
            raise ValueError(f"答案不能超过 {MAX_ANSWER_LENGTH} 个字符")
        return value

    @field_validator("reasoning")
    @classmethod
    def reasoning_is_bounded(cls, value: str) -> str:
        if len(value) > MAX_REASONING_LENGTH:
            raise ValueError(f"思考过程不能超过 {MAX_REASONING_LENGTH} 个字符")
        return value

    @field_validator("python_code")
    @classmethod
    def python_code_is_bounded(cls, value: str) -> str:
        if len(value) > MAX_PYTHON_CODE_LENGTH:
            raise ValueError(f"Python 代码不能超过 {MAX_PYTHON_CODE_LENGTH} 个字符")
        return value


class LearningSubmissionRequest(BaseModel):
    """Validated command metadata plus the learner-authored submission."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    learner_id: str
    session_id: str
    question_id: str
    submission: LearnerSubmission
    hint_level: int
    idempotency_key: str | None = None

    @field_validator("learner_id", "session_id", "question_id", mode="before")
    @classmethod
    def identifiers_are_text(cls, value: object) -> object:
        if not isinstance(value, str):
            raise ValueError("学习者、会话和题目标识必须是文本")
        return value

    @field_validator("learner_id", "session_id", "question_id")
    @classmethod
    def identifiers_are_bounded(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("学习者、会话和题目标识不能为空")
        if len(value) > 200:
            raise ValueError("学习者、会话和题目标识不能超过 200 个字符")
        return value

    @field_validator("hint_level", mode="before")
    @classmethod
    def hint_level_is_supported(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int) or value not in range(5):
            raise ValueError("提示层级必须是 0、1、2、3 或 4")
        return value

    @field_validator("idempotency_key", mode="before")
    @classmethod
    def idempotency_key_is_text(cls, value: object) -> object:
        if value is not None and not isinstance(value, str):
            raise ValueError("幂等键必须是文本")
        return value

    @field_validator("idempotency_key")
    @classmethod
    def idempotency_key_is_safe(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not 8 <= len(value) <= 128:
            raise ValueError("幂等键长度必须为 8–128 个字符")
        normalized = value.replace("_", "").replace("-", "")
        if not normalized.isascii() or not normalized.isalnum():
            raise ValueError("幂等键只能包含字母、数字、下划线和连字符")
        return value


class LearnerEvidence(BaseModel):
    """One exact, source-labelled excerpt from a learner submission."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: SubmissionField
    quote: str = Field(min_length=1, max_length=500)


class ConceptId(StrEnum):
    """Published, deterministically graded knowledge points."""

    DATA_QUALITY = "data_quality"
    MEAN_MEDIAN = "mean_median"
    VARIANCE_STD = "variance_std"
    PROBABILITY_SIMULATION = "probability_simulation"
    COMMON_DISTRIBUTIONS = "common_distributions"
    JOINT_CORRELATION = "joint_correlation"
    SAMPLING_STANDARD_ERROR = "sampling_standard_error"
    CONFIDENCE_INTERVAL = "confidence_interval"
    HYPOTHESIS_TESTING = "hypothesis_testing"


class DeepUnitId(StrEnum):
    """Eight deep units promised by the competition curriculum boundary."""

    DATA_QUALITY = "data_quality"
    DESCRIPTIVE_STATISTICS = "descriptive_statistics"
    PROBABILITY_SIMULATION = "probability_simulation"
    COMMON_DISTRIBUTIONS = "common_distributions"
    JOINT_CORRELATION = "joint_correlation"
    SAMPLING_INFERENCE = "sampling_inference"
    ESTIMATION_CONFIDENCE_INTERVAL = "estimation_confidence_interval"
    HYPOTHESIS_TESTING = "hypothesis_testing"


class ContentReviewStatus(StrEnum):
    """Human review state; only a teacher may move content to approved."""

    DRAFT = "draft"
    PENDING_TEACHER_REVIEW = "pending_teacher_review"
    APPROVED = "approved"


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


CAPABILITY_LABELS_ZH = {
    CapabilityDimension.CONCEPT: "统计概念理解",
    CapabilityDimension.CALCULATION: "数学计算",
    CapabilityDimension.PYTHON: "Python 实现",
    CapabilityDimension.INTERPRETATION: "数据解释",
}


class EvidenceFinding(BaseModel):
    """One rule-backed finding tied to an exact learner-authored source."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rule_id: str = Field(min_length=1, pattern=r"^[a-z0-9_]+$")
    source: SubmissionField
    dimension: CapabilityDimension | None = None
    verdict: EvidenceVerdict
    message_zh: str = Field(min_length=1)
    quote: str | None = Field(default=None, min_length=1, max_length=500)
    misconception_tag: str | None = None


class GradeResult(BaseModel):
    """Unified, deterministic result returned by every grader."""

    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0.0, le=1.0)
    is_correct: bool
    evidence: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    misconception_candidates: list[str] = Field(default_factory=list)
    findings: list[EvidenceFinding] = Field(default_factory=list)


class TextEvidenceRule(BaseModel):
    """A bounded phrase rule; arbitrary regular expressions are not accepted."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rule_id: str = Field(min_length=1, pattern=r"^[a-z0-9_]+$")
    source: SubmissionField
    dimension: CapabilityDimension | None = None
    verdict: EvidenceVerdict
    phrases: tuple[str, ...] = Field(min_length=1)
    negation_guards: tuple[str, ...] = ()
    message_zh: str = Field(min_length=1)
    misconception_tag: str | None = None


class PythonStaticVariant(BaseModel):
    """One accepted set of observable AST features for a Python response."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    required_calls: tuple[str, ...] = ()
    required_names: tuple[str, ...] = ()
    required_operators: tuple[str, ...] = ()
    allowed_root_kinds: tuple[str, ...] = ()
    allowed_operators: tuple[str, ...] | None = None
    required_constants: tuple[str, ...] = ()
    required_keywords: tuple[str, ...] = ()


class PythonMismatchRule(BaseModel):
    """One specific AST-observable misconception used after valid variants fail."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rule_id: str = Field(min_length=1, pattern=r"^[a-z0-9_]+$")
    kind: PythonMismatchKind
    message_zh: str = Field(min_length=1)
    misconception_tag: str = Field(min_length=1, pattern=r"^[a-z0-9_]+$")


class PythonStaticSpec(BaseModel):
    """Question-owned AST requirements; matching never executes learner code."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    structure_kind: PythonStructureKind
    variants: tuple[PythonStaticVariant, ...] = Field(min_length=1)
    mismatch_rules: tuple[PythonMismatchRule, ...] = ()
    mismatch_rule_id: str = Field(pattern=r"^[a-z0-9_]+$")
    mismatch_message_zh: str = Field(min_length=1)
    misconception_tag: str | None = None


class ReasoningInsufficientRule(BaseModel):
    """Question-owned label for a missing reasoning condition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rule_id: str = Field(min_length=1, pattern=r"^[a-z0-9_]+$")
    message_zh: str = Field(min_length=1)
    misconception_tag: str = Field(min_length=1, pattern=r"^[a-z0-9_]+$")


class EvidencePolicy(BaseModel):
    """Structured multi-evidence policy attached to one question."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reasoning_required: bool = False
    reasoning_support_any: tuple[str, ...] = ()
    reasoning_support_groups: tuple[tuple[str, ...], ...] = ()
    reasoning_support_negation_guards: tuple[str, ...] = ()
    reasoning_insufficient_rule: ReasoningInsufficientRule | None = None
    relevance_terms: tuple[str, ...] = ()
    text_rules: tuple[TextEvidenceRule, ...] = ()
    python_static_spec: PythonStaticSpec | None = None

    @model_validator(mode="after")
    def reasoning_support_contract_is_unambiguous(self) -> Self:
        if self.reasoning_support_any and self.reasoning_support_groups:
            raise ValueError(
                "reasoning_support_any 和 reasoning_support_groups 不能同时配置"
            )
        has_empty_group_or_phrase = any(
            not group or any(not phrase.strip() for phrase in group)
            for group in self.reasoning_support_groups
        )
        if has_empty_group_or_phrase:
            raise ValueError("reasoning_support_groups 的每组都必须包含非空短语")
        if any(not guard.strip() for guard in self.reasoning_support_negation_guards):
            raise ValueError("reasoning_support_negation_guards 不能包含空文本")
        return self


class PythonCallFeature(BaseModel):
    """Safe structural description of one observed Python call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1)
    keyword_constants: tuple[str, ...] = ()


class PythonExpressionFeatures(BaseModel):
    """Connected AST features belonging to one result-bearing expression."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    root_kind: str = Field(min_length=1)
    structure_kinds: tuple[PythonStructureKind, ...] = ()
    mismatch_kinds: tuple[PythonMismatchKind, ...] = ()
    calls: tuple[PythonCallFeature, ...] = ()
    attributes: tuple[str, ...] = ()
    names: tuple[str, ...] = ()
    operators: tuple[str, ...] = ()
    constants: tuple[str, ...] = ()


class PythonStaticAnalysis(BaseModel):
    """Bounded AST facts extracted without compiling or running code."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    syntax_valid: bool
    node_count: int = Field(ge=0)
    calls: tuple[PythonCallFeature, ...] = ()
    attributes: tuple[str, ...] = ()
    names: tuple[str, ...] = ()
    operators: tuple[str, ...] = ()
    constants: tuple[str, ...] = ()
    result_expressions: tuple[PythonExpressionFeatures, ...] = ()
    unsafe_features: tuple[str, ...] = ()
    error_zh: str | None = None


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
    hint_level: int = Field(ge=0, le=4)


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
            self.mastery = {
                **{concept: MasteryScores() for concept in missing},
                **self.mastery,
            }
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
    recommended_hint_level: int = Field(default=0, ge=0, le=4)
    reason: str


class RecommendationKind(StrEnum):
    """Deterministic category for the learner's immediate next action."""

    NEXT_QUESTION = "next_question"
    COMPLETE = "complete"
    BLOCKED = "blocked"
    RETRY_CONTRADICTION = "retry_contradiction"
    RETRY_INSUFFICIENT = "retry_insufficient"
    RETRY_IRRELEVANT = "retry_irrelevant"
    RETRY_UNSAFE = "retry_unsafe"
    RETRY_ANSWER = "retry_answer"


class RecommendationDecision(BaseModel):
    """Evidence-linked recommendation that the optional model cannot override."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: RecommendationKind
    action_zh: str = Field(min_length=1)
    next_question_id: str | None = None
    target_dimension: CapabilityDimension | None = None
    source_rule_id: str | None = Field(
        default=None,
        pattern=r"^[a-z0-9_]+$",
    )


class KnowledgeContextStatus(StrEnum):
    """Whether the deterministic local knowledge base found usable context."""

    MATCHED = "matched"
    NO_MATCH = "no_match"
    INDEX_UNAVAILABLE = "index_unavailable"
    POLICY_BLOCKED = "policy_blocked"


class DeliveryMode(StrEnum):
    """How the learner-facing explanation was delivered for this submission."""

    DETERMINISTIC_OFFLINE = "deterministic_offline"
    MODEL_ENHANCED = "model_enhanced"
    MODEL_FALLBACK = "model_fallback"
    SAFETY_ISOLATED = "safety_isolated"


class KnowledgeCitation(BaseModel):
    """One auditable citation produced by the deterministic local retriever."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    source_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    source_title: str = Field(min_length=1, max_length=200)
    citation_id: str = Field(pattern=r"^R[1-5]$")
    chunk_id: str = Field(pattern=r"^[a-z][a-z0-9_.:@-]*$")
    section: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    quote: str | None = Field(default=None, min_length=1, max_length=600)
    content_checksum: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    source_checksum: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    min_hint_level: int = Field(ge=1, le=4)
    review_status: ContentReviewStatus


class DiagnosticReport(BaseModel):
    """Structured tutor output whose factual fields come from deterministic tools."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    overall_correctness: float = Field(ge=0.0, le=1.0)
    dimension_scores: MasteryScores
    evidence: list[str] = Field(min_length=1)
    learner_evidence: list[LearnerEvidence] = Field(default_factory=list)
    grader_findings: list[EvidenceFinding] = Field(default_factory=list)
    misconception_tags: list[str]
    feedback: str
    hint_level: int = Field(ge=0, le=4)
    recommended_action: str
    recommendation_kind: RecommendationKind | None = None
    recommendation_rule_id: str | None = None
    recommendation_dimension: CapabilityDimension | None = None
    next_question_id: str | None
    uncertainty: str
    knowledge_context_status: KnowledgeContextStatus = KnowledgeContextStatus.NO_MATCH
    knowledge_context_message: str = "尚未检索本地知识库。"
    knowledge_citations: list[KnowledgeCitation] = Field(default_factory=list)
    delivery_mode: DeliveryMode = DeliveryMode.DETERMINISTIC_OFFLINE
    delivery_message_zh: str = "本次使用确定性本地诊断。"


class CompleteExplanation(BaseModel):
    """The fourth hint explicitly covers all four learning dimensions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    concept: str = Field(min_length=1, max_length=1200)
    calculation: str = Field(min_length=1, max_length=1200)
    python: str = Field(min_length=1, max_length=1200)
    interpretation: str = Field(min_length=1, max_length=1200)

    def render_zh(self) -> str:
        """Render a beginner-readable explanation while retaining typed parts."""

        return (
            f"完整解释：\n概念：{self.concept}\n计算：{self.calculation}\n"
            f"Python：{self.python}\n情境解释：{self.interpretation}"
        )


class ProgressiveHints(BaseModel):
    """Four increasingly explicit hints; only level four is a full explanation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    concept_cue: str = Field(min_length=1, max_length=1200)
    method_cue: str = Field(min_length=1, max_length=1200)
    partial_step: str = Field(min_length=1, max_length=1600)
    complete_explanation: CompleteExplanation

    def for_level(self, level: int) -> str:
        """Return one validated level without silently changing the request."""

        hints: dict[int, str] = {
            1: self.concept_cue,
            2: self.method_cue,
            3: self.partial_step,
            4: self.complete_explanation.render_zh(),
        }
        try:
            return hints[level]
        except KeyError as error:
            raise ValueError("提示层级必须是 1、2、3 或 4") from error


class Question(BaseModel):
    """One validated curriculum question."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, pattern=r"^[a-z0-9_]+$")
    title: str = Field(min_length=1)
    unit_id: DeepUnitId
    concept_id: ConceptId
    knowledge_node_ids: tuple[str, ...] = ()
    prerequisites: list[ConceptId]
    difficulty: float = Field(ge=0.0, le=1.0)
    question_type: QuestionType
    prompt: str = Field(min_length=1)
    dataset: dict[str, Any]
    expected_answer: Any
    accepted_answers: tuple[str, ...] = ()
    numeric_tolerance: float | None = Field(default=None, ge=0.0)
    rubric: list[str] = Field(min_length=1)
    misconception_tags: list[str]
    dimension_weights: DimensionWeights
    evidence_policy: EvidencePolicy = Field(default_factory=EvidencePolicy)
    hints: ProgressiveHints | None = None
    python_code_required: bool = False
    review_status: ContentReviewStatus = ContentReviewStatus.DRAFT
    version: str = Field(default="0.1.0", pattern=r"^\d+\.\d+\.\d+$")


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
            if len(concept_questions) < 3:
                raise ValueError(f"知识点 {concept.value} 必须至少包含三道题")
            actual_types = {question.question_type for question in concept_questions}
            if not set(QuestionType) <= actual_types:
                raise ValueError(f"知识点 {concept.value} 必须至少各有一道概念、Python 和解释题")

        return self
