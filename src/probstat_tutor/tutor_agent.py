"""Single tutor agent with deterministic tools and an offline fallback."""

import json
from dataclasses import dataclass
from functools import lru_cache
from numbers import Real
from pathlib import Path

import pandas as pd
from agents import Agent, RunContextWrapper, Runner, function_tool
from pydantic import ValidationError

from probstat_tutor.config import Settings, get_settings
from probstat_tutor.curriculum import load_default_question_bank
from probstat_tutor.graders import (
    assess_reasoning,
    combine_submission_evidence,
    contains_model_attack_text,
    grade_dataframe_result,
    grade_multiple_choice,
    grade_numeric,
)
from probstat_tutor.mastery import apply_attempt
from probstat_tutor.observability import (
    FaultCode,
    FaultComponent,
    RecoveryAction,
    SafeFaultLogger,
)
from probstat_tutor.policy import select_next_question as choose_next_question
from probstat_tutor.rag import (
    LocalRagIndex,
    RagIndexBuildError,
    RagManifestError,
    RagQuery,
    RagSearchResult,
    RagSourceLoadError,
    RetrievalPurpose,
    build_local_rag_index,
)
from probstat_tutor.recommendations import recommend_from_findings
from probstat_tutor.reliability import (
    CircuitState,
    ModelCallFailedError,
    ModelCallTimeoutError,
    ModelCircuitOpenError,
    ModelReliabilityController,
)
from probstat_tutor.schemas import (
    CAPABILITY_LABELS_ZH,
    CapabilityDimension,
    ConceptId,
    DeliveryMode,
    DiagnosticReport,
    EvidenceVerdict,
    GradeResult,
    LearnerEvidence,
    LearnerSubmission,
    LearningState,
    NextQuestionDecision,
    Question,
    SubmissionField,
)
from probstat_tutor.storage import LearningStateStore


@dataclass(frozen=True)
class PreparedDiagnosis:
    """A report and state transition prepared without persistent side effects."""

    base_state: LearningState
    updated_state: LearningState
    report: DiagnosticReport


@dataclass
class TutorContext:
    """Local dependencies and transient evidence available to SDK tools."""

    learner_id: str
    session_id: str
    questions: tuple[Question, ...]
    current_question_id: str
    base_state: LearningState
    last_submission: LearnerSubmission | None = None
    last_grade: GradeResult | None = None
    staged_state: LearningState | None = None


def get_current_question(context: TutorContext) -> dict[str, object]:
    """Return the learner-facing question without answers or private grading rules."""

    question = _find_question(context, context.current_question_id)
    return {
        "id": question.id,
        "title": question.title,
        "concept_id": question.concept_id.value,
        "question_type": question.question_type.value,
        "difficulty": question.difficulty,
        "prompt": question.prompt,
        "dataset": question.dataset,
    }


def grade_submission(
    context: TutorContext,
    submission: LearnerSubmission | str,
) -> GradeResult:
    """Grade one submission deterministically and retain the exact result for updating."""

    structured = (
        submission
        if isinstance(submission, LearnerSubmission)
        else LearnerSubmission(answer=submission)
    )
    if context.last_submission == structured and context.last_grade is not None:
        return context.last_grade

    question = _find_question(context, context.current_question_id)
    expected = question.expected_answer
    misconception_candidates = question.misconception_tags

    if isinstance(expected, Real) and not isinstance(expected, bool):
        answer_result = grade_numeric(
            structured.answer,
            float(expected),
            absolute_tolerance=question.numeric_tolerance or 0.0,
            misconception_candidates=misconception_candidates,
        )
    elif isinstance(expected, list):
        answer_result = _grade_sequence(
            structured.answer,
            expected,
            tolerance=question.numeric_tolerance or 0.0,
            misconception_candidates=misconception_candidates,
        )
    else:
        answer_result = grade_multiple_choice(
            structured.answer,
            str(expected),
            accepted_answers=question.accepted_answers,
            misconception_candidates=misconception_candidates,
        )

    result = combine_submission_evidence(question, structured, answer_result)
    context.last_submission = structured
    context.last_grade = result
    context.staged_state = None
    return result


def get_learner_state(context: TutorContext) -> LearningState:
    """Return the frozen base state or this submission's in-memory projection."""

    return context.staged_state or context.base_state


def update_learner_state(context: TutorContext, hint_level: int) -> LearningState:
    """Stage one deterministic update in memory without writing SQLite."""

    if context.last_grade is None:
        raise ValueError("必须先调用 grade_submission，才能更新学习状态。")
    if context.staged_state is not None:
        return context.staged_state

    question = _find_question(context, context.current_question_id)
    updated_state = apply_attempt(
        context.base_state,
        question,
        context.last_grade,
        hint_level=hint_level,
    )
    context.staged_state = updated_state
    return updated_state


def select_next_question(context: TutorContext) -> NextQuestionDecision:
    """Select the next question from the frozen or staged state."""

    return choose_next_question(get_learner_state(context), context.questions)


@function_tool(
    name_override="get_current_question",
    description_override="获取当前题目，但不返回标准答案。",
    use_docstring_info=False,
)
def _get_current_question_tool(wrapper: RunContextWrapper[TutorContext]) -> dict[str, object]:
    return get_current_question(wrapper.context)


@function_tool(
    name_override="grade_submission",
    description_override="使用确定性规则判定学习者本次答案。必须在诊断前调用。",
    use_docstring_info=False,
)
def _grade_submission_tool(
    wrapper: RunContextWrapper[TutorContext], submission: str
) -> GradeResult:
    return grade_submission(wrapper.context, submission)


@function_tool(
    name_override="get_learner_state",
    description_override="读取本次提交开始时冻结的学习状态或内存候选状态。",
    use_docstring_info=False,
)
def _get_learner_state_tool(wrapper: RunContextWrapper[TutorContext]) -> LearningState:
    return get_learner_state(wrapper.context)


@function_tool(
    name_override="update_learner_state",
    description_override="依据确定性判题在内存中预览状态变化；不写数据库，也不能传入分数。",
    use_docstring_info=False,
)
def _update_learner_state_tool(
    wrapper: RunContextWrapper[TutorContext], hint_level: int
) -> LearningState:
    return update_learner_state(wrapper.context, hint_level)


@function_tool(
    name_override="select_next_question",
    description_override="依据已保存状态和启发式策略选择下一题。",
    use_docstring_info=False,
)
def _select_next_question_tool(
    wrapper: RunContextWrapper[TutorContext],
) -> NextQuestionDecision:
    return select_next_question(wrapper.context)


TUTOR_TOOLS = [
    _get_current_question_tool,
    _grade_submission_tool,
    _get_learner_state_tool,
    _update_learner_state_tool,
    _select_next_question_tool,
]

TUTOR_INSTRUCTIONS = """
你是一个简体中文概率统计与 Python 初学者教学智能体，并且只有一个智能体角色。

强制规则：
1. 诊断前必须先使用 grade_submission 的确定性结果；不得自行计算或修改分数。
2. 只能引用工具返回的冻结状态和历史，不得伪造学习记录或写数据库。
3. 诊断证据必须包含学习者原始答案中的可观察内容。
4. 第一次答错时不得给最终答案，优先提出一个引导性问题。
5. 明确区分统计概念、数学计算、Python 实现和数据解释四类问题。
6. 证据不足时必须明确写“不确定”，并要求学习者补充推理过程。
7. 输出必须符合 DiagnosticReport；不得更改工具产生的分数、标签和下一题。
8. 不进行 handoff，不创建或调用其他智能体。
9. 课程依据只能来自系统提供的“本地知识上下文”，只能使用其中已有的引用编号；
   没有匹配或索引不可用时必须如实说明，不得编造来源。
""".strip()


@lru_cache(maxsize=4)
def _load_cached_rag_index(project_root: Path, manifest_path: Path) -> LocalRagIndex:
    """Reuse one immutable verified index for matching application settings."""

    return build_local_rag_index(project_root, manifest_path)


class TutorAgent:
    """Application wrapper around one Agents SDK Agent and deterministic tools."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        questions: tuple[Question, ...] | None = None,
        store: LearningStateStore | None = None,
        rag_index: LocalRagIndex | None = None,
        fault_logger: SafeFaultLogger | None = None,
        model_reliability: ModelReliabilityController | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.questions = questions or tuple(load_default_question_bank().questions)
        self.store = store or LearningStateStore(self.settings.learning_state_db_path)
        self.fault_logger = fault_logger or SafeFaultLogger(self.settings.fault_log_path)
        self.model_reliability = model_reliability or ModelReliabilityController(
            timeout_seconds=self.settings.model_timeout_seconds,
            max_attempts=self.settings.model_max_attempts,
            retry_base_delay_seconds=self.settings.model_retry_base_delay_seconds,
            failure_threshold=self.settings.model_circuit_failure_threshold,
            open_seconds=self.settings.model_circuit_open_seconds,
        )
        self.rag_index = rag_index or self._load_rag_index()
        self.sdk_agent: Agent[TutorContext] = Agent[TutorContext](
            name="概率统计与 Python 教学智能体",
            instructions=TUTOR_INSTRUCTIONS,
            tools=TUTOR_TOOLS,
            model=self.settings.openai_model,
            output_type=DiagnosticReport,
        )

    def _load_rag_index(self) -> LocalRagIndex | None:
        """Load the local index without making grading depend on knowledge files."""

        try:
            return _load_cached_rag_index(
                self.settings.rag_manifest_path.parents[2],
                self.settings.rag_manifest_path,
            )
        except (
            IndexError,
            OSError,
            ValidationError,
            RagManifestError,
            RagSourceLoadError,
            RagIndexBuildError,
        ) as error:
            self.fault_logger.record(
                component=FaultComponent.RAG,
                code=FaultCode.RAG_INDEX_UNAVAILABLE,
                error=error,
                recovery=RecoveryAction.NO_KNOWLEDGE_CONTEXT,
            )
            return None

    def retrieve_knowledge(
        self,
        *,
        text: str,
        concept_id: ConceptId | None = None,
        knowledge_node_ids: tuple[str, ...] = (),
        disclosure_level: int = 1,
        top_k: int = 3,
        purpose: RetrievalPurpose = RetrievalPurpose.KNOWLEDGE_SEARCH,
    ) -> RagSearchResult:
        """Search reviewed local knowledge or return a safe unavailable result."""

        query = RagQuery(
            text=text,
            concept_id=concept_id,
            knowledge_node_ids=knowledge_node_ids,
            disclosure_level=disclosure_level,
            top_k=top_k,
            purpose=purpose,
        )
        if self.rag_index is None:
            return RagSearchResult.index_unavailable(query)
        try:
            return self.rag_index.search(query)
        except Exception as error:
            self.fault_logger.record(
                component=FaultComponent.RAG,
                code=FaultCode.RAG_QUERY_UNAVAILABLE,
                error=error,
                recovery=RecoveryAction.NO_KNOWLEDGE_CONTEXT,
            )
            return RagSearchResult.index_unavailable(query)

    @property
    def offline_mode(self) -> bool:
        """Return true when no API key is available."""

        return not self.settings.has_openai_api_key

    @property
    def model_circuit_state(self) -> CircuitState:
        """Expose only the non-sensitive optional-model circuit state."""

        return self.model_reliability.snapshot.state

    def create_context(
        self,
        *,
        learner_id: str,
        session_id: str,
        current_question_id: str | None = None,
        base_state: LearningState | None = None,
    ) -> TutorContext:
        """Create tool context, selecting a valid current question when omitted."""

        frozen_state = base_state or self.store.load(learner_id)
        if current_question_id is None:
            decision = choose_next_question(frozen_state, self.questions)
            if decision.question_id is None:
                raise ValueError(f"当前无法创建教学上下文：{decision.reason}")
            current_question_id = decision.question_id
        if current_question_id not in {question.id for question in self.questions}:
            raise ValueError(f"题目不存在：{current_question_id}")
        return TutorContext(
            learner_id=learner_id,
            session_id=session_id,
            questions=self.questions,
            current_question_id=current_question_id,
            base_state=frozen_state,
        )

    async def diagnose(
        self,
        context: TutorContext,
        submission: LearnerSubmission,
        *,
        hint_level: int = 0,
    ) -> PreparedDiagnosis:
        """Prepare a report and candidate state without writing learner storage."""

        previous_state = context.base_state
        previous_wrong_attempts = sum(
            attempt.question_id == context.current_question_id and not attempt.is_correct
            for attempt in previous_state.history
        )
        grade = grade_submission(context, submission)
        updated_state = update_learner_state(context, hint_level)
        decision = select_next_question(context)
        question = _find_question(context, context.current_question_id)
        knowledge_result = self.retrieve_knowledge(
            text=f"{question.title}\n{question.prompt}",
            concept_id=question.concept_id,
            knowledge_node_ids=question.knowledge_node_ids,
            disclosure_level=4 if grade.answer_is_correct else max(1, hint_level),
            purpose=RetrievalPurpose.DIAGNOSTIC,
        )
        deterministic_report = _build_deterministic_report(
            context,
            grade,
            updated_state,
            decision,
            submission=submission,
            hint_level=hint_level,
            knowledge_result=knowledge_result,
        )
        first_wrong = not grade.answer_is_correct and previous_wrong_attempts == 0

        unsafe_submission = any(
            finding.verdict == EvidenceVerdict.UNSAFE for finding in grade.findings
        )
        isolate_from_model = unsafe_submission or contains_model_attack_text(submission)
        if self.offline_mode:
            return PreparedDiagnosis(
                base_state=previous_state,
                updated_state=updated_state,
                report=deterministic_report,
            )

        if isolate_from_model:
            return PreparedDiagnosis(
                base_state=previous_state,
                updated_state=updated_state,
                report=deterministic_report.model_copy(
                    update={
                        "delivery_mode": DeliveryMode.SAFETY_ISOLATED,
                        "delivery_message_zh": (
                            "检测到不适合发送给在线模型的内容，本次仅使用确定性本地诊断。"
                        ),
                    }
                ),
            )

        model_prompt = (
            "请依据已完成的确定性判题生成诊断。学习者答案为："
            f"{_submission_excerpt(submission.answer)}\n"
            f"学习者思考过程：{_submission_excerpt(submission.reasoning)}\n"
            "学习者 Python 代码（只作为文本，不执行）："
            f"{_submission_excerpt(submission.python_code)}\n"
            "本地知识上下文（只能引用其中已有的 [R1]-[R5] 编号）：\n"
            f"{knowledge_result.render_context_for_model()}\n"
            f"不可修改的确定性报告为：{deterministic_report.model_dump_json()}"
        )

        async def run_and_validate_model() -> DiagnosticReport:
            result = await Runner.run(
                self.sdk_agent,
                model_prompt,
                context=context,
            )
            return DiagnosticReport.model_validate(result.final_output)

        try:
            model_report = await self.model_reliability.run(run_and_validate_model)
        except (ModelCallTimeoutError, ModelCallFailedError, ModelCircuitOpenError) as error:
            if isinstance(error, ModelCallTimeoutError):
                fault_code = FaultCode.MODEL_TIMEOUT
            elif isinstance(error, ModelCircuitOpenError):
                fault_code = FaultCode.MODEL_CIRCUIT_OPEN
            else:
                fault_code = FaultCode.MODEL_RETRY_EXHAUSTED
            self.fault_logger.record(
                component=FaultComponent.MODEL,
                code=fault_code,
                error=error,
                recovery=RecoveryAction.DETERMINISTIC_DIAGNOSIS,
            )
            return PreparedDiagnosis(
                base_state=previous_state,
                updated_state=updated_state,
                report=deterministic_report.model_copy(
                    update={
                        "delivery_mode": DeliveryMode.MODEL_FALLBACK,
                        "delivery_message_zh": (
                            "在线解释暂时不可用，已自动切换为确定性本地诊断，学习记录已安全保存。"
                        ),
                    }
                ),
            )

        locked_report = _lock_deterministic_fields(
            deterministic_report,
            model_report,
            first_wrong=first_wrong,
        ).model_copy(
            update={
                "delivery_mode": DeliveryMode.MODEL_ENHANCED,
                "delivery_message_zh": "本次在确定性判题基础上生成了在线解释。",
            }
        )
        return PreparedDiagnosis(
            base_state=previous_state,
            updated_state=updated_state,
            report=locked_report,
        )


def _grade_sequence(
    submission: str,
    expected: list[object],
    *,
    tolerance: float,
    misconception_candidates: list[str],
) -> GradeResult:
    try:
        actual = json.loads(submission)
    except json.JSONDecodeError:
        return GradeResult(
            score=0.0,
            is_correct=False,
            errors=["列表答案必须使用 JSON 格式，例如 [46.08, 53.92]。"],
            misconception_candidates=misconception_candidates,
        )
    if not isinstance(actual, list):
        return GradeResult(
            score=0.0,
            is_correct=False,
            errors=["这道题需要提交一个列表答案。"],
            misconception_candidates=misconception_candidates,
        )
    return grade_dataframe_result(
        pd.DataFrame([actual]),
        pd.DataFrame([expected]),
        absolute_tolerance=tolerance,
        misconception_candidates=misconception_candidates,
    )


def _build_deterministic_report(
    context: TutorContext,
    grade: GradeResult,
    state: LearningState,
    decision: NextQuestionDecision,
    *,
    submission: LearnerSubmission,
    hint_level: int,
    knowledge_result: RagSearchResult,
) -> DiagnosticReport:
    question = _find_question(context, context.current_question_id)
    primary_dimension = max(
        CapabilityDimension,
        key=lambda dimension: getattr(question.dimension_weights, dimension.value),
    )
    answer_evidence = f"学习者答案：{_submission_excerpt(submission.answer)}"
    evidence = [answer_evidence, *grade.evidence]
    if submission.reasoning.strip():
        evidence.append(f"学习者思考过程：{_submission_excerpt(submission.reasoning)}")
    if submission.python_code.strip():
        evidence.append(
            f"学习者提交的代码文本：{_submission_excerpt(submission.python_code)}"
        )

    recommendation = recommend_from_findings(question, grade, decision)
    reasoning_assessment = assess_reasoning(question, submission, grade.findings)
    if grade.answer_is_correct:
        if reasoning_assessment.verdict == EvidenceVerdict.SUPPORTS:
            reasoning_feedback = "理由也包含了与正确依据一致的统计关系。"
        elif reasoning_assessment.verdict in {
            EvidenceVerdict.CONTRADICTS,
            EvidenceVerdict.INSUFFICIENT,
        }:
            reasoning_feedback = (
                "答案判定不受影响；理由仍需要根据单独诊断继续完善。"
            )
        else:
            reasoning_feedback = reasoning_assessment.message_zh
        feedback = (
            "确定性判题显示答案正确。"
            f"{reasoning_feedback}"
            f"本题主要观察{CAPABILITY_LABELS_ZH[primary_dimension]}维度。"
        )
        uncertainty = "无：答案正确性只由答案通道的确定性规则判定。"
    else:
        feedback = _guiding_feedback(primary_dimension)
        if isinstance(question.expected_answer, str):
            uncertainty = "不确定：仅凭当前自由文本无法判断完整理解，请补充你的理由。"
        else:
            uncertainty = "无：数值或结构差异已由确定性规则识别。"

    return DiagnosticReport(
        question_id=question.id,
        overall_correctness=(
            grade.answer_score if grade.answer_score is not None else grade.score
        ),
        dimension_scores=state.mastery[question.concept_id],
        evidence=evidence,
        learner_evidence=_learner_evidence(submission),
        grader_findings=grade.findings,
        reasoning_assessment=reasoning_assessment,
        misconception_tags=grade.misconception_candidates,
        feedback=feedback,
        hint_level=hint_level,
        recommended_action=recommendation.action_zh,
        recommendation_kind=recommendation.kind,
        recommendation_rule_id=recommendation.source_rule_id,
        recommendation_dimension=recommendation.target_dimension,
        next_question_id=recommendation.next_question_id,
        uncertainty=uncertainty,
        knowledge_context_status=knowledge_result.status,
        knowledge_context_message=knowledge_result.message_zh,
        knowledge_citations=list(knowledge_result.citations),
    )


def _guiding_feedback(dimension: CapabilityDimension) -> str:
    prompts = {
        CapabilityDimension.CONCEPT: "先想一想：题目中的哪个统计概念决定了应选的方法？",
        CapabilityDimension.CALCULATION: "先检查一步：你使用的公式和每个代入值分别是什么？",
        CapabilityDimension.PYTHON: "先观察代码：调用了哪个函数，它的默认参数会怎样影响结果？",
        CapabilityDimension.INTERPRETATION: "先回到数据情境：这个数值能说明什么，又不能说明什么？",
    }
    return f"本次主要需要检查{CAPABILITY_LABELS_ZH[dimension]}维度。{prompts[dimension]}"


def _lock_deterministic_fields(
    deterministic: DiagnosticReport,
    model_report: DiagnosticReport,
    *,
    first_wrong: bool,
) -> DiagnosticReport:
    feedback = deterministic.feedback if first_wrong else model_report.feedback
    uncertainty = (
        deterministic.uncertainty
        if deterministic.uncertainty.startswith("不确定")
        else model_report.uncertainty
    )
    return deterministic.model_copy(update={"feedback": feedback, "uncertainty": uncertainty})


def _find_question(context: TutorContext, question_id: str) -> Question:
    try:
        return next(question for question in context.questions if question.id == question_id)
    except StopIteration as error:
        raise ValueError(f"题目不存在：{question_id}") from error


def _submission_excerpt(submission: str) -> str:
    normalized = submission.strip()
    if len(normalized) <= 500:
        return normalized
    return f"{normalized[:499]}…"


def _learner_evidence(submission: LearnerSubmission) -> list[LearnerEvidence]:
    authored_fields = (
        (SubmissionField.ANSWER, submission.answer),
        (SubmissionField.REASONING, submission.reasoning),
        (SubmissionField.PYTHON_CODE, submission.python_code),
    )
    return [
        LearnerEvidence(source=source, quote=excerpt)
        for source, text in authored_fields
        if (excerpt := _submission_excerpt(text))
    ]
