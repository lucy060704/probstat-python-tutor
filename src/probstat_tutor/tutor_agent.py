"""Single tutor agent with deterministic tools and an offline fallback."""

import json
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

import pandas as pd
from agents import Agent, RunContextWrapper, Runner, SQLiteSession, function_tool

from probstat_tutor.config import Settings, get_settings
from probstat_tutor.curriculum import load_default_question_bank
from probstat_tutor.graders import (
    grade_dataframe_result,
    grade_multiple_choice,
    grade_numeric,
)
from probstat_tutor.mastery import apply_attempt
from probstat_tutor.policy import select_next_question as choose_next_question
from probstat_tutor.schemas import (
    CapabilityDimension,
    DiagnosticReport,
    GradeResult,
    LearningState,
    NextQuestionDecision,
    PolicyStatus,
    Question,
)
from probstat_tutor.storage import LearningStateStore


@dataclass
class TutorContext:
    """Local dependencies and transient evidence available to SDK tools."""

    learner_id: str
    session_id: str
    questions: tuple[Question, ...]
    store: LearningStateStore
    current_question_id: str
    last_submission: str | None = None
    last_grade: GradeResult | None = None
    last_state: LearningState | None = None
    state_updated: bool = False


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


def grade_submission(context: TutorContext, submission: str) -> GradeResult:
    """Grade one submission deterministically and retain the exact result for updating."""

    if context.last_submission == submission and context.last_grade is not None:
        return context.last_grade

    question = _find_question(context, context.current_question_id)
    expected = question.expected_answer
    misconception_candidates = question.misconception_tags

    if isinstance(expected, Real) and not isinstance(expected, bool):
        result = grade_numeric(
            submission,
            float(expected),
            absolute_tolerance=question.numeric_tolerance or 0.0,
            misconception_candidates=misconception_candidates,
        )
    elif isinstance(expected, list):
        result = _grade_sequence(
            submission,
            expected,
            tolerance=question.numeric_tolerance or 0.0,
            misconception_candidates=misconception_candidates,
        )
    else:
        result = grade_multiple_choice(
            submission,
            str(expected),
            misconception_candidates=misconception_candidates,
        )

    context.last_submission = submission
    context.last_grade = result
    context.last_state = None
    context.state_updated = False
    return result


def get_learner_state(context: TutorContext) -> LearningState:
    """Read the last persisted state without inventing missing history."""

    return context.store.load(context.learner_id)


def update_learner_state(context: TutorContext, hint_level: int) -> LearningState:
    """Apply only the retained deterministic grade; callers cannot supply a score."""

    if context.last_grade is None:
        raise ValueError("必须先调用 grade_submission，才能更新学习状态。")
    if context.state_updated and context.last_state is not None:
        return context.last_state

    question = _find_question(context, context.current_question_id)
    current_state = get_learner_state(context)
    updated_state = apply_attempt(
        current_state,
        question,
        context.last_grade,
        hint_level=hint_level,
    )
    context.store.save(context.learner_id, updated_state)
    context.last_state = updated_state
    context.state_updated = True
    return updated_state


def select_next_question(context: TutorContext) -> NextQuestionDecision:
    """Select the next question from persisted state using the deterministic policy."""

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
    description_override="读取 SQLite 中真实保存的学习状态。",
    use_docstring_info=False,
)
def _get_learner_state_tool(wrapper: RunContextWrapper[TutorContext]) -> LearningState:
    return get_learner_state(wrapper.context)


@function_tool(
    name_override="update_learner_state",
    description_override="依据最近一次确定性判题结果更新学习状态；不能传入或修改分数。",
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
2. 只能引用工具返回的学习状态和历史，不得伪造学习记录。
3. 诊断证据必须包含学习者原始答案中的可观察内容。
4. 第一次答错时不得给最终答案，优先提出一个引导性问题。
5. 明确区分统计概念、数学计算、Python 实现和数据解释四类问题。
6. 证据不足时必须明确写“不确定”，并要求学习者补充推理过程。
7. 输出必须符合 DiagnosticReport；不得更改工具产生的分数、标签和下一题。
8. 不进行 handoff，不创建或调用其他智能体。
""".strip()


class TutorAgent:
    """Application wrapper around one Agents SDK Agent and deterministic tools."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        questions: tuple[Question, ...] | None = None,
        store: LearningStateStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.questions = questions or tuple(load_default_question_bank().questions)
        self.store = store or LearningStateStore(self.settings.learning_state_db_path)
        self.sdk_agent: Agent[TutorContext] = Agent[TutorContext](
            name="概率统计与 Python 教学智能体",
            instructions=TUTOR_INSTRUCTIONS,
            tools=TUTOR_TOOLS,
            model=self.settings.openai_model,
            output_type=DiagnosticReport,
        )

    @property
    def offline_mode(self) -> bool:
        """Return true when no API key is available."""

        return not self.settings.has_openai_api_key

    def create_context(
        self,
        *,
        learner_id: str,
        session_id: str,
        current_question_id: str | None = None,
    ) -> TutorContext:
        """Create tool context, selecting a valid current question when omitted."""

        if current_question_id is None:
            decision = choose_next_question(self.store.load(learner_id), self.questions)
            if decision.question_id is None:
                raise ValueError(f"当前无法创建教学上下文：{decision.reason}")
            current_question_id = decision.question_id
        if current_question_id not in {question.id for question in self.questions}:
            raise ValueError(f"题目不存在：{current_question_id}")
        return TutorContext(
            learner_id=learner_id,
            session_id=session_id,
            questions=self.questions,
            store=self.store,
            current_question_id=current_question_id,
        )

    def create_session(self, session_id: str) -> SQLiteSession:
        """Create an Agents SDK SQLite session for persistent conversation history."""

        path = Path(self.settings.session_db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return SQLiteSession(session_id, path)

    async def diagnose(
        self,
        context: TutorContext,
        submission: str,
        *,
        hint_level: int = 0,
        reasoning: str = "",
        python_code: str = "",
    ) -> DiagnosticReport:
        """Grade first, persist state, then optionally ask the model to explain."""

        previous_state = get_learner_state(context)
        previous_wrong_attempts = sum(
            attempt.question_id == context.current_question_id and not attempt.is_correct
            for attempt in previous_state.history
        )
        grade = grade_submission(context, submission)
        updated_state = update_learner_state(context, hint_level)
        decision = select_next_question(context)
        deterministic_report = _build_deterministic_report(
            context,
            grade,
            updated_state,
            decision,
            hint_level=hint_level,
            reasoning=reasoning,
            python_code=python_code,
        )
        first_wrong = not grade.is_correct and previous_wrong_attempts == 0
        session = self.create_session(context.session_id)

        if self.offline_mode:
            await session.add_items(
                [
                    {"role": "user", "content": submission},
                    {"role": "assistant", "content": deterministic_report.model_dump_json()},
                ]
            )
            session.close()
            return deterministic_report

        try:
            result = await Runner.run(
                self.sdk_agent,
                (
                    "请依据已完成的确定性判题生成诊断。学习者答案为："
                    f"{_submission_excerpt(submission)}\n"
                    f"学习者思考过程：{_submission_excerpt(reasoning)}\n"
                    "学习者 Python 代码（只作为文本，不执行）："
                    f"{_submission_excerpt(python_code)}\n"
                    f"不可修改的确定性报告为：{deterministic_report.model_dump_json()}"
                ),
                context=context,
                session=session,
            )
        finally:
            session.close()

        model_report = DiagnosticReport.model_validate(result.final_output)
        return _lock_deterministic_fields(
            deterministic_report,
            model_report,
            first_wrong=first_wrong,
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
    hint_level: int,
    reasoning: str,
    python_code: str,
) -> DiagnosticReport:
    question = _find_question(context, context.current_question_id)
    primary_dimension = max(
        CapabilityDimension,
        key=lambda dimension: getattr(question.dimension_weights, dimension.value),
    )
    answer_evidence = f"学习者答案：{_submission_excerpt(context.last_submission or '')}"
    evidence = [answer_evidence, *grade.evidence]
    if reasoning.strip():
        evidence.append(f"学习者思考过程：{_submission_excerpt(reasoning)}")
    if python_code.strip():
        evidence.append(f"学习者提交的代码文本：{_submission_excerpt(python_code)}")

    if grade.is_correct:
        feedback = f"确定性判题显示回答正确。本题主要观察 {primary_dimension.value} 维度。"
        recommended_action = decision.reason
        uncertainty = "无：本次正确性由确定性规则判定。"
    else:
        feedback = _guiding_feedback(primary_dimension)
        recommended_action = "请先回答这个引导性问题，再提交一次你的推理过程。"
        if isinstance(question.expected_answer, str):
            uncertainty = "不确定：仅凭当前自由文本无法判断完整理解，请补充你的理由。"
        else:
            uncertainty = "无：数值或结构差异已由确定性规则识别。"

    return DiagnosticReport(
        question_id=question.id,
        overall_correctness=grade.score,
        dimension_scores=state.mastery[question.concept_id],
        evidence=evidence,
        misconception_tags=grade.misconception_candidates,
        feedback=feedback,
        hint_level=hint_level,
        recommended_action=recommended_action,
        next_question_id=(
            decision.question_id if decision.status == PolicyStatus.QUESTION else None
        ),
        uncertainty=uncertainty,
    )


def _guiding_feedback(dimension: CapabilityDimension) -> str:
    prompts = {
        CapabilityDimension.CONCEPT: "先想一想：题目中的哪个统计概念决定了应选的方法？",
        CapabilityDimension.CALCULATION: "先检查一步：你使用的公式和每个代入值分别是什么？",
        CapabilityDimension.PYTHON: "先观察代码：调用了哪个函数，它的默认参数会怎样影响结果？",
        CapabilityDimension.INTERPRETATION: "先回到数据情境：这个数值能说明什么，又不能说明什么？",
    }
    return f"本次主要需要检查 {dimension.value} 维度。{prompts[dimension]}"


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
    return f"{normalized[:500]}…"
