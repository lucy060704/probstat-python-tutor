"""Application service used by Streamlit without embedding business logic in the page."""

import hashlib
import json
import secrets

from pydantic import BaseModel, ConfigDict, ValidationError

from probstat_tutor.analytics import TeacherDashboard, build_teacher_dashboard
from probstat_tutor.config import Settings, get_settings
from probstat_tutor.curriculum import load_default_question_bank
from probstat_tutor.policy import select_next_question
from probstat_tutor.rag import RagSearchResult, RetrievalPurpose
from probstat_tutor.schemas import (
    ConceptId,
    DiagnosticReport,
    LearnerSubmission,
    LearningState,
    LearningSubmissionRequest,
    PolicyStatus,
    Question,
)
from probstat_tutor.storage import CommitSubmissionStatus, LearningStateStore
from probstat_tutor.tutor_agent import TutorAgent

MAX_COMMIT_ATTEMPTS = 3


class LearningServiceError(RuntimeError):
    """A safe, learner-facing service error."""


class LearningRecommendationUnavailableError(LearningServiceError):
    """No question can currently be selected under the learning policy."""


class LearningServiceUnavailableError(LearningServiceError):
    """A temporary infrastructure failure prevented the service operation."""


class LearningIdempotencyConflictError(LearningServiceError):
    """One idempotency key was reused for a different logical submission."""


class LearningStateConflictError(LearningServiceError):
    """Concurrent learning-state changes exhausted bounded commit retries."""


class LearningDashboard(BaseModel):
    """Data needed by the left side of the single-page UI."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    state: LearningState
    selected_mastery: dict[str, float]
    recent_records: tuple[dict[str, object], ...]


class LearningService:
    """Coordinate curriculum, persistence, deterministic tools, and TutorAgent."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        store: LearningStateStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.questions = tuple(load_default_question_bank().questions)
        self.store = store or LearningStateStore(self.settings.learning_state_db_path)
        self.tutor = TutorAgent(
            settings=self.settings,
            questions=self.questions,
            store=self.store,
        )

    @property
    def offline_mode(self) -> bool:
        return self.tutor.offline_mode

    @staticmethod
    def create_anonymous_learner_id() -> str:
        """Create a local random profile identifier that does not ask for a name."""

        return f"local_anon_{secrets.token_hex(8)}"

    def get_dashboard(self, learner_id: str, concept_id: ConceptId) -> LearningDashboard:
        state = self.store.load(learner_id)
        mastery = state.mastery[concept_id]
        recent = tuple(
            {
                "question_id": attempt.question_id,
                "score": attempt.score,
                "is_correct": attempt.is_correct,
                "hint_level": attempt.hint_level,
            }
            for attempt in state.history[-5:]
        )
        return LearningDashboard(
            state=state,
            selected_mastery={
                "concept": mastery.concept,
                "calculation": mastery.calculation,
                "python": mastery.python,
                "interpretation": mastery.interpretation,
            },
            recent_records=recent,
        )

    def get_teacher_dashboard(self) -> TeacherDashboard:
        """Return only k-anonymous aggregates derived from state-only storage rows."""

        return build_teacher_dashboard(self.store.load_anonymized_states())

    def choose_question(self, learner_id: str, concept_id: ConceptId) -> Question:
        candidates = [question for question in self.questions if question.concept_id == concept_id]
        try:
            state = self.store.load(learner_id)
        except Exception as error:
            raise LearningServiceUnavailableError(
                "教学服务暂时不可用，请稍后重试。"
            ) from error

        decision = select_next_question(state, candidates)
        if decision.status != PolicyStatus.QUESTION or decision.question_id is None:
            raise LearningRecommendationUnavailableError(decision.reason)
        return self.get_question(decision.question_id)

    def get_question(self, question_id: str) -> Question:
        try:
            return next(question for question in self.questions if question.id == question_id)
        except StopIteration as error:
            raise LearningServiceError(f"找不到题目：{question_id}") from error

    def get_hint(self, question_id: str, hint_level: int) -> str:
        question = self.get_question(question_id)
        level = min(4, max(1, hint_level))
        if question.hints is not None:
            return question.hints.for_level(level)
        hints = {
            1: f"概念提示：先判断这道题主要考查 {question.concept_id.value} 的哪个基本含义。",
            2: "方法提示：写出你准备使用的统计量、公式或 Python 函数，再代入数据。",
            3: "步骤提示：先完成第一个中间结果，并检查单位、参数和统计口径是否一致。",
            4: "完整解释：回到定义，依次核对统计口径、计算过程、Python 表达和情境结论。",
        }
        return hints[level]

    def retrieve_knowledge(
        self,
        query_text: str,
        *,
        concept_id: ConceptId | None = None,
        knowledge_node_ids: tuple[str, ...] = (),
        disclosure_level: int = 1,
        top_k: int = 3,
    ) -> RagSearchResult:
        """Expose the bounded local retriever without involving a language model."""

        try:
            return self.tutor.retrieve_knowledge(
                text=query_text,
                concept_id=concept_id,
                knowledge_node_ids=knowledge_node_ids,
                disclosure_level=disclosure_level,
                top_k=top_k,
                purpose=RetrievalPurpose.KNOWLEDGE_SEARCH,
            )
        except ValidationError as error:
            message = _first_chinese_validation_message(error)
            raise LearningServiceError(f"检索内容无效：{message}。") from None

    async def submit(
        self,
        *,
        learner_id: str,
        session_id: str,
        question_id: str,
        answer: str,
        reasoning: str = "",
        python_code: str = "",
        hint_level: int,
        idempotency_key: str | None = None,
    ) -> DiagnosticReport:
        command = _validate_submission_request(
            learner_id=learner_id,
            session_id=session_id,
            question_id=question_id,
            answer=answer,
            reasoning=reasoning,
            python_code=python_code,
            hint_level=hint_level,
            idempotency_key=idempotency_key,
        )
        try:
            request_fingerprint = _request_fingerprint(command)
            submission_key = _submission_key(command)
            cached = self.store.load_submission_receipt(
                command.learner_id, submission_key
            )
            if cached is not None:
                if not cached.matches(
                    request_fingerprint=request_fingerprint,
                    submission_key=submission_key,
                ):
                    raise LearningIdempotencyConflictError(
                        "同一幂等键不能用于不同的提交内容。"
                    )
                return cached.report

            for _attempt in range(MAX_COMMIT_ATTEMPTS):
                base_state = self.store.load(command.learner_id)
                context = self.tutor.create_context(
                    learner_id=command.learner_id,
                    session_id=command.session_id,
                    current_question_id=command.question_id,
                    base_state=base_state,
                )
                prepared = await self.tutor.diagnose(
                    context,
                    command.submission,
                    hint_level=command.hint_level,
                )
                committed = self.store.commit_submission(
                    learner_id=command.learner_id,
                    submission_key=submission_key,
                    request_fingerprint=request_fingerprint,
                    expected_state=prepared.base_state,
                    updated_state=prepared.updated_state,
                    report=prepared.report,
                )
                if committed.status == CommitSubmissionStatus.CONFLICT:
                    continue
                if committed.status == CommitSubmissionStatus.IDEMPOTENCY_CONFLICT:
                    raise LearningIdempotencyConflictError(
                        "同一幂等键不能用于不同的提交内容。"
                    )
                if committed.report is None:
                    raise RuntimeError("提交事务没有返回诊断报告")
                return committed.report
        except LearningIdempotencyConflictError:
            raise
        except Exception as error:
            raise LearningServiceUnavailableError(
                "教学服务暂时不可用，请稍后重试。你的输入仍保留在页面中。"
            ) from error

        raise LearningStateConflictError("学习状态刚刚发生变化，请重新提交本次答案。")

    def reset_demo_learner(self, learner_id: str) -> None:
        self.store.reset(learner_id)


def _submission_key(command: LearningSubmissionRequest) -> str:
    if command.idempotency_key is not None:
        idempotency_payload = json.dumps(
            {
                "learner_id": command.learner_id,
                "idempotency_key": command.idempotency_key,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(idempotency_payload.encode("utf-8")).hexdigest()

    return _request_fingerprint(command)


def _request_fingerprint(command: LearningSubmissionRequest) -> str:
    payload = json.dumps(
        {
            "learner_id": command.learner_id,
            "question_id": command.question_id,
            "answer": command.submission.answer,
            "reasoning": command.submission.reasoning,
            "python_code": command.submission.python_code,
            "hint_level": command.hint_level,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_submission_request(
    *,
    learner_id: object,
    session_id: object,
    question_id: object,
    answer: object,
    reasoning: object,
    python_code: object,
    hint_level: object,
    idempotency_key: object = None,
) -> LearningSubmissionRequest:
    try:
        return LearningSubmissionRequest(
            learner_id=learner_id,
            session_id=session_id,
            question_id=question_id,
            submission=LearnerSubmission(
                answer=answer,
                reasoning=reasoning,
                python_code=python_code,
            ),
            hint_level=hint_level,
            idempotency_key=idempotency_key,
        )
    except ValidationError as error:
        message = _first_chinese_validation_message(error)
        raise LearningServiceError(f"提交内容无效：{message}。") from None


def _first_chinese_validation_message(error: ValidationError) -> str:
    for issue in error.errors():
        context = issue.get("ctx")
        if isinstance(context, dict):
            cause = context.get("error")
            if cause is not None:
                message = str(cause).strip()
                if message:
                    return message.rstrip("。")
    return "请检查答案、思考过程和 Python 代码是否为有效文本"
