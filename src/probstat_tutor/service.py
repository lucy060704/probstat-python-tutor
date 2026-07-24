"""Application service used by Streamlit without embedding business logic in the page."""

import hashlib
import json
from dataclasses import dataclass

from probstat_tutor.config import Settings, get_settings
from probstat_tutor.curriculum import load_default_question_bank
from probstat_tutor.policy import select_next_question
from probstat_tutor.schemas import (
    ConceptId,
    DiagnosticReport,
    LearningState,
    PolicyStatus,
    Question,
)
from probstat_tutor.storage import LearningStateStore
from probstat_tutor.tutor_agent import TutorAgent


class LearningServiceError(RuntimeError):
    """A safe, learner-facing service error."""


@dataclass(frozen=True)
class LearningDashboard:
    """Data needed by the left side of the single-page UI."""

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

    def choose_question(self, learner_id: str, concept_id: ConceptId) -> Question:
        candidates = [question for question in self.questions if question.concept_id == concept_id]
        decision = select_next_question(self.store.load(learner_id), candidates)
        if decision.status != PolicyStatus.QUESTION or decision.question_id is None:
            raise LearningServiceError(decision.reason)
        return self.get_question(decision.question_id)

    def get_question(self, question_id: str) -> Question:
        try:
            return next(question for question in self.questions if question.id == question_id)
        except StopIteration as error:
            raise LearningServiceError(f"找不到题目：{question_id}") from error

    def get_hint(self, question_id: str, hint_level: int) -> str:
        question = self.get_question(question_id)
        level = min(3, max(1, hint_level))
        hints = {
            1: f"概念提示：先判断这道题主要考查 {question.concept_id.value} 的哪个基本含义。",
            2: "方法提示：写出你准备使用的统计量、公式或 Python 函数，再代入数据。",
            3: "步骤提示：先完成第一个中间结果，并检查单位、参数和统计口径是否一致。",
        }
        return hints[level]

    async def submit(
        self,
        *,
        learner_id: str,
        session_id: str,
        question_id: str,
        answer: str,
        reasoning: str,
        python_code: str,
        hint_level: int,
    ) -> DiagnosticReport:
        submission_key = _submission_key(
            learner_id=learner_id,
            question_id=question_id,
            answer=answer,
            reasoning=reasoning,
            python_code=python_code,
            hint_level=hint_level,
        )
        cached = self.store.load_submission_report(learner_id, submission_key)
        if cached is not None:
            return cached

        try:
            context = self.tutor.create_context(
                learner_id=learner_id,
                session_id=session_id,
                current_question_id=question_id,
            )
            report = await self.tutor.diagnose(
                context,
                answer,
                hint_level=hint_level,
                reasoning=reasoning,
                python_code=python_code,
            )
        except Exception as error:
            raise LearningServiceError(
                "教学服务暂时不可用，请稍后重试。你的输入仍保留在页面中。"
            ) from error

        self.store.save_submission_report(learner_id, submission_key, report)
        return report

    def reset_demo_learner(self, learner_id: str) -> None:
        self.store.reset(learner_id)


def _submission_key(
    *,
    learner_id: str,
    question_id: str,
    answer: str,
    reasoning: str,
    python_code: str,
    hint_level: int,
) -> str:
    payload = json.dumps(
        {
            "learner_id": learner_id,
            "question_id": question_id,
            "answer": answer,
            "reasoning": reasoning,
            "python_code": python_code,
            "hint_level": hint_level,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
