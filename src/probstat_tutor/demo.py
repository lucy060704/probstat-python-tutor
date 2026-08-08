"""Reproducible offline demonstrations used by local stage gates."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from probstat_tutor.config import Settings
from probstat_tutor.schemas import ConceptId
from probstat_tutor.service import LearningService
from probstat_tutor.storage import LearningStateStore


class G1DemoRun(BaseModel):
    """Evidence from one complete wrong-answer, retry, and idempotency journey."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_number: int = Field(ge=1)
    learner_id: str = Field(min_length=1)
    selected_question_id: str = Field(min_length=1)
    hint_is_progressive: bool
    wrong_answer_is_rejected: bool
    wrong_answer_hides_next_question: bool
    correction_is_accepted: bool
    correction_has_next_question: bool
    duplicate_is_idempotent: bool
    history_count: int = Field(ge=0)
    passed: bool


class G1DemoSummary(BaseModel):
    """Machine-readable result from five consecutive offline journeys."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    offline_mode: bool
    requested_runs: int = Field(ge=1)
    passed_runs: int = Field(ge=0)
    all_passed: bool
    runs: tuple[G1DemoRun, ...]


async def run_g1_offline_demo(
    workdir: Path,
    *,
    repetitions: int = 5,
) -> G1DemoSummary:
    """Run the same learner-visible G1 loop repeatedly without network or API keys."""

    if repetitions < 1:
        raise ValueError("离线演示次数必须至少为 1")
    workdir.mkdir(parents=True, exist_ok=True)
    settings = Settings(
        openai_api_key=None,
        openai_model=None,
        session_db_path=workdir / "sessions.sqlite3",
        learning_state_db_path=workdir / "learning.sqlite3",
    )
    service = LearningService(
        settings=settings,
        store=LearningStateStore(settings.learning_state_db_path),
    )
    runs = tuple(
        [await _run_g1_journey(service, run_number) for run_number in range(1, repetitions + 1)]
    )
    passed_runs = sum(run.passed for run in runs)
    return G1DemoSummary(
        offline_mode=service.offline_mode,
        requested_runs=repetitions,
        passed_runs=passed_runs,
        all_passed=service.offline_mode and passed_runs == repetitions,
        runs=runs,
    )


async def _run_g1_journey(
    service: LearningService,
    run_number: int,
) -> G1DemoRun:
    learner_id = f"g1-demo-{run_number}"
    session_id = f"g1-demo-session-{run_number}"
    question = service.choose_question(learner_id, ConceptId.MEAN_MEDIAN)
    hint = service.get_hint(question.id, 1)
    wrong_report = await service.submit(
        learner_id=learner_id,
        session_id=session_id,
        question_id=question.id,
        answer="mean",
        reasoning="无论有没有异常值，均值都最有代表性。",
        hint_level=1,
    )
    correct_report = await service.submit(
        learner_id=learner_id,
        session_id=session_id,
        question_id=question.id,
        answer="median",
        reasoning="极端值会明显拉高均值，中位数更适合描述典型水平。",
        hint_level=1,
    )
    duplicate_report = await service.submit(
        learner_id=learner_id,
        session_id=session_id,
        question_id=question.id,
        answer="median",
        reasoning="极端值会明显拉高均值，中位数更适合描述典型水平。",
        hint_level=1,
    )
    dashboard = service.get_dashboard(learner_id, ConceptId.MEAN_MEDIAN)
    history_count = len(dashboard.state.history)
    checks = (
        hint.startswith("概念提示："),
        wrong_report.overall_correctness == 0.0,
        wrong_report.next_question_id is None,
        correct_report.overall_correctness == 1.0,
        correct_report.next_question_id is not None,
        duplicate_report == correct_report and history_count == 2,
    )
    return G1DemoRun(
        run_number=run_number,
        learner_id=learner_id,
        selected_question_id=question.id,
        hint_is_progressive=checks[0],
        wrong_answer_is_rejected=checks[1],
        wrong_answer_hides_next_question=checks[2],
        correction_is_accepted=checks[3],
        correction_has_next_question=checks[4],
        duplicate_is_idempotent=checks[5],
        history_count=history_count,
        passed=all(checks),
    )
