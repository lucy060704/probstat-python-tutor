"""Tests for tutor tools, SQLite persistence, and offline mode."""

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from probstat_tutor.config import Settings
from probstat_tutor.schemas import (
    DiagnosticReport,
    LearnerSubmission,
    PolicyStatus,
    RecommendationKind,
    SubmissionField,
)
from probstat_tutor.storage import LearningStateStore
from probstat_tutor.tutor_agent import (
    TUTOR_TOOLS,
    TutorAgent,
    get_current_question,
    get_learner_state,
    grade_submission,
    select_next_question,
    update_learner_state,
)


@pytest.fixture
def tutor(tmp_path: Path) -> TutorAgent:
    settings = Settings(
        openai_api_key=None,
        openai_model=None,
        session_db_path=tmp_path / "sessions.sqlite3",
        learning_state_db_path=tmp_path / "learning.sqlite3",
    )
    return TutorAgent(settings=settings, store=LearningStateStore(settings.learning_state_db_path))


def test_sdk_agent_exposes_exactly_five_required_tools(tutor: TutorAgent) -> None:
    tool_names = {tool.name for tool in TUTOR_TOOLS}

    assert tool_names == {
        "get_current_question",
        "grade_submission",
        "get_learner_state",
        "update_learner_state",
        "select_next_question",
    }
    assert tutor.sdk_agent.handoffs == []
    assert tutor.sdk_agent.output_type is DiagnosticReport


def test_model_name_comes_from_openai_model_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "configured-test-model")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = Settings(
        session_db_path=tmp_path / "sessions.sqlite3",
        learning_state_db_path=tmp_path / "learning.sqlite3",
    )
    tutor = TutorAgent(settings=settings)

    assert settings.openai_model == "configured-test-model"
    assert tutor.sdk_agent.model == "configured-test-model"
    assert tutor.offline_mode is True


def test_model_reliability_limits_come_from_bounded_environment_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("MODEL_TIMEOUT_SECONDS", "1.5")
    monkeypatch.setenv("MODEL_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("MODEL_RETRY_BASE_DELAY_SECONDS", "0.2")
    monkeypatch.setenv("MODEL_CIRCUIT_FAILURE_THRESHOLD", "4")
    monkeypatch.setenv("MODEL_CIRCUIT_OPEN_SECONDS", "12")
    settings = Settings(
        session_db_path=tmp_path / "sessions.sqlite3",
        learning_state_db_path=tmp_path / "learning.sqlite3",
    )
    tutor = TutorAgent(settings=settings)

    assert tutor.model_reliability.timeout_seconds == 1.5
    assert tutor.model_reliability.max_attempts == 3
    assert tutor.model_reliability.retry_base_delay_seconds == 0.2
    assert tutor.model_reliability.failure_threshold == 4
    assert tutor.model_reliability.open_seconds == 12.0


def test_current_question_hides_answer_and_grading_rules(tutor: TutorAgent) -> None:
    context = tutor.create_context(learner_id="learner", session_id="session")

    question = get_current_question(context)

    assert question["id"] == "mean_median_concept_01"
    assert "expected_answer" not in question
    assert "rubric" not in question


def test_tools_grade_then_stage_state_without_persisting(tutor: TutorAgent) -> None:
    context = tutor.create_context(
        learner_id="learner",
        session_id="session",
        current_question_id="mean_median_python_01",
    )

    grade = grade_submission(
        context,
        LearnerSubmission(answer="8", python_code='df["value"].median()'),
    )
    updated = update_learner_state(context, hint_level=0)
    staged = get_learner_state(context)
    persisted = tutor.store.load("learner")
    decision = select_next_question(context)

    assert grade.is_correct is True
    assert updated == staged
    assert persisted.history == ()
    assert "mean_median_python_01" in staged.completed_question_ids
    assert staged.history[-1].score == grade.score
    assert decision.status in {PolicyStatus.QUESTION, PolicyStatus.BLOCKED}


def test_update_requires_deterministic_grade_first(tutor: TutorAgent) -> None:
    context = tutor.create_context(learner_id="learner", session_id="session")

    with pytest.raises(ValueError, match="必须先调用 grade_submission"):
        update_learner_state(context, hint_level=0)


def test_repeated_update_is_idempotent(tutor: TutorAgent) -> None:
    context = tutor.create_context(
        learner_id="learner",
        session_id="session",
        current_question_id="mean_median_python_01",
    )
    grade_submission(
        context,
        LearnerSubmission(answer="8", python_code='df["value"].median()'),
    )

    first = update_learner_state(context, hint_level=0)
    second = update_learner_state(context, hint_level=0)

    assert first == second
    assert len(second.history) == 1
    assert tutor.store.load("learner").history == ()


def test_offline_mode_never_calls_runner_and_returns_locked_report(tutor: TutorAgent) -> None:
    context = tutor.create_context(
        learner_id="offline-learner",
        session_id="offline-session",
        current_question_id="mean_median_concept_01",
    )

    with patch(
        "probstat_tutor.tutor_agent.Runner.run",
        side_effect=AssertionError("离线模式不应调用模型"),
    ):
        prepared = asyncio.run(
            tutor.diagnose(
                context,
                LearnerSubmission(
                    answer="mean",
                    reasoning="我认为平均数就是中位数。",
                    python_code='df["value"].mean()',
                ),
                hint_level=0,
            )
        )
    report = prepared.report

    assert report.question_id == "mean_median_concept_01"
    assert report.overall_correctness == 0.0
    assert any("学习者答案：mean" == evidence for evidence in report.evidence)
    assert [(item.source, item.quote) for item in report.learner_evidence] == [
        (SubmissionField.ANSWER, "mean"),
        (SubmissionField.REASONING, "我认为平均数就是中位数。"),
        (SubmissionField.PYTHON_CODE, 'df["value"].mean()'),
    ]
    assert "?" in report.feedback or "？" in report.feedback
    assert "median" not in report.feedback.casefold()
    assert report.uncertainty.startswith("不确定")
    assert report.recommendation_kind == RecommendationKind.RETRY_INSUFFICIENT
    assert report.recommendation_rule_id is not None
    assert report.next_question_id is None
    assert prepared.updated_state.history[-1].score == report.overall_correctness
    assert tutor.store.load("offline-learner").history == ()


def test_diagnose_does_not_create_sdk_session_database(tutor: TutorAgent) -> None:
    session_path = Path(tutor.settings.session_db_path)
    context = tutor.create_context(
        learner_id="offline-learner",
        session_id="persisted-session",
        current_question_id="mean_median_python_01",
    )
    assert not session_path.exists()

    asyncio.run(
        tutor.diagnose(
            context,
            LearnerSubmission(answer="8"),
            hint_level=0,
        )
    )

    assert not session_path.exists()
