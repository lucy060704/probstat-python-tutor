"""Service tests for persistence, idempotency, reset, and friendly failures."""

import asyncio
from pathlib import Path

import pytest

from probstat_tutor.config import Settings
from probstat_tutor.schemas import ConceptId, DiagnosticReport
from probstat_tutor.service import LearningService, LearningServiceError
from probstat_tutor.storage import LearningStateStore


def _service(tmp_path: Path) -> LearningService:
    settings = Settings(
        openai_api_key=None,
        openai_model=None,
        session_db_path=tmp_path / "sessions.sqlite3",
        learning_state_db_path=tmp_path / "learning.sqlite3",
    )
    return LearningService(
        settings=settings,
        store=LearningStateStore(settings.learning_state_db_path),
    )


def _submit(service: LearningService, *, answer: str = "8") -> DiagnosticReport:
    return asyncio.run(
        service.submit(
            learner_id="demo",
            session_id="service-session",
            question_id="mean_median_python_01",
            answer=answer,
            reasoning="四个数排序后，中间两个数是 6 和 10。",
            python_code='df["value"].median()',
            hint_level=0,
        )
    )


def test_offline_service_demonstrates_question_grading_mastery_and_next(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)

    question = service.choose_question("demo", ConceptId.MEAN_MEDIAN)
    report = _submit(service)
    dashboard = service.get_dashboard("demo", ConceptId.MEAN_MEDIAN)

    assert service.offline_mode is True
    assert question.concept_id == ConceptId.MEAN_MEDIAN
    assert report.overall_correctness == 1.0
    assert dashboard.state.history[-1].question_id == "mean_median_python_01"
    assert report.next_question_id is not None
    assert any("思考过程" in evidence for evidence in report.evidence)
    assert any("代码文本" in evidence for evidence in report.evidence)


def test_identical_consecutive_submission_is_written_once(tmp_path: Path) -> None:
    service = _service(tmp_path)

    first = _submit(service)
    second = _submit(service)
    history = service.get_dashboard("demo", ConceptId.MEAN_MEDIAN).state.history

    assert first == second
    assert len(history) == 1


def test_state_survives_new_service_instance(tmp_path: Path) -> None:
    first_service = _service(tmp_path)
    _submit(first_service)

    refreshed_service = _service(tmp_path)
    state = refreshed_service.get_dashboard("demo", ConceptId.MEAN_MEDIAN).state

    assert len(state.history) == 1
    assert "mean_median_python_01" in state.completed_question_ids


def test_reset_demo_learner_clears_state_and_receipts(tmp_path: Path) -> None:
    service = _service(tmp_path)
    _submit(service)

    service.reset_demo_learner("demo")
    reset_state = service.get_dashboard("demo", ConceptId.MEAN_MEDIAN).state

    assert reset_state.history == ()
    assert reset_state.completed_question_ids == frozenset()


def test_service_converts_agent_failure_to_friendly_chinese_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _service(tmp_path)

    async def fail_diagnose(*args: object, **kwargs: object) -> None:
        raise RuntimeError("provider details must not reach the learner")

    monkeypatch.setattr(service.tutor, "diagnose", fail_diagnose)

    with pytest.raises(LearningServiceError, match="教学服务暂时不可用"):
        _submit(service)
