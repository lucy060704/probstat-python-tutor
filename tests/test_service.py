"""Service tests for persistence, idempotency, reset, and friendly failures."""

import asyncio
import json
import sqlite3
from pathlib import Path

import pytest
from pydantic import SecretStr

from probstat_tutor.config import Settings
from probstat_tutor.reliability import CircuitState
from probstat_tutor.schemas import (
    ConceptId,
    DeliveryMode,
    DiagnosticReport,
    MasteryScores,
    SubmissionField,
)
from probstat_tutor.service import (
    LearningIdempotencyConflictError,
    LearningService,
    LearningServiceError,
)
from probstat_tutor.storage import (
    CommitSubmissionResult,
    CommitSubmissionStatus,
    LearningStateStore,
)


def _service(tmp_path: Path) -> LearningService:
    settings = Settings(
        openai_api_key=None,
        openai_model=None,
        session_db_path=tmp_path / "sessions.sqlite3",
        learning_state_db_path=tmp_path / "learning.sqlite3",
        fault_log_path=tmp_path / "logs" / "faults.jsonl",
    )
    return LearningService(
        settings=settings,
        store=LearningStateStore(settings.learning_state_db_path),
    )


def _online_service(tmp_path: Path) -> LearningService:
    settings = Settings(
        openai_api_key=SecretStr("test-key"),
        openai_model="test-model",
        session_db_path=tmp_path / "online-sessions.sqlite3",
        learning_state_db_path=tmp_path / "online-learning.sqlite3",
        fault_log_path=tmp_path / "online-logs" / "faults.jsonl",
        model_retry_base_delay_seconds=0.0,
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


def test_data_quality_question_uses_structured_fourth_hint(tmp_path: Path) -> None:
    service = _service(tmp_path)

    first_hint = service.get_hint("data_quality_python_01", 1)
    fourth_hint = service.get_hint("data_quality_python_01", 4)

    assert first_hint.startswith("概念提示：")
    assert 'df["score"].isna().sum()' not in first_hint
    assert fourth_hint.startswith("完整解释：")
    assert "概念：" in fourth_hint
    assert "计算：" in fourth_hint
    assert "Python：" in fourth_hint
    assert "情境解释：" in fourth_hint


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
    assert [evidence.source for evidence in report.learner_evidence] == [
        SubmissionField.ANSWER,
        SubmissionField.REASONING,
        SubmissionField.PYTHON_CODE,
    ]


def test_answer_only_python_submission_reports_missing_code(tmp_path: Path) -> None:
    service = _service(tmp_path)

    report = asyncio.run(
        service.submit(
            learner_id="answer-only",
            session_id="answer-only-session",
            question_id="mean_median_python_01",
            answer="8",
            hint_level=0,
        )
    )

    assert report.overall_correctness == 0.0
    assert report.misconception_tags == ["python_code_missing"]
    assert [(item.source, item.quote) for item in report.learner_evidence] == [
        (SubmissionField.ANSWER, "8")
    ]


def test_invalid_submission_returns_chinese_error_without_writing_state(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)

    with pytest.raises(LearningServiceError, match="提交内容无效：答案不能为空"):
        asyncio.run(
            service.submit(
                learner_id="invalid",
                session_id="invalid-session",
                question_id="mean_median_python_01",
                answer="   ",
                hint_level=0,
            )
        )

    state = service.get_dashboard("invalid", ConceptId.MEAN_MEDIAN).state
    assert state.history == ()


def test_invalid_hint_level_returns_chinese_error_without_writing_receipt(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)

    with pytest.raises(LearningServiceError, match="提示层级必须是 0、1、2、3 或 4"):
        asyncio.run(
            service.submit(
                learner_id="invalid-hint",
                session_id="invalid-hint-session",
                question_id="mean_median_python_01",
                answer="8",
                hint_level=object(),  # type: ignore[arg-type]
            )
        )

    state = service.get_dashboard("invalid-hint", ConceptId.MEAN_MEDIAN).state
    assert state.history == ()
    with sqlite3.connect(service.store.db_path) as connection:
        receipt_count = connection.execute(
            "SELECT COUNT(*) FROM submission_receipts WHERE learner_id = ?",
            ("invalid-hint",),
        ).fetchone()[0]
    assert receipt_count == 0


def test_answer_with_conflicting_reasoning_and_code_is_preserved_as_evidence(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)

    report = asyncio.run(
        service.submit(
            learner_id="conflict",
            session_id="conflict-session",
            question_id="mean_median_python_01",
            answer="8",
            reasoning="中位数就是这组数据的最大值。",
            python_code='df["value"].mean()',
            hint_level=0,
        )
    )

    assert [(item.source, item.quote) for item in report.learner_evidence] == [
        (SubmissionField.ANSWER, "8"),
        (SubmissionField.REASONING, "中位数就是这组数据的最大值。"),
        (SubmissionField.PYTHON_CODE, 'df["value"].mean()'),
    ]


def test_identical_consecutive_submission_is_written_once(tmp_path: Path) -> None:
    service = _service(tmp_path)

    first = _submit(service)
    second = _submit(service)
    history = service.get_dashboard("demo", ConceptId.MEAN_MEDIAN).state.history

    assert first == second
    assert len(history) == 1


def test_cached_submission_skips_tutor_diagnosis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    first = _submit(service)

    async def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("缓存命中后不应再次调用 TutorAgent")

    monkeypatch.setattr(service.tutor, "diagnose", fail_if_called)

    assert _submit(service) == first


def test_explicit_idempotency_key_replays_same_body_and_rejects_changed_body(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    common = {
        "learner_id": "idempotent-learner",
        "question_id": "mean_median_python_01",
        "python_code": 'df["value"].median()',
        "hint_level": 0,
        "idempotency_key": "idem_service_001",
    }

    first = asyncio.run(
        service.submit(
            **common,
            session_id="first-request",
            answer="8",
        )
    )
    replay = asyncio.run(
        service.submit(
            **common,
            session_id="retry-request",
            answer="8",
        )
    )
    with pytest.raises(
        LearningIdempotencyConflictError,
        match="同一幂等键不能用于不同的提交内容",
    ):
        asyncio.run(
            service.submit(
                **common,
                session_id="changed-request",
                answer="7",
            )
        )

    history = service.get_dashboard(
        "idempotent-learner", ConceptId.MEAN_MEDIAN
    ).state.history
    assert replay == first
    assert len(history) == 1


def test_concurrent_identical_submissions_update_state_once(tmp_path: Path) -> None:
    service = _service(tmp_path)

    async def submit_both() -> tuple[DiagnosticReport, DiagnosticReport]:
        first, second = await asyncio.gather(
            service.submit(
                learner_id="concurrent-same",
                session_id="concurrent-same-1",
                question_id="mean_median_python_01",
                answer="8",
                python_code='df["value"].median()',
                hint_level=0,
            ),
                service.submit(
                    learner_id="concurrent-same",
                    session_id="concurrent-same-2",
                    question_id="mean_median_python_01",
                    answer="8",
                    python_code='df["value"].median()',
                    hint_level=0,
                ),
        )
        return first, second

    first, second = asyncio.run(submit_both())
    history = service.get_dashboard(
        "concurrent-same", ConceptId.MEAN_MEDIAN
    ).state.history

    assert first == second
    assert len(history) == 1


def test_concurrent_distinct_submissions_retry_without_losing_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    original_diagnose = service.tutor.diagnose

    async def submit_both() -> int:
        ready = 0
        diagnose_calls = 0
        release_first_pair = asyncio.Event()

        async def synchronized_diagnose(*args: object, **kwargs: object) -> object:
            nonlocal diagnose_calls, ready
            prepared = await original_diagnose(*args, **kwargs)
            diagnose_calls += 1
            current_call = diagnose_calls
            if current_call <= 2:
                ready += 1
                if ready == 2:
                    release_first_pair.set()
                await release_first_pair.wait()
            return prepared

        monkeypatch.setattr(service.tutor, "diagnose", synchronized_diagnose)
        await asyncio.gather(
            service.submit(
                learner_id="concurrent-distinct",
                session_id="concurrent-distinct-1",
                question_id="mean_median_python_01",
                answer="8",
                python_code='df["value"].median()',
                hint_level=0,
            ),
            service.submit(
                learner_id="concurrent-distinct",
                session_id="concurrent-distinct-2",
                question_id="mean_median_python_01",
                answer="7",
                python_code='df["value"].median()',
                hint_level=0,
            ),
        )
        return diagnose_calls

    diagnose_calls = asyncio.run(submit_both())
    history = service.get_dashboard(
        "concurrent-distinct", ConceptId.MEAN_MEDIAN
    ).state.history

    assert diagnose_calls == 3
    assert len(history) == 2
    assert {attempt.score for attempt in history} == {0.0, 1.0}


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

    assert service.get_dashboard("demo", ConceptId.MEAN_MEDIAN).state.history == ()


def test_online_model_failure_falls_back_and_commits_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _online_service(tmp_path)

    model_calls = 0

    async def fail_model(*args: object, **kwargs: object) -> None:
        nonlocal model_calls
        model_calls += 1
        raise RuntimeError("simulated provider failure")

    monkeypatch.setattr("probstat_tutor.tutor_agent.Runner.run", fail_model)

    report = _submit(service)

    assert report.delivery_mode == DeliveryMode.MODEL_FALLBACK
    assert model_calls == 2
    assert "确定性本地诊断" in report.delivery_message_zh
    assert len(service.get_dashboard("demo", ConceptId.MEAN_MEDIAN).state.history) == 1
    assert not Path(service.tutor.settings.session_db_path).exists()
    with sqlite3.connect(service.store.db_path) as connection:
        receipt_count = connection.execute(
            "SELECT COUNT(*) FROM submission_receipts WHERE learner_id = ?",
            ("demo",),
        ).fetchone()[0]
    assert receipt_count == 1
    log_text = service.settings.fault_log_path.read_text(encoding="utf-8")
    assert "model_retry_exhausted" in log_text
    assert "simulated provider failure" not in log_text


def test_transient_model_failure_retries_then_commits_one_enhanced_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _online_service(tmp_path)
    model_calls = 0

    class ModelResult:
        final_output = DiagnosticReport(
            question_id="mean_median_python_01",
            overall_correctness=1.0,
            dimension_scores=MasteryScores(),
            evidence=["模型解释层输出"],
            misconception_tags=[],
            feedback="在线解释",
            hint_level=0,
            recommended_action="继续学习",
            next_question_id=None,
            uncertainty="无",
        )

    async def fail_once(*args: object, **kwargs: object) -> ModelResult:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            raise RuntimeError("temporary-provider-detail")
        return ModelResult()

    monkeypatch.setattr("probstat_tutor.tutor_agent.Runner.run", fail_once)

    report = _submit(service)

    assert model_calls == 2
    assert report.delivery_mode == DeliveryMode.MODEL_ENHANCED
    assert report.feedback == "在线解释"
    assert len(service.get_dashboard("demo", ConceptId.MEAN_MEDIAN).state.history) == 1
    assert service.tutor.model_circuit_state == CircuitState.CLOSED
    assert not service.settings.fault_log_path.exists()


def test_invalid_model_output_falls_back_without_losing_learning_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _online_service(tmp_path)

    class InvalidModelResult:
        final_output = {"feedback": "缺少确定性报告必填字段"}

    async def invalid_model(*args: object, **kwargs: object) -> InvalidModelResult:
        return InvalidModelResult()

    monkeypatch.setattr("probstat_tutor.tutor_agent.Runner.run", invalid_model)

    report = _submit(service)

    assert report.delivery_mode == DeliveryMode.MODEL_FALLBACK
    assert len(service.get_dashboard("demo", ConceptId.MEAN_MEDIAN).state.history) == 1
    assert not Path(service.tutor.settings.session_db_path).exists()
    with sqlite3.connect(service.store.db_path) as connection:
        receipt_count = connection.execute(
            "SELECT COUNT(*) FROM submission_receipts WHERE learner_id = ?",
            ("demo",),
        ).fetchone()[0]
    assert receipt_count == 1


def test_model_timeout_opens_circuit_without_breaking_offline_core(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        openai_api_key=SecretStr("test-key"),
        openai_model="test-model",
        session_db_path=tmp_path / "timeout-sessions.sqlite3",
        learning_state_db_path=tmp_path / "timeout-learning.sqlite3",
        fault_log_path=tmp_path / "timeout-logs" / "faults.jsonl",
        model_timeout_seconds=0.001,
        model_max_attempts=2,
        model_retry_base_delay_seconds=0.0,
        model_circuit_failure_threshold=2,
        model_circuit_open_seconds=30.0,
    )
    service = LearningService(
        settings=settings,
        store=LearningStateStore(settings.learning_state_db_path),
    )
    model_calls = 0

    async def too_slow(*args: object, **kwargs: object) -> None:
        nonlocal model_calls
        model_calls += 1
        await asyncio.sleep(0.05)

    monkeypatch.setattr("probstat_tutor.tutor_agent.Runner.run", too_slow)

    reports = tuple(
        asyncio.run(
            service.submit(
                learner_id=f"timeout-learner-{index}",
                session_id=f"timeout-session-{index}",
                question_id="mean_median_python_01",
                answer="8",
                python_code='df["value"].median()',
                hint_level=0,
            )
        )
        for index in range(3)
    )

    assert all(report.delivery_mode == DeliveryMode.MODEL_FALLBACK for report in reports)
    assert model_calls == 4
    assert service.tutor.model_circuit_state == CircuitState.OPEN
    assert all(
        len(
            service.get_dashboard(
                f"timeout-learner-{index}", ConceptId.MEAN_MEDIAN
            ).state.history
        )
        == 1
        for index in range(3)
    )
    fault_codes = [
        json.loads(line)["code"]
        for line in settings.fault_log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert fault_codes == ["model_timeout", "model_timeout", "model_circuit_open"]


def test_receipt_commit_failure_rolls_back_full_service_call(
    tmp_path: Path,
) -> None:
    service = _online_service(tmp_path)

    class ModelResult:
        final_output = DiagnosticReport(
            question_id="mean_median_python_01",
            overall_correctness=0.0,
            dimension_scores=MasteryScores(),
            evidence=["模型解释层输出"],
            misconception_tags=[],
            feedback="模型解释",
            hint_level=0,
            recommended_action="继续学习",
            next_question_id=None,
            uncertainty="无",
        )

    async def successful_model(*args: object, **kwargs: object) -> ModelResult:
        return ModelResult()

    with sqlite3.connect(service.store.db_path) as connection:
        connection.execute(
            """
            CREATE TRIGGER fail_receipt_insert_from_service
            BEFORE INSERT ON submission_receipts
            BEGIN
                SELECT RAISE(ABORT, 'injected service receipt failure');
            END
            """
        )

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr("probstat_tutor.tutor_agent.Runner.run", successful_model)
        with pytest.raises(LearningServiceError, match="教学服务暂时不可用"):
            _submit(service)

    assert service.get_dashboard("demo", ConceptId.MEAN_MEDIAN).state.history == ()
    assert not Path(service.tutor.settings.session_db_path).exists()
    with sqlite3.connect(service.store.db_path) as connection:
        receipt_count = connection.execute(
            "SELECT COUNT(*) FROM submission_receipts WHERE learner_id = ?",
            ("demo",),
        ).fetchone()[0]
    assert receipt_count == 0


def test_repeated_snapshot_conflicts_return_chinese_retry_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    commit_attempts = 0

    def always_conflict(*args: object, **kwargs: object) -> CommitSubmissionResult:
        nonlocal commit_attempts
        commit_attempts += 1
        return CommitSubmissionResult(status=CommitSubmissionStatus.CONFLICT)

    monkeypatch.setattr(service.store, "commit_submission", always_conflict)

    with pytest.raises(LearningServiceError, match="学习状态刚刚发生变化"):
        _submit(service)

    assert commit_attempts == 3
    assert service.get_dashboard("demo", ConceptId.MEAN_MEDIAN).state.history == ()
