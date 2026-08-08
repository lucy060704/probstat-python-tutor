"""Safe fault logging tests: stable codes in, sensitive messages out."""

import json
from pathlib import Path

from probstat_tutor.observability import (
    FaultCode,
    FaultComponent,
    RecoveryAction,
    SafeFaultLogger,
)


def test_fault_log_contains_only_allow_listed_fields(tmp_path: Path) -> None:
    path = tmp_path / "logs" / "faults.jsonl"
    logger = SafeFaultLogger(path)
    logger.record(
        component=FaultComponent.MODEL,
        code=FaultCode.MODEL_FALLBACK,
        error=RuntimeError("learner@example.test secret-answer sk-sensitive"),
        recovery=RecoveryAction.DETERMINISTIC_DIAGNOSIS,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert set(payload) == {
        "event_id",
        "timestamp_utc",
        "component",
        "code",
        "exception_type",
        "recovery",
    }
    assert payload["exception_type"] == "RuntimeError"
    assert "learner@example.test" not in path.read_text(encoding="utf-8")
    assert "secret-answer" not in path.read_text(encoding="utf-8")
    assert "sk-sensitive" not in path.read_text(encoding="utf-8")


def test_logging_failure_does_not_raise(tmp_path: Path) -> None:
    blocking_file = tmp_path / "not-a-directory"
    blocking_file.write_text("block", encoding="utf-8")

    event = SafeFaultLogger(blocking_file / "faults.jsonl").record(
        component=FaultComponent.RAG,
        code=FaultCode.RAG_INDEX_UNAVAILABLE,
        error=OSError("unavailable"),
        recovery=RecoveryAction.NO_KNOWLEDGE_CONTEXT,
    )

    assert event.code == FaultCode.RAG_INDEX_UNAVAILABLE
