"""Privacy-safe local fault events for optional product components."""

import json
import re
import threading
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FaultComponent(StrEnum):
    """Optional component that failed without breaking deterministic learning."""

    MODEL = "model"
    RAG = "rag"


class FaultCode(StrEnum):
    """Stable, non-sensitive failure categories used in local demonstrations."""

    MODEL_FALLBACK = "model_fallback"
    MODEL_TIMEOUT = "model_timeout"
    MODEL_RETRY_EXHAUSTED = "model_retry_exhausted"
    MODEL_CIRCUIT_OPEN = "model_circuit_open"
    RAG_INDEX_UNAVAILABLE = "rag_index_unavailable"
    RAG_QUERY_UNAVAILABLE = "rag_query_unavailable"


class RecoveryAction(StrEnum):
    """Recovery path taken by the application."""

    DETERMINISTIC_DIAGNOSIS = "deterministic_diagnosis"
    NO_KNOWLEDGE_CONTEXT = "no_knowledge_context"


class FaultEvent(BaseModel):
    """Allow-listed fault record that excludes submissions and identifiers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    timestamp_utc: str
    component: FaultComponent
    code: FaultCode
    exception_type: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_.]{0,99}$")
    recovery: RecoveryAction

    @field_validator("timestamp_utc")
    @classmethod
    def timestamp_is_utc(cls, value: str) -> str:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
            raise ValueError("故障时间戳必须使用 UTC 时区")
        return value


class SafeFaultLogger:
    """Append allow-listed JSONL events; logging failure never breaks learning."""

    _lock = threading.Lock()

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def record(
        self,
        *,
        component: FaultComponent,
        code: FaultCode,
        error: Exception,
        recovery: RecoveryAction,
    ) -> FaultEvent:
        exception_type = type(error).__name__
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]{0,99}", exception_type) is None:
            exception_type = "Exception"
        event = FaultEvent(
            event_id=uuid.uuid4().hex,
            timestamp_utc=datetime.now(UTC).isoformat(),
            component=component,
            code=code,
            exception_type=exception_type,
            recovery=recovery,
        )
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            serialized = json.dumps(event.model_dump(mode="json"), ensure_ascii=False)
            with self._lock, self.path.open("a", encoding="utf-8") as stream:
                stream.write(f"{serialized}\n")
        except OSError:
            pass
        return event
