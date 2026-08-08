"""Versioned Pydantic request and response contracts for the local API."""

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from probstat_tutor.schemas import (
    ConceptId,
    DiagnosticReport,
    LearnerSubmission,
    QuestionType,
)

API_SCHEMA_VERSION = "1.0.0"
REQUEST_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]{7,127}$"
IDEMPOTENCY_KEY_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]{7,127}$"
ANONYMOUS_PROFILE_PATTERN = r"^(?:local|api)_anon_[0-9a-f]{16,64}$"
QUESTION_ID_PATTERN = r"^[a-z0-9_]{3,100}$"


class ApiRequestBase(BaseModel):
    """Metadata required for every JSON operation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0.0"]
    request_id: str = Field(pattern=REQUEST_ID_PATTERN)
    idempotency_key: str = Field(pattern=IDEMPOTENCY_KEY_PATTERN)


class DiagnoseRequest(ApiRequestBase):
    """One deterministic learning submission."""

    anonymous_profile_id: str = Field(pattern=ANONYMOUS_PROFILE_PATTERN)
    question_id: str = Field(pattern=QUESTION_ID_PATTERN)
    submission: LearnerSubmission
    hint_level: int = Field(ge=0, le=4)


class HintRequest(ApiRequestBase):
    """Request exactly one progressive hint level."""

    question_id: str = Field(pattern=QUESTION_ID_PATTERN)
    hint_level: int = Field(ge=1, le=4)


class RecommendRequest(ApiRequestBase):
    """Request a safe public question for one anonymous profile and concept."""

    anonymous_profile_id: str = Field(pattern=ANONYMOUS_PROFILE_PATTERN)
    concept_id: ConceptId


class OptionalModelStatus(StrEnum):
    """Whether optional online explanation is configured, without exposing secrets."""

    DISABLED = "disabled"
    ENABLED = "enabled"
    DEGRADED = "degraded"


class HealthResponse(BaseModel):
    """Minimal health data safe to expose through a future HTTPS edge."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0.0"] = API_SCHEMA_VERSION
    request_id: str = Field(pattern=REQUEST_ID_PATTERN)
    status: Literal["ok"] = "ok"
    service_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    offline_core_available: Literal[True] = True
    optional_model_status: OptionalModelStatus
    local_knowledge_base_available: bool


class PublicQuestion(BaseModel):
    """Question fields allowed across the API; answers and grading rules are absent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=QUESTION_ID_PATTERN)
    title: str
    concept_id: ConceptId
    question_type: QuestionType
    difficulty: float = Field(ge=0.0, le=1.0)
    prompt: str
    dataset: dict[str, Any]


class DiagnoseResponse(BaseModel):
    """Versioned deterministic report envelope."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0.0"] = API_SCHEMA_VERSION
    request_id: str = Field(pattern=REQUEST_ID_PATTERN)
    idempotency_key: str = Field(pattern=IDEMPOTENCY_KEY_PATTERN)
    report: DiagnosticReport


class HintResponse(BaseModel):
    """One progressive hint without hidden answer fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0.0"] = API_SCHEMA_VERSION
    request_id: str = Field(pattern=REQUEST_ID_PATTERN)
    idempotency_key: str = Field(pattern=IDEMPOTENCY_KEY_PATTERN)
    question_id: str = Field(pattern=QUESTION_ID_PATTERN)
    hint_level: int = Field(ge=1, le=4)
    hint_zh: str
    complete_explanation_revealed: bool


class RecommendResponse(BaseModel):
    """Public next-question envelope."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0.0"] = API_SCHEMA_VERSION
    request_id: str = Field(pattern=REQUEST_ID_PATTERN)
    idempotency_key: str = Field(pattern=IDEMPOTENCY_KEY_PATTERN)
    question: PublicQuestion


class ApiErrorCode(StrEnum):
    """Stable categories clients may handle without parsing prose."""

    INVALID_JSON = "invalid_json"
    INVALID_REQUEST = "invalid_request"
    PAYLOAD_TOO_LARGE = "payload_too_large"
    UNSUPPORTED_MEDIA_TYPE = "unsupported_media_type"
    NOT_FOUND = "not_found"
    METHOD_NOT_ALLOWED = "method_not_allowed"
    STATE_CONFLICT = "state_conflict"
    SERVICE_UNAVAILABLE = "service_unavailable"
    INTERNAL_ERROR = "internal_error"


class ApiErrorDetail(BaseModel):
    """Safe structured error; invalid values and exception messages are excluded."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: ApiErrorCode
    message_zh: str
    retryable: bool
    invalid_fields: tuple[str, ...] = ()


class ApiErrorResponse(BaseModel):
    """Versioned error envelope returned for every non-2xx API response."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0.0"] = API_SCHEMA_VERSION
    request_id: str = Field(pattern=REQUEST_ID_PATTERN)
    error: ApiErrorDetail
