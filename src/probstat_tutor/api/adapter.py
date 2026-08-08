"""Thin API adapter that delegates all learning behavior to LearningService."""

from dataclasses import dataclass

from probstat_tutor import __version__
from probstat_tutor.api.schemas import (
    ApiErrorCode,
    DiagnoseRequest,
    DiagnoseResponse,
    HealthResponse,
    HintRequest,
    HintResponse,
    OptionalModelStatus,
    PublicQuestion,
    RecommendRequest,
    RecommendResponse,
)
from probstat_tutor.reliability import CircuitState
from probstat_tutor.service import (
    LearningIdempotencyConflictError,
    LearningRecommendationUnavailableError,
    LearningService,
    LearningServiceError,
    LearningServiceUnavailableError,
    LearningStateConflictError,
)


@dataclass(frozen=True)
class ApiOperationError(RuntimeError):
    """Safe transport-level mapping for an expected service outcome."""

    code: ApiErrorCode
    status_code: int
    message_zh: str
    retryable: bool = False


class LearningApiAdapter:
    """Map versioned API messages to the existing application service."""

    def __init__(self, service: LearningService) -> None:
        self.service = service

    def health(self, request_id: str) -> HealthResponse:
        return HealthResponse(
            request_id=request_id,
            service_version=__version__,
            optional_model_status=(
                OptionalModelStatus.DISABLED
                if self.service.offline_mode
                else (
                    OptionalModelStatus.ENABLED
                    if self.service.tutor.model_circuit_state == CircuitState.CLOSED
                    else OptionalModelStatus.DEGRADED
                )
            ),
            local_knowledge_base_available=self.service.tutor.rag_index is not None,
        )

    async def diagnose(self, request: DiagnoseRequest) -> DiagnoseResponse:
        try:
            self.service.get_question(request.question_id)
        except LearningServiceError as error:
            raise ApiOperationError(
                code=ApiErrorCode.NOT_FOUND,
                status_code=404,
                message_zh="请求的题目不存在。",
            ) from error

        try:
            report = await self.service.submit(
                learner_id=request.anonymous_profile_id,
                session_id=f"api-{request.request_id}",
                question_id=request.question_id,
                answer=request.submission.answer,
                reasoning=request.submission.reasoning,
                python_code=request.submission.python_code,
                hint_level=request.hint_level,
                idempotency_key=request.idempotency_key,
            )
        except (LearningIdempotencyConflictError, LearningStateConflictError) as error:
            raise ApiOperationError(
                code=ApiErrorCode.STATE_CONFLICT,
                status_code=409,
                message_zh=(
                    "幂等键已用于不同的提交内容，请为新提交生成新键。"
                    if isinstance(error, LearningIdempotencyConflictError)
                    else "学习状态刚刚发生变化，请刷新后重试。"
                ),
            ) from error
        except LearningServiceError as error:
            raise ApiOperationError(
                code=ApiErrorCode.SERVICE_UNAVAILABLE,
                status_code=503,
                message_zh="诊断服务暂时不可用，请稍后使用相同幂等键重试。",
                retryable=True,
            ) from error

        return DiagnoseResponse(
            request_id=request.request_id,
            idempotency_key=request.idempotency_key,
            report=report,
        )

    def hint(self, request: HintRequest) -> HintResponse:
        try:
            hint = self.service.get_hint(request.question_id, request.hint_level)
        except LearningServiceError as error:
            raise ApiOperationError(
                code=ApiErrorCode.NOT_FOUND,
                status_code=404,
                message_zh="请求的题目不存在。",
            ) from error
        return HintResponse(
            request_id=request.request_id,
            idempotency_key=request.idempotency_key,
            question_id=request.question_id,
            hint_level=request.hint_level,
            hint_zh=hint,
            complete_explanation_revealed=request.hint_level == 4,
        )

    def recommend(self, request: RecommendRequest) -> RecommendResponse:
        try:
            question = self.service.choose_question(
                request.anonymous_profile_id,
                request.concept_id,
            )
        except LearningRecommendationUnavailableError as error:
            raise ApiOperationError(
                code=ApiErrorCode.STATE_CONFLICT,
                status_code=409,
                message_zh="当前学习状态暂时没有可推荐的题目。",
            ) from error
        except LearningServiceUnavailableError as error:
            raise ApiOperationError(
                code=ApiErrorCode.SERVICE_UNAVAILABLE,
                status_code=503,
                message_zh="推荐服务暂时不可用，请稍后使用相同幂等键重试。",
                retryable=True,
            ) from error
        return RecommendResponse(
            request_id=request.request_id,
            idempotency_key=request.idempotency_key,
            question=PublicQuestion(
                id=question.id,
                title=question.title,
                concept_id=question.concept_id,
                question_type=question.question_type,
                difficulty=question.difficulty,
                prompt=question.prompt,
                dataset=question.dataset,
            ),
        )
