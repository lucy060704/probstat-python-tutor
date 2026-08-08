"""Small ASGI application intended for a future controlled HTTPS edge."""

import json
import re
import uuid
from collections.abc import Awaitable, Callable
from typing import TypeVar

from pydantic import BaseModel, ValidationError
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from probstat_tutor.api.adapter import ApiOperationError, LearningApiAdapter
from probstat_tutor.api.schemas import (
    REQUEST_ID_PATTERN,
    ApiErrorCode,
    ApiErrorDetail,
    ApiErrorResponse,
    DiagnoseRequest,
    HintRequest,
    RecommendRequest,
)
from probstat_tutor.service import LearningService

MAX_JSON_BODY_BYTES = 32_768
RequestModel = TypeVar("RequestModel", bound=BaseModel)
Handler = Callable[[Request], Awaitable[Response]]
_REQUEST_ID_RE = re.compile(REQUEST_ID_PATTERN)


def create_api_app(service: LearningService | None = None) -> Starlette:
    """Build an injectable ASGI app without opening a socket or contacting ADP."""

    adapter = LearningApiAdapter(service or LearningService())

    async def health(request: Request) -> Response:
        request_id = _request_id_from_header(request)
        return _model_response(adapter.health(request_id))

    async def diagnose(request: Request) -> Response:
        return await _handle_json_operation(request, DiagnoseRequest, adapter.diagnose)

    async def hint(request: Request) -> Response:
        return await _handle_json_operation(request, HintRequest, adapter.hint)

    async def recommend(request: Request) -> Response:
        return await _handle_json_operation(request, RecommendRequest, adapter.recommend)

    async def http_error(request: Request, error: Exception) -> Response:
        if not isinstance(error, HTTPException):
            return _error_response(
                request_id=_request_id_from_header(request),
                status_code=500,
                code=ApiErrorCode.INTERNAL_ERROR,
                message_zh="服务发生未预期错误。",
                retryable=True,
            )
        code = (
            ApiErrorCode.METHOD_NOT_ALLOWED
            if error.status_code == 405
            else ApiErrorCode.NOT_FOUND
        )
        message = "请求方法不受支持。" if error.status_code == 405 else "接口不存在。"
        return _error_response(
            request_id=_request_id_from_header(request),
            status_code=error.status_code,
            code=code,
            message_zh=message,
            retryable=False,
        )

    return Starlette(
        debug=False,
        routes=[
            Route("/health", health, methods=["GET"]),
            Route("/v1/diagnose", diagnose, methods=["POST"]),
            Route("/v1/hint", hint, methods=["POST"]),
            Route("/v1/recommend", recommend, methods=["POST"]),
        ],
        exception_handlers={HTTPException: http_error, Exception: http_error},
    )


async def _handle_json_operation(
    request: Request,
    model_type: type[RequestModel],
    operation: Callable[[RequestModel], object],
) -> Response:
    parsed = await _parse_json_request(request, model_type)
    if isinstance(parsed, Response):
        return parsed
    try:
        result = operation(parsed)
        if isinstance(result, Awaitable):
            result = await result
        if not isinstance(result, BaseModel):
            raise TypeError("API operation must return a Pydantic model")
        return _model_response(result)
    except ApiOperationError as error:
        return _error_response(
            request_id=parsed.request_id,
            status_code=error.status_code,
            code=error.code,
            message_zh=error.message_zh,
            retryable=error.retryable,
        )
    except Exception:
        return _error_response(
            request_id=parsed.request_id,
            status_code=500,
            code=ApiErrorCode.INTERNAL_ERROR,
            message_zh="服务发生未预期错误。",
            retryable=True,
        )


async def _parse_json_request(
    request: Request,
    model_type: type[RequestModel],
) -> RequestModel | Response:
    request_id = _request_id_from_header(request)
    if request.headers.get("content-type", "").split(";", maxsplit=1)[0].strip().lower() != (
        "application/json"
    ):
        return _error_response(
            request_id=request_id,
            status_code=415,
            code=ApiErrorCode.UNSUPPORTED_MEDIA_TYPE,
            message_zh="请求必须使用 application/json。",
            retryable=False,
        )

    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_JSON_BODY_BYTES:
            return _error_response(
                request_id=request_id,
                status_code=413,
                code=ApiErrorCode.PAYLOAD_TOO_LARGE,
                message_zh="请求内容过大。",
                retryable=False,
            )
    try:
        data = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return _error_response(
            request_id=request_id,
            status_code=400,
            code=ApiErrorCode.INVALID_JSON,
            message_zh="请求正文不是有效 JSON。",
            retryable=False,
        )
    try:
        return model_type.model_validate(data)
    except ValidationError as error:
        invalid_fields = tuple(
            sorted({".".join(str(item) for item in issue["loc"]) for issue in error.errors()})
        )
        return _error_response(
            request_id=request_id,
            status_code=422,
            code=ApiErrorCode.INVALID_REQUEST,
            message_zh="请求字段不符合 API 契约。",
            retryable=False,
            invalid_fields=invalid_fields,
        )


def _request_id_from_header(request: Request) -> str:
    candidate = request.headers.get("x-request-id", "")
    if _REQUEST_ID_RE.fullmatch(candidate):
        return candidate
    return f"server_{uuid.uuid4().hex}"


def _model_response(model: BaseModel, *, status_code: int = 200) -> JSONResponse:
    return JSONResponse(
        model.model_dump(mode="json"),
        status_code=status_code,
        headers={
            "cache-control": "no-store",
            "x-content-type-options": "nosniff",
        },
    )


def _error_response(
    *,
    request_id: str,
    status_code: int,
    code: ApiErrorCode,
    message_zh: str,
    retryable: bool,
    invalid_fields: tuple[str, ...] = (),
) -> JSONResponse:
    return _model_response(
        ApiErrorResponse(
            request_id=request_id,
            error=ApiErrorDetail(
                code=code,
                message_zh=message_zh,
                retryable=retryable,
                invalid_fields=invalid_fields,
            ),
        ),
        status_code=status_code,
    )
