"""Generate a machine-readable OpenAPI contract from the Pydantic schemas."""

from typing import Literal

from pydantic import BaseModel
from pydantic.json_schema import models_json_schema

from probstat_tutor import __version__
from probstat_tutor.api.schemas import (
    API_SCHEMA_VERSION,
    REQUEST_ID_PATTERN,
    ApiErrorResponse,
    DiagnoseRequest,
    DiagnoseResponse,
    HealthResponse,
    HintRequest,
    HintResponse,
    RecommendRequest,
    RecommendResponse,
)

SchemaMode = Literal["validation", "serialization"]
_MODELS: tuple[tuple[type[BaseModel], SchemaMode], ...] = (
    (DiagnoseRequest, "validation"),
    (DiagnoseResponse, "serialization"),
    (HintRequest, "validation"),
    (HintResponse, "serialization"),
    (RecommendRequest, "validation"),
    (RecommendResponse, "serialization"),
    (HealthResponse, "serialization"),
    (ApiErrorResponse, "serialization"),
)


def build_openapi_contract() -> dict[str, object]:
    """Return the deterministic G3.4 OpenAPI 3.1 document."""

    _, definitions = models_json_schema(
        list(_MODELS),
        ref_template="#/components/schemas/{model}",
    )
    schemas = definitions.get("$defs", {})
    return {
        "openapi": "3.1.0",
        "info": {
            "title": "概率统计 × Python 学习诊断 API",
            "version": API_SCHEMA_VERSION,
            "description": (
                "本地验证的、平台无关的 JSON 契约。当前实现只允许回环地址；"
                "公网部署必须另加 HTTPS、鉴权、限流和密钥管理。"
            ),
        },
        "x-service-version": __version__,
        "x-public-deployment-ready": False,
        "paths": {
            "/health": {
                "get": {
                    "operationId": "health",
                    "parameters": [_request_id_header()],
                    "responses": {
                        "200": _response("服务健康", "HealthResponse"),
                        "500": _response("安全内部错误", "ApiErrorResponse"),
                    },
                },
                "head": {
                    "operationId": "healthHead",
                    "parameters": [_request_id_header()],
                    "responses": {
                        "200": {"description": "服务健康；响应不含正文"},
                        "500": {"description": "安全内部错误；响应不含正文"},
                    },
                },
            },
            "/v1/diagnose": _post_operation(
                "diagnose",
                "DiagnoseRequest",
                "DiagnoseResponse",
                extra_statuses=("404", "409", "503"),
            ),
            "/v1/hint": _post_operation(
                "hint",
                "HintRequest",
                "HintResponse",
                extra_statuses=("404",),
            ),
            "/v1/recommend": _post_operation(
                "recommend",
                "RecommendRequest",
                "RecommendResponse",
                extra_statuses=("409", "503"),
            ),
        },
        "components": {"schemas": schemas},
    }


def _post_operation(
    operation_id: str,
    request_schema: str,
    response_schema: str,
    *,
    extra_statuses: tuple[str, ...],
) -> dict[str, object]:
    responses = {
        "200": _response("成功", response_schema),
        "400": _response("JSON 无效", "ApiErrorResponse"),
        "413": _response("请求过大", "ApiErrorResponse"),
        "415": _response("媒体类型不支持", "ApiErrorResponse"),
        "422": _response("字段校验失败", "ApiErrorResponse"),
        "500": _response("安全内部错误", "ApiErrorResponse"),
    }
    responses.update(
        {status: _response("操作未完成", "ApiErrorResponse") for status in extra_statuses}
    )
    return {
        "post": {
            "operationId": operation_id,
            "parameters": [_request_id_header()],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"$ref": f"#/components/schemas/{request_schema}"}
                    }
                },
            },
            "responses": responses,
        }
    }


def _response(description: str, schema_name: str) -> dict[str, object]:
    return {
        "description": description,
        "content": {
            "application/json": {
                "schema": {"$ref": f"#/components/schemas/{schema_name}"}
            }
        },
    }


def _request_id_header() -> dict[str, object]:
    return {
        "name": "X-Request-ID",
        "in": "header",
        "required": False,
        "description": (
            "用于健康检查及请求正文校验前错误的可选追踪标识；"
            "有效 POST 请求以正文 request_id 为准。"
        ),
        "schema": {
            "type": "string",
            "pattern": REQUEST_ID_PATTERN,
        },
    }
