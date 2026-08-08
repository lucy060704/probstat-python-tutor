"""Black-box ASGI tests for the platform-independent G3.4 API contract."""

import asyncio
import json
import sqlite3
import subprocess
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr
from starlette.routing import Route

from probstat_tutor.api import create_api_app
from probstat_tutor.api.openapi import build_openapi_contract
from probstat_tutor.api.schemas import API_SCHEMA_VERSION
from probstat_tutor.config import Settings
from probstat_tutor.reliability import ModelCallFailedError
from probstat_tutor.schemas import ConceptId
from probstat_tutor.service import (
    LearningRecommendationUnavailableError,
    LearningService,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILE_ID = "api_anon_0123456789abcdef"


def _service(tmp_path: Path, name: str = "api") -> LearningService:
    settings = Settings(
        openai_api_key=None,
        openai_model=None,
        learning_state_db_path=tmp_path / f"{name}-learning.sqlite3",
        session_db_path=tmp_path / f"{name}-sessions.sqlite3",
        fault_log_path=tmp_path / name / "faults.jsonl",
    )
    return LearningService(settings=settings)


def _base_payload() -> dict[str, object]:
    return {
        "schema_version": API_SCHEMA_VERSION,
        "request_id": "request_test_001",
        "idempotency_key": "idem_test_001",
    }


async def _request(
    service: LearningService,
    method: str,
    path: str,
    **kwargs: object,
) -> httpx.Response:
    transport = httpx.ASGITransport(
        app=create_api_app(service),
        raise_app_exceptions=False,
    )
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://local-contract.test",
    ) as client:
        return await client.request(method, path, **kwargs)


def test_health_is_versioned_and_contains_no_secret_configuration(tmp_path: Path) -> None:
    response = asyncio.run(
        _request(
            _service(tmp_path),
            "GET",
            "/health",
            headers={"x-request-id": "health_test_001"},
        )
    )

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.json() == {
        "schema_version": "1.0.0",
        "request_id": "health_test_001",
        "status": "ok",
        "service_version": "0.1.0",
        "offline_core_available": True,
        "optional_model_status": "disabled",
        "local_knowledge_base_available": True,
    }
    serialized = response.text.lower()
    assert "api_key" not in serialized
    assert "openai" not in serialized
    assert "environment" not in serialized


def test_recommendation_exposes_question_but_not_answers_or_rules(tmp_path: Path) -> None:
    response = asyncio.run(
        _request(
            _service(tmp_path),
            "POST",
            "/v1/recommend",
            json={
                **_base_payload(),
                "anonymous_profile_id": PROFILE_ID,
                "concept_id": "data_quality",
            },
        )
    )

    assert response.status_code == 200
    question = response.json()["question"]
    assert question["id"] == "data_quality_concept_01"
    assert set(question) == {
        "id",
        "title",
        "concept_id",
        "question_type",
        "difficulty",
        "prompt",
        "dataset",
    }
    assert all(
        token not in response.text
        for token in ("expected_answer", "accepted_answers", "rubric", "hints")
    )


def test_hint_contract_preserves_progressive_disclosure(tmp_path: Path) -> None:
    service = _service(tmp_path)
    first = asyncio.run(
        _request(
            service,
            "POST",
            "/v1/hint",
            json={
                **_base_payload(),
                "question_id": "data_quality_concept_01",
                "hint_level": 1,
            },
        )
    )
    fourth = asyncio.run(
        _request(
            service,
            "POST",
            "/v1/hint",
            json={
                **_base_payload(),
                "request_id": "request_test_004",
                "idempotency_key": "idem_test_004",
                "question_id": "data_quality_concept_01",
                "hint_level": 4,
            },
        )
    )

    assert first.status_code == fourth.status_code == 200
    assert first.json()["complete_explanation_revealed"] is False
    assert "完整解释" not in first.json()["hint_zh"]
    assert fourth.json()["complete_explanation_revealed"] is True
    assert fourth.json()["hint_zh"].startswith("完整解释：")


def test_diagnose_reuses_service_and_same_idempotency_key_updates_once(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    payload = {
        **_base_payload(),
        "anonymous_profile_id": PROFILE_ID,
        "question_id": "data_quality_concept_01",
        "submission": {
            "answer": "2",
            "reasoning": "0 是合法分数，两个空白才是缺失。",
            "python_code": "",
        },
        "hint_level": 1,
    }

    first = asyncio.run(_request(service, "POST", "/v1/diagnose", json=payload))
    second = asyncio.run(_request(service, "POST", "/v1/diagnose", json=payload))
    history = service.get_dashboard(PROFILE_ID, ConceptId.DATA_QUALITY).state.history

    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert first.json()["report"]["overall_correctness"] == 1.0
    assert len(history) == 1


def test_diagnose_rejects_same_idempotency_key_with_changed_body(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    payload = {
        **_base_payload(),
        "anonymous_profile_id": PROFILE_ID,
        "question_id": "data_quality_concept_01",
        "submission": {"answer": "2"},
        "hint_level": 1,
    }

    first = asyncio.run(_request(service, "POST", "/v1/diagnose", json=payload))
    changed = asyncio.run(
        _request(
            service,
            "POST",
            "/v1/diagnose",
            json={
                **payload,
                "request_id": "request_test_changed_001",
                "submission": {"answer": "3"},
            },
        )
    )
    history = service.get_dashboard(PROFILE_ID, ConceptId.DATA_QUALITY).state.history

    assert first.status_code == 200
    assert changed.status_code == 409
    assert changed.json()["error"] == {
        "code": "state_conflict",
        "message_zh": "幂等键已用于不同的提交内容，请为新提交生成新键。",
        "retryable": False,
        "invalid_fields": [],
    }
    assert "request_test_001" not in changed.text
    assert len(history) == 1


def test_health_reports_optional_model_circuit_as_degraded(tmp_path: Path) -> None:
    settings = Settings(
        openai_api_key=SecretStr("test-key"),
        openai_model="test-model",
        learning_state_db_path=tmp_path / "degraded-learning.sqlite3",
        session_db_path=tmp_path / "degraded-sessions.sqlite3",
        fault_log_path=tmp_path / "degraded" / "faults.jsonl",
        model_max_attempts=1,
        model_retry_base_delay_seconds=0.0,
        model_circuit_failure_threshold=1,
    )
    service = LearningService(settings=settings)

    async def fail() -> None:
        raise RuntimeError("provider-secret")

    with pytest.raises(ModelCallFailedError):
        asyncio.run(service.tutor.model_reliability.run(fail))
    response = asyncio.run(_request(service, "GET", "/health"))

    assert response.status_code == 200
    assert response.json()["offline_core_available"] is True
    assert response.json()["optional_model_status"] == "degraded"
    assert "provider-secret" not in response.text


@pytest.mark.parametrize(
    ("method", "path", "expected_status", "expected_code"),
    [
        ("POST", "/health", 405, "method_not_allowed"),
        ("GET", "/v1/unknown", 404, "not_found"),
    ],
)
def test_routing_errors_use_the_same_safe_envelope(
    tmp_path: Path,
    method: str,
    path: str,
    expected_status: int,
    expected_code: str,
) -> None:
    response = asyncio.run(_request(_service(tmp_path), method, path))

    assert response.status_code == expected_status
    assert response.json()["schema_version"] == API_SCHEMA_VERSION
    assert response.json()["error"]["code"] == expected_code
    assert response.json()["error"]["retryable"] is False


def test_invalid_json_media_type_size_and_fields_are_bounded(tmp_path: Path) -> None:
    service = _service(tmp_path)
    invalid_json = asyncio.run(
        _request(
            service,
            "POST",
            "/v1/hint",
            content=b"{broken",
            headers={"content-type": "application/json"},
        )
    )
    wrong_media = asyncio.run(
        _request(
            service,
            "POST",
            "/v1/hint",
            content=b"{}",
            headers={"content-type": "text/plain"},
        )
    )
    oversized = asyncio.run(
        _request(
            service,
            "POST",
            "/v1/hint",
            content=json.dumps({"padding": "x" * 33_000}).encode(),
            headers={"content-type": "application/json"},
        )
    )
    invalid_fields = asyncio.run(
        _request(
            service,
            "POST",
            "/v1/hint",
            json={
                **_base_payload(),
                "question_id": "data_quality_concept_01",
                "hint_level": 0,
                "secret_answer": "must-not-echo",
            },
        )
    )

    assert (invalid_json.status_code, invalid_json.json()["error"]["code"]) == (
        400,
        "invalid_json",
    )
    assert (wrong_media.status_code, wrong_media.json()["error"]["code"]) == (
        415,
        "unsupported_media_type",
    )
    assert (oversized.status_code, oversized.json()["error"]["code"]) == (
        413,
        "payload_too_large",
    )
    assert invalid_fields.status_code == 422
    assert set(invalid_fields.json()["error"]["invalid_fields"]) == {
        "hint_level",
        "secret_answer",
    }
    assert "must-not-echo" not in invalid_fields.text


def test_unknown_question_and_service_failure_do_not_leak_internals(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    payload = {
        **_base_payload(),
        "anonymous_profile_id": PROFILE_ID,
        "question_id": "unknown_question_01",
        "submission": {"answer": "2"},
        "hint_level": 0,
    }
    missing = asyncio.run(_request(service, "POST", "/v1/diagnose", json=payload))

    async def fail_diagnosis(*args: object, **kwargs: object) -> None:
        raise RuntimeError("database-password-and-provider-detail")

    service.tutor.diagnose = fail_diagnosis  # type: ignore[method-assign]
    payload["question_id"] = "data_quality_concept_01"
    unavailable = asyncio.run(_request(service, "POST", "/v1/diagnose", json=payload))

    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "not_found"
    assert unavailable.status_code == 503
    assert unavailable.json()["error"] == {
        "code": "service_unavailable",
        "message_zh": "诊断服务暂时不可用，请稍后使用相同幂等键重试。",
        "retryable": True,
        "invalid_fields": [],
    }
    assert "database-password" not in unavailable.text
    assert service.get_dashboard(PROFILE_ID, ConceptId.DATA_QUALITY).state.history == ()


def test_recommendation_state_conflict_is_structured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)

    def no_question(*args: object, **kwargs: object) -> None:
        raise LearningRecommendationUnavailableError("private policy detail")

    monkeypatch.setattr(service, "choose_question", no_question)
    response = asyncio.run(
        _request(
            service,
            "POST",
            "/v1/recommend",
            json={
                **_base_payload(),
                "anonymous_profile_id": PROFILE_ID,
                "concept_id": "data_quality",
            },
        )
    )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "state_conflict"
    assert "private policy detail" not in response.text


def test_recommendation_storage_failure_is_safe_and_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)

    def fail_load(_learner_id: str) -> None:
        raise sqlite3.OperationalError("secret-storage-path")

    monkeypatch.setattr(service.store, "load", fail_load)
    response = asyncio.run(
        _request(
            service,
            "POST",
            "/v1/recommend",
            json={
                **_base_payload(),
                "anonymous_profile_id": PROFILE_ID,
                "concept_id": "data_quality",
            },
        )
    )

    assert response.status_code == 503
    assert response.json()["error"] == {
        "code": "service_unavailable",
        "message_zh": "推荐服务暂时不可用，请稍后使用相同幂等键重试。",
        "retryable": True,
        "invalid_fields": [],
    }
    assert "secret-storage-path" not in response.text


def test_api_rejects_names_and_emails_as_profile_identifiers(tmp_path: Path) -> None:
    payload = {
        **_base_payload(),
        "anonymous_profile_id": "learner@example.test",
        "concept_id": "data_quality",
    }

    response = asyncio.run(
        _request(_service(tmp_path), "POST", "/v1/recommend", json=payload)
    )

    assert response.status_code == 422
    assert response.json()["error"]["invalid_fields"] == ["anonymous_profile_id"]
    assert "learner@example.test" not in response.text


def test_local_api_runner_preflight_does_not_open_a_socket() -> None:
    result = subprocess.run(
        [
            str(PROJECT_ROOT / ".venv" / "bin" / "python"),
            str(PROJECT_ROOT / "scripts" / "run_local_api.py"),
            "--check",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "未打开端口" in result.stdout


def test_runner_is_loopback_only_and_has_no_public_host_argument() -> None:
    content = (PROJECT_ROOT / "scripts" / "run_local_api.py").read_text(encoding="utf-8")

    assert 'LOCAL_API_HOST = "127.0.0.1"' in content
    assert "--host" not in content
    assert "0.0.0.0" not in content


def test_checked_in_openapi_matches_pydantic_contract() -> None:
    path = PROJECT_ROOT / "docs" / "api" / "openapi.json"

    assert json.loads(path.read_text(encoding="utf-8")) == build_openapi_contract()


def test_openapi_covers_every_public_route_and_marks_public_deployment_unsafe() -> None:
    contract = build_openapi_contract()

    assert set(contract["paths"]) == {
        "/health",
        "/v1/diagnose",
        "/v1/hint",
        "/v1/recommend",
    }
    assert contract["x-public-deployment-ready"] is False
    recommend_operation = contract["paths"]["/v1/recommend"]["post"]
    assert {"409", "503"} <= set(recommend_operation["responses"])
    schemas = contract["components"]["schemas"]
    assert {"DiagnoseRequest", "DiagnoseResponse", "ApiErrorResponse"} <= set(schemas)


def test_openapi_exactly_matches_runtime_methods_headers_and_statuses(
    tmp_path: Path,
) -> None:
    app = create_api_app(_service(tmp_path))
    runtime_methods = {
        route.path: {method.casefold() for method in route.methods or set()}
        for route in app.routes
        if isinstance(route, Route)
    }
    contract = build_openapi_contract()
    contract_paths = contract["paths"]
    contract_methods = {
        path: set(path_item)
        for path, path_item in contract_paths.items()
    }

    assert contract_methods == runtime_methods
    expected_statuses = {
        ("/health", "get"): {"200", "500"},
        ("/health", "head"): {"200", "500"},
        ("/v1/diagnose", "post"): {
            "200",
            "400",
            "404",
            "409",
            "413",
            "415",
            "422",
            "500",
            "503",
        },
        ("/v1/hint", "post"): {
            "200",
            "400",
            "404",
            "413",
            "415",
            "422",
            "500",
        },
        ("/v1/recommend", "post"): {
            "200",
            "400",
            "409",
            "413",
            "415",
            "422",
            "500",
            "503",
        },
    }
    for (path, method), statuses in expected_statuses.items():
        operation = contract_paths[path][method]
        assert set(operation["responses"]) == statuses
        assert operation["parameters"] == [
            {
                "name": "X-Request-ID",
                "in": "header",
                "required": False,
                "description": (
                    "用于健康检查及请求正文校验前错误的可选追踪标识；"
                    "有效 POST 请求以正文 request_id 为准。"
                ),
                "schema": {
                    "type": "string",
                    "pattern": r"^[A-Za-z0-9][A-Za-z0-9_-]{7,127}$",
                },
            }
        ]


def test_health_head_is_explicit_and_has_no_response_body(tmp_path: Path) -> None:
    response = asyncio.run(_request(_service(tmp_path), "HEAD", "/health"))

    assert response.status_code == 200
    assert response.content == b""
    assert response.headers["cache-control"] == "no-store"
