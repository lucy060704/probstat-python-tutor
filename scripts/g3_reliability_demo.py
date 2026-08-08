"""Run the G3.5 reliability gate without external network or ADP access."""

import argparse
import asyncio
import json
import math
import os
import platform
import sqlite3
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from probstat_tutor.api import create_api_app
from probstat_tutor.api.retry import BoundedApiRetryPolicy
from probstat_tutor.config import Settings
from probstat_tutor.reliability import CircuitState
from probstat_tutor.schemas import ConceptId, DeliveryMode
from probstat_tutor.service import LearningService

PERFORMANCE_SAMPLE_SIZE = 24
PERFORMANCE_CONCURRENCY = 6
DIAGNOSE_P95_LIMIT_MS = 3_000.0


class LocalEnvironment(BaseModel):
    """Non-identifying machine conditions needed to interpret local timings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    python_version: str
    operating_system: str
    machine: str
    logical_cpu_count: int | None = Field(ge=1)


class G35ReliabilitySummary(BaseModel):
    """Machine-readable G3.5 acceptance evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "1.0.0"
    transport: str = "asgi_in_process_no_external_network"
    environment: LocalEnvironment
    exact_replay_status: int
    changed_body_conflict_status: int
    history_after_conflict: int
    transient_retry_attempt_count: int
    transient_retry_final_status: int
    transient_retry_history_count: int
    transient_retry_internal_detail_hidden: bool
    atomic_rollback_status: int
    atomic_rollback_history_count: int
    model_timeout_attempt_count: int
    model_fallback_report_count: int
    model_circuit_state: str
    model_health_status: str
    fault_codes: tuple[str, ...]
    performance_sample_size: int
    performance_concurrency: int
    performance_success_count: int
    performance_history_count: int
    diagnose_p50_ms: float = Field(ge=0.0)
    diagnose_p95_ms: float = Field(ge=0.0)
    diagnose_max_ms: float = Field(ge=0.0)
    diagnose_p95_limit_ms: float = DIAGNOSE_P95_LIMIT_MS
    serious_fault_count: int = Field(ge=0)
    passed: bool


def _settings(root: Path, name: str, *, online: bool = False) -> Settings:
    return Settings(
        openai_api_key=SecretStr("synthetic-test-key") if online else None,
        openai_model="synthetic-test-model" if online else None,
        learning_state_db_path=root / f"{name}-learning.sqlite3",
        session_db_path=root / f"{name}-sessions.sqlite3",
        fault_log_path=root / name / "faults.jsonl",
        model_timeout_seconds=0.001 if online else 8.0,
        model_max_attempts=2,
        model_retry_base_delay_seconds=0.0,
        model_circuit_failure_threshold=2,
        model_circuit_open_seconds=30.0,
    )


def _diagnosis_payload(
    *,
    profile_id: str,
    request_id: str,
    idempotency_key: str,
    answer: str = "2",
) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "request_id": request_id,
        "idempotency_key": idempotency_key,
        "anonymous_profile_id": profile_id,
        "question_id": "data_quality_concept_01",
        "submission": {
            "answer": answer,
            "reasoning": "0 是合法分数，两个空白才是缺失。",
            "python_code": "",
        },
        "hint_level": 1,
    }


async def _exercise_idempotency(root: Path) -> tuple[int, int, int, bool]:
    service = LearningService(settings=_settings(root, "idempotency"))
    profile_id = "api_anon_1111111111111111"
    payload = _diagnosis_payload(
        profile_id=profile_id,
        request_id="request_g35_idempotency_001",
        idempotency_key="idem_g35_idempotency_001",
    )
    transport = httpx.ASGITransport(app=create_api_app(service))
    with patch(
        "probstat_tutor.tutor_agent.Runner.run",
        side_effect=AssertionError("离线可靠性演示不得调用在线模型"),
    ):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://local-reliability.test",
        ) as client:
            first = await client.post("/v1/diagnose", json=payload)
            replay = await client.post(
                "/v1/diagnose",
                json={**payload, "request_id": "request_g35_idempotency_002"},
            )
            changed = await client.post(
                "/v1/diagnose",
                json={
                    **payload,
                    "request_id": "request_g35_idempotency_003",
                    "submission": {"answer": "3"},
                },
            )

    history_count = len(
        service.get_dashboard(profile_id, ConceptId.DATA_QUALITY).state.history
    )
    replay_matches = (
        first.status_code == replay.status_code == 200
        and first.json()["report"] == replay.json()["report"]
    )
    no_leak = "request_g35_idempotency_001" not in changed.text
    return replay.status_code, changed.status_code, history_count, replay_matches and no_leak


async def _exercise_atomic_rollback(root: Path) -> tuple[int, int, bool]:
    service = LearningService(settings=_settings(root, "rollback"))
    profile_id = "api_anon_2222222222222222"
    with sqlite3.connect(service.store.db_path) as connection:
        connection.execute(
            """
            CREATE TRIGGER fail_g35_receipt
            BEFORE INSERT ON submission_receipts
            BEGIN
                SELECT RAISE(ABORT, 'private-g35-storage-detail');
            END
            """
        )
    transport = httpx.ASGITransport(
        app=create_api_app(service),
        raise_app_exceptions=False,
    )
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://local-reliability.test",
    ) as client:
        response = await client.post(
            "/v1/diagnose",
            json=_diagnosis_payload(
                profile_id=profile_id,
                request_id="request_g35_rollback_001",
                idempotency_key="idem_g35_rollback_001",
            ),
        )

    history_count = len(
        service.get_dashboard(profile_id, ConceptId.DATA_QUALITY).state.history
    )
    return response.status_code, history_count, "private-g35-storage-detail" not in response.text


async def _exercise_transient_retry(root: Path) -> tuple[int, int, int, bool]:
    service = LearningService(settings=_settings(root, "transient-retry"))
    original_diagnose = service.tutor.diagnose
    diagnosis_calls = 0

    async def fail_once(*args: object, **kwargs: object) -> object:
        nonlocal diagnosis_calls
        diagnosis_calls += 1
        if diagnosis_calls == 1:
            raise RuntimeError("private-transient-detail")
        return await original_diagnose(*args, **kwargs)

    service.tutor.diagnose = fail_once  # type: ignore[method-assign]
    payload = _diagnosis_payload(
        profile_id="api_anon_4444444444444444",
        request_id="request_g35_transient_001",
        idempotency_key="idem_g35_transient_001",
    )
    policy = BoundedApiRetryPolicy(
        max_attempts=3,
        base_delay_seconds=0.001,
        max_delay_seconds=0.002,
    )
    attempts = 0
    transport = httpx.ASGITransport(
        app=create_api_app(service),
        raise_app_exceptions=False,
    )
    internal_detail_hidden = True
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://local-reliability.test",
    ) as client:
        while True:
            attempts += 1
            response = await client.post("/v1/diagnose", json=payload)
            internal_detail_hidden = (
                internal_detail_hidden and "private-transient-detail" not in response.text
            )
            error = response.json().get("error", {})
            decision = policy.decide(
                method="POST",
                completed_attempts=attempts,
                status_code=response.status_code,
                retryable=bool(error.get("retryable", False)),
                original_idempotency_key="idem_g35_transient_001",
                current_idempotency_key=str(payload["idempotency_key"]),
            )
            if not decision.should_retry:
                break
            await asyncio.sleep(decision.delay_seconds)

    history_count = len(
        service.get_dashboard(
            "api_anon_4444444444444444", ConceptId.DATA_QUALITY
        ).state.history
    )
    return attempts, response.status_code, history_count, internal_detail_hidden


async def _exercise_model_reliability(
    root: Path,
) -> tuple[int, int, str, str, tuple[str, ...]]:
    service = LearningService(settings=_settings(root, "model", online=True))
    model_calls = 0

    async def too_slow(*args: object, **kwargs: object) -> None:
        nonlocal model_calls
        model_calls += 1
        await asyncio.sleep(0.05)

    reports = []
    with patch("probstat_tutor.tutor_agent.Runner.run", too_slow):
        for index in range(3):
            reports.append(
                await service.submit(
                    learner_id=f"api_anon_3{index:015x}",
                    session_id=f"model-session-{index}",
                    question_id="data_quality_concept_01",
                    answer="2",
                    reasoning="0 是合法分数，两个空白才是缺失。",
                    hint_level=1,
                    idempotency_key=f"idem_g35_model_{index:03d}",
                )
            )
    transport = httpx.ASGITransport(app=create_api_app(service))
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://local-reliability.test",
    ) as client:
        health = await client.get("/health")

    fault_codes = tuple(
        json.loads(line)["code"]
        for line in service.settings.fault_log_path.read_text(encoding="utf-8").splitlines()
    )
    fallback_count = sum(
        report.delivery_mode == DeliveryMode.MODEL_FALLBACK for report in reports
    )
    return (
        model_calls,
        fallback_count,
        service.tutor.model_circuit_state.value,
        health.json()["optional_model_status"],
        fault_codes,
    )


async def _exercise_performance(root: Path) -> tuple[int, int, list[float]]:
    service = LearningService(settings=_settings(root, "performance"))
    transport = httpx.ASGITransport(app=create_api_app(service))
    semaphore = asyncio.Semaphore(PERFORMANCE_CONCURRENCY)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://local-reliability.test",
    ) as client:

        async def diagnose(index: int) -> tuple[int, float]:
            async with semaphore:
                started = time.perf_counter()
                response = await client.post(
                    "/v1/diagnose",
                    json=_diagnosis_payload(
                        profile_id=f"api_anon_{index + 16:016x}",
                        request_id=f"request_g35_performance_{index:03d}",
                        idempotency_key=f"idem_g35_performance_{index:03d}",
                    ),
                )
                elapsed_ms = (time.perf_counter() - started) * 1_000.0
                return response.status_code, elapsed_ms

        with patch(
            "probstat_tutor.tutor_agent.Runner.run",
            side_effect=AssertionError("离线性能基线不得调用在线模型"),
        ):
            results = await asyncio.gather(
                *(diagnose(index) for index in range(PERFORMANCE_SAMPLE_SIZE))
            )

    success_count = sum(status == 200 for status, _latency in results)
    history_count = sum(
        len(
            service.get_dashboard(
                f"api_anon_{index + 16:016x}", ConceptId.DATA_QUALITY
            ).state.history
        )
        for index in range(PERFORMANCE_SAMPLE_SIZE)
    )
    return success_count, history_count, [latency for _status, latency in results]


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    rank = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[rank]


async def run_demo(output_path: Path | None = None) -> G35ReliabilitySummary:
    """Run deterministic faults and a local-only API timing baseline."""

    with TemporaryDirectory(prefix="probstat-g35-") as directory:
        root = Path(directory)
        replay_status, conflict_status, conflict_history, replay_safe = (
            await _exercise_idempotency(root)
        )
        rollback_status, rollback_history, rollback_safe = (
            await _exercise_atomic_rollback(root)
        )
        retry_attempts, retry_status, retry_history, retry_detail_hidden = (
            await _exercise_transient_retry(root)
        )
        (
            model_attempts,
            fallback_count,
            circuit_state,
            model_health_status,
            fault_codes,
        ) = await _exercise_model_reliability(root)
        success_count, performance_history, latencies = await _exercise_performance(root)

        p50_ms = _percentile(latencies, 0.50)
        p95_ms = _percentile(latencies, 0.95)
        max_ms = max(latencies)
        serious_faults = sum(
            (
                not replay_safe,
                replay_status != 200,
                conflict_status != 409,
                conflict_history != 1,
                retry_attempts != 2,
                retry_status != 200,
                retry_history != 1,
                not retry_detail_hidden,
                rollback_status != 503,
                rollback_history != 0,
                not rollback_safe,
                model_attempts != 4,
                fallback_count != 3,
                circuit_state != CircuitState.OPEN.value,
                model_health_status != "degraded",
                fault_codes
                != ("model_timeout", "model_timeout", "model_circuit_open"),
                success_count != PERFORMANCE_SAMPLE_SIZE,
                performance_history != PERFORMANCE_SAMPLE_SIZE,
                p95_ms > DIAGNOSE_P95_LIMIT_MS,
            )
        )
        summary = G35ReliabilitySummary(
            environment=LocalEnvironment(
                python_version=platform.python_version(),
                operating_system=platform.system(),
                machine=platform.machine(),
                logical_cpu_count=os.cpu_count(),
            ),
            exact_replay_status=replay_status,
            changed_body_conflict_status=conflict_status,
            history_after_conflict=conflict_history,
            transient_retry_attempt_count=retry_attempts,
            transient_retry_final_status=retry_status,
            transient_retry_history_count=retry_history,
            transient_retry_internal_detail_hidden=retry_detail_hidden,
            atomic_rollback_status=rollback_status,
            atomic_rollback_history_count=rollback_history,
            model_timeout_attempt_count=model_attempts,
            model_fallback_report_count=fallback_count,
            model_circuit_state=circuit_state,
            model_health_status=model_health_status,
            fault_codes=fault_codes,
            performance_sample_size=PERFORMANCE_SAMPLE_SIZE,
            performance_concurrency=PERFORMANCE_CONCURRENCY,
            performance_success_count=success_count,
            performance_history_count=performance_history,
            diagnose_p50_ms=round(p50_ms, 3),
            diagnose_p95_ms=round(p95_ms, 3),
            diagnose_max_ms=round(max_ms, 3),
            serious_fault_count=serious_faults,
            passed=serious_faults == 0,
        )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(f"{summary.model_dump_json(indent=2)}\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = asyncio.run(run_demo(args.output))
    print(summary.model_dump_json(indent=2))
    return 0 if summary.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
