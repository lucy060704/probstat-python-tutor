"""Exercise the G3.4 API contract in-process without network or ADP access."""

import argparse
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx
from pydantic import BaseModel, ConfigDict

from probstat_tutor.api import create_api_app
from probstat_tutor.config import Settings
from probstat_tutor.schemas import ConceptId
from probstat_tutor.service import LearningService


class ApiContractDemoSummary(BaseModel):
    """Machine-readable G3.4 acceptance evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "1.0.0"
    transport: str = "asgi_in_process_no_external_network"
    health_ok: bool
    offline_core_available: bool
    recommendation_ok: bool
    level_one_safe: bool
    level_four_complete: bool
    diagnosis_correct: bool
    idempotent_history_count: int
    invalid_request_status: int
    unavailable_status: int
    unavailable_retryable: bool
    passed: bool


def _settings(root: Path, name: str) -> Settings:
    return Settings(
        openai_api_key=None,
        openai_model=None,
        learning_state_db_path=root / f"{name}-learning.sqlite3",
        session_db_path=root / f"{name}-sessions.sqlite3",
        fault_log_path=root / name / "faults.jsonl",
    )


async def run_demo(output_path: Path | None = None) -> ApiContractDemoSummary:
    """Run public routes, one retry, validation failure, and service failure."""

    with TemporaryDirectory(prefix="probstat-g3-api-") as directory:
        root = Path(directory)
        service = LearningService(settings=_settings(root, "main"))
        profile_id = "api_anon_0123456789abcdef"
        base = {
            "schema_version": "1.0.0",
            "request_id": "request_demo_001",
            "idempotency_key": "idem_demo_001",
        }
        transport = httpx.ASGITransport(app=create_api_app(service))
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://local-contract.test",
        ) as client:
            health = await client.get(
                "/health",
                headers={"x-request-id": "health_demo_001"},
            )
            recommendation = await client.post(
                "/v1/recommend",
                json={
                    **base,
                    "anonymous_profile_id": profile_id,
                    "concept_id": ConceptId.DATA_QUALITY.value,
                },
            )
            question_id = recommendation.json()["question"]["id"]
            hint_one = await client.post(
                "/v1/hint",
                json={**base, "question_id": question_id, "hint_level": 1},
            )
            hint_four = await client.post(
                "/v1/hint",
                json={
                    **base,
                    "request_id": "request_demo_004",
                    "idempotency_key": "idem_demo_004",
                    "question_id": question_id,
                    "hint_level": 4,
                },
            )
            diagnosis_payload = {
                **base,
                "request_id": "request_demo_005",
                "idempotency_key": "idem_diagnose_001",
                "anonymous_profile_id": profile_id,
                "question_id": question_id,
                "submission": {
                    "answer": "2",
                    "reasoning": "0 是合法分数，两个空白才是缺失。",
                    "python_code": "",
                },
                "hint_level": 1,
            }
            diagnosis = await client.post("/v1/diagnose", json=diagnosis_payload)
            replay = await client.post("/v1/diagnose", json=diagnosis_payload)
            invalid = await client.post(
                "/v1/diagnose",
                json={**diagnosis_payload, "unexpected_secret": "must-not-echo"},
            )

        history_count = len(
            service.get_dashboard(profile_id, ConceptId.DATA_QUALITY).state.history
        )

        unavailable_service = LearningService(settings=_settings(root, "unavailable"))

        async def fail_diagnosis(*args: object, **kwargs: object) -> None:
            raise RuntimeError("internal provider detail must stay private")

        unavailable_service.tutor.diagnose = fail_diagnosis  # type: ignore[method-assign]
        unavailable_transport = httpx.ASGITransport(
            app=create_api_app(unavailable_service),
            raise_app_exceptions=False,
        )
        async with httpx.AsyncClient(
            transport=unavailable_transport,
            base_url="http://local-contract.test",
        ) as client:
            unavailable = await client.post("/v1/diagnose", json=diagnosis_payload)

        health_json = health.json()
        hint_one_json = hint_one.json()
        hint_four_json = hint_four.json()
        diagnosis_json = diagnosis.json()
        unavailable_json = unavailable.json()
        summary = ApiContractDemoSummary(
            health_ok=health.status_code == 200 and health_json["status"] == "ok",
            offline_core_available=health_json.get("offline_core_available") is True,
            recommendation_ok=(
                recommendation.status_code == 200
                and "expected_answer" not in recommendation.text
                and "rubric" not in recommendation.text
            ),
            level_one_safe=(
                hint_one.status_code == 200
                and hint_one_json["complete_explanation_revealed"] is False
                and "完整解释" not in hint_one_json["hint_zh"]
            ),
            level_four_complete=(
                hint_four.status_code == 200
                and hint_four_json["complete_explanation_revealed"] is True
                and hint_four_json["hint_zh"].startswith("完整解释：")
            ),
            diagnosis_correct=(
                diagnosis.status_code == 200
                and replay.status_code == 200
                and diagnosis_json["report"]["overall_correctness"] == 1.0
            ),
            idempotent_history_count=history_count,
            invalid_request_status=invalid.status_code,
            unavailable_status=unavailable.status_code,
            unavailable_retryable=unavailable_json["error"]["retryable"],
            passed=False,
        )
        summary = summary.model_copy(
            update={
                "passed": (
                    summary.health_ok
                    and summary.offline_core_available
                    and summary.recommendation_ok
                    and summary.level_one_safe
                    and summary.level_four_complete
                    and summary.diagnosis_correct
                    and summary.idempotent_history_count == 1
                    and summary.invalid_request_status == 422
                    and summary.unavailable_status == 503
                    and summary.unavailable_retryable
                    and "must-not-echo" not in invalid.text
                    and "internal provider detail" not in unavailable.text
                )
            }
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
