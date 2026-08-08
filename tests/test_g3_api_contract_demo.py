"""Acceptance test for the no-network G3.4 API simulation."""

import asyncio
from pathlib import Path

from scripts.g3_api_contract_demo import run_demo


def test_g3_api_contract_demo_passes_without_external_network(tmp_path: Path) -> None:
    output = tmp_path / "g3_4_api_contract_result.json"

    summary = asyncio.run(run_demo(output))

    assert summary.passed is True
    assert summary.transport == "asgi_in_process_no_external_network"
    assert summary.health_ok is True
    assert summary.offline_core_available is True
    assert summary.recommendation_ok is True
    assert summary.level_one_safe is True
    assert summary.level_four_complete is True
    assert summary.diagnosis_correct is True
    assert summary.idempotent_history_count == 1
    assert summary.invalid_request_status == 422
    assert summary.unavailable_status == 503
    assert summary.unavailable_retryable is True
    assert output.exists()
