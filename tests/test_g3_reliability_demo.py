"""Acceptance test for the no-network G3.5 reliability demonstration."""

import asyncio
import json
from pathlib import Path

from scripts.g3_reliability_demo import (
    DIAGNOSE_P95_LIMIT_MS,
    PERFORMANCE_SAMPLE_SIZE,
    run_demo,
)


def test_g35_reliability_faults_and_performance_gate(tmp_path: Path) -> None:
    output_path = tmp_path / "g3_5_reliability_result.json"

    summary = asyncio.run(run_demo(output_path))

    assert summary.passed is True
    assert summary.serious_fault_count == 0
    assert summary.transport == "asgi_in_process_no_external_network"
    assert summary.exact_replay_status == 200
    assert summary.changed_body_conflict_status == 409
    assert summary.history_after_conflict == 1
    assert summary.transient_retry_attempt_count == 2
    assert summary.transient_retry_final_status == 200
    assert summary.transient_retry_history_count == 1
    assert summary.transient_retry_internal_detail_hidden is True
    assert summary.atomic_rollback_status == 503
    assert summary.atomic_rollback_history_count == 0
    assert summary.model_timeout_attempt_count == 4
    assert summary.model_fallback_report_count == 3
    assert summary.model_circuit_state == "open"
    assert summary.model_health_status == "degraded"
    assert summary.fault_codes == (
        "model_timeout",
        "model_timeout",
        "model_circuit_open",
    )
    assert summary.performance_success_count == PERFORMANCE_SAMPLE_SIZE
    assert summary.performance_history_count == PERFORMANCE_SAMPLE_SIZE
    assert summary.diagnose_p95_ms <= DIAGNOSE_P95_LIMIT_MS
    assert json.loads(output_path.read_text(encoding="utf-8")) == summary.model_dump(
        mode="json"
    )
