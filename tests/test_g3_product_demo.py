"""Acceptance test for the five-run G3.3 product demonstration."""

import asyncio
from pathlib import Path

from scripts.g3_product_demo import run_demo


def test_five_product_journeys_and_model_fallback(tmp_path: Path) -> None:
    output_path = tmp_path / "g3_3_demo_result.json"

    summary = asyncio.run(run_demo(output_path))

    assert summary.passed is True
    assert summary.run_count == 5
    assert summary.serious_fault_count == 0
    assert summary.teacher_profile_count == 5
    assert summary.teacher_attempt_count == 10
    assert summary.teacher_raw_answer_fields_present is False
    assert summary.offline_core_available is True
    assert summary.model_failure_fallback_verified is True
    assert summary.fault_event_count == 1
    assert output_path.exists()
