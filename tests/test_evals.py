"""Tests for evaluation case coverage and separate metric reporting."""

import asyncio
from pathlib import Path

from evals.run_evals import (
    REQUIRED_CATEGORIES,
    EvalSummary,
    load_cases,
    run_evaluations,
)

ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "evals" / "cases.jsonl"


def test_eval_cases_have_required_size_and_coverage() -> None:
    cases = load_cases(CASES_PATH)
    covered = {category for case in cases for category in case.categories}

    assert len(cases) >= 30
    assert REQUIRED_CATEGORIES <= covered
    assert len({case.id for case in cases}) == len(cases)


def test_eval_runner_returns_six_separate_metrics(tmp_path: Path) -> None:
    cases = load_cases(CASES_PATH)[:3]

    summary = asyncio.run(run_evaluations(cases, tmp_path))

    assert 0.0 <= summary.deterministic_grading_accuracy.value <= 1.0
    assert 0.0 <= summary.misconception_tag_accuracy.value <= 1.0
    assert 0.0 <= summary.recommended_action_match_rate.value <= 1.0
    assert 0.0 <= summary.level_one_hint_leak_rate.value <= 1.0
    assert summary.average_latency_ms >= 0.0
    assert 0.0 <= summary.api_failure_rate.value <= 1.0
    assert "total" not in EvalSummary.model_fields
