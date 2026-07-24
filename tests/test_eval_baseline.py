"""Regression tests for the frozen v0.1 offline evaluation baseline."""

import json
from pathlib import Path

import pytest

from evals.run_evals import (
    EvaluationBaseline,
    analyze_case_distribution,
    baseline_mismatches,
    load_baseline,
    load_cases,
    normalized_case_sha256,
    write_baseline,
)

ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "evals" / "cases.jsonl"
BASELINE_PATH = ROOT / "evals" / "baselines" / "v0.1-mvp.json"
FROZEN_CASE_SHA256 = "0c761826f86e4c8fbf1d716b796574ec2f96230c422e1f0c1568c6ecc918e4a6"


def test_normalized_case_hash_ignores_field_order_and_line_endings(
    tmp_path: Path,
) -> None:
    raw_cases = [
        json.loads(line)
        for line in CASES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    reversed_fields = [dict(reversed(list(case.items()))) for case in raw_cases]
    reordered_path = tmp_path / "cases-with-crlf.jsonl"
    reordered_content = "\r\n".join(
        json.dumps(case, ensure_ascii=False) for case in reversed_fields
    )
    reordered_path.write_bytes(f"{reordered_content}\r\n".encode())

    original_hash = normalized_case_sha256(load_cases(CASES_PATH))
    reordered_hash = normalized_case_sha256(load_cases(reordered_path))

    assert original_hash == FROZEN_CASE_SHA256
    assert reordered_hash == original_hash


def test_case_distribution_records_current_imbalances() -> None:
    distribution = analyze_case_distribution(load_cases(CASES_PATH))

    assert distribution.by_concept_id == {
        "mean_median": 11,
        "variance_std": 7,
        "sampling_standard_error": 9,
        "confidence_interval": 9,
    }
    assert distribution.by_primary_dimension == {
        "concept": 10,
        "calculation": 0,
        "python": 16,
        "interpretation": 10,
    }
    assert distribution.by_observed_dimension == {
        "concept": 36,
        "calculation": 20,
        "python": 16,
        "interpretation": 20,
    }
    assert distribution.by_case_category == {
        "calculation_correct_conclusion_wrong": 3,
        "ci_probability_misinterpretation": 4,
        "code_correct_interpretation_wrong": 3,
        "concept_correct_code_wrong": 3,
        "correct_after_hint": 3,
        "fully_correct": 12,
        "insufficient_information": 2,
        "irrelevant": 2,
        "pandas_syntax_error": 3,
        "sd_se_confusion": 3,
    }
    assert sum(distribution.by_concept_id.values()) == 36
    assert sum(distribution.by_primary_dimension.values()) == 36
    assert sum(distribution.by_observed_dimension.values()) == 92
    assert sum(distribution.by_case_category.values()) == 38


def test_frozen_baseline_contains_real_metrics_and_required_metadata() -> None:
    baseline = load_baseline(BASELINE_PATH)
    metrics = baseline.metrics
    cases = load_cases(CASES_PATH)

    assert baseline.baseline_name == "v0.1-mvp-offline-eval"
    assert baseline.git_tag == "v0.1-mvp"
    assert baseline.evaluation_mode == "offline"
    assert baseline.case_count == 36
    assert baseline.normalized_case_sha256 == FROZEN_CASE_SHA256
    assert baseline.case_distribution == analyze_case_distribution(cases)
    assert baseline.python_version
    assert baseline.operating_system
    assert baseline.generated_at.utcoffset() is not None
    assert baseline.known_limitations

    assert (
        metrics.deterministic_grading_accuracy.numerator,
        metrics.deterministic_grading_accuracy.denominator,
    ) == (30, 36)
    assert (
        metrics.misconception_tag_accuracy.numerator,
        metrics.misconception_tag_accuracy.denominator,
    ) == (26, 36)
    assert (
        metrics.recommended_action_match_rate.numerator,
        metrics.recommended_action_match_rate.denominator,
    ) == (30, 36)
    assert (
        metrics.level_one_hint_leak_rate.numerator,
        metrics.level_one_hint_leak_rate.denominator,
    ) == (0, 3)
    assert (
        metrics.api_failure_rate.numerator,
        metrics.api_failure_rate.denominator,
    ) == (0, 36)
    assert "不代表真实 OpenAI API" in metrics.api_failure_rate.metric_scope
    assert metrics.average_latency_ms.value >= 0.0
    assert metrics.average_latency_ms.exact_regression is False
    assert "total" not in EvaluationBaseline.model_fields


def test_baseline_mismatch_messages_identify_each_drift_type() -> None:
    expected = load_baseline(BASELINE_PATH)
    current = expected.model_copy(deep=True)
    current.normalized_case_sha256 = "f" * 64
    current.case_count += 1
    current.case_distribution.by_concept_id["mean_median"] += 1
    current.metrics.deterministic_grading_accuracy.numerator = 29
    current.metrics.deterministic_grading_accuracy.value = 29 / 36

    messages = baseline_mismatches(expected, current)

    assert any(message.startswith("案例指纹漂移") for message in messages)
    assert any(message.startswith("案例数量漂移") for message in messages)
    assert any(message.startswith("案例分布漂移") for message in messages)
    assert any(
        "确定性指标漂移（deterministic_grading_accuracy）" in message
        for message in messages
    )


def test_latency_is_validated_but_not_exactly_compared() -> None:
    expected = load_baseline(BASELINE_PATH)
    current = expected.model_copy(deep=True)
    current.metrics.average_latency_ms.numerator = 360_000.0
    current.metrics.average_latency_ms.denominator = 36
    current.metrics.average_latency_ms.value = 10_000.0

    assert baseline_mismatches(expected, current) == []


def test_baseline_json_round_trip(tmp_path: Path) -> None:
    baseline = load_baseline(BASELINE_PATH)
    copied_path = tmp_path / "baseline-copy.json"

    write_baseline(copied_path, baseline)

    assert load_baseline(copied_path) == baseline
    assert load_baseline(copied_path).metrics.average_latency_ms.value == pytest.approx(
        baseline.metrics.average_latency_ms.value
    )
