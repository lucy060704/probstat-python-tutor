"""Validation and isolation tests for the v0.2 evaluation data sets."""

import sys
from pathlib import Path

import pytest

import evals.run_evals as runner
from evals.dataset import (
    CoverageGroup,
    DifficultyLevel,
    SourceType,
    analyze_v2_distribution,
    load_v2_cases,
    normalized_input_sha256,
    validate_combined_v2_cases,
)
from probstat_tutor.schemas import CapabilityDimension, ConceptId

ROOT = Path(__file__).resolve().parents[1]
V1_CASES_PATH = ROOT / "evals" / "cases.jsonl"
DEV_CASES_PATH = ROOT / "evals" / "development" / "cases_v0.2_dev.jsonl"
BLIND_CASES_PATH = ROOT / "evals" / "blind" / "cases_v0.2_blind.jsonl"
FROZEN_V1_SHA256 = "0c761826f86e4c8fbf1d716b796574ec2f96230c422e1f0c1568c6ecc918e4a6"
FROZEN_V2_CONCEPTS = {
    ConceptId.MEAN_MEDIAN,
    ConceptId.VARIANCE_STD,
    ConceptId.SAMPLING_STANDARD_ERROR,
    ConceptId.CONFIDENCE_INTERVAL,
}


@pytest.fixture
def v2_cases():
    development = load_v2_cases(DEV_CASES_PATH, expected_split="development")
    blind = load_v2_cases(BLIND_CASES_PATH, expected_split="blind")
    return development, blind


def test_three_datasets_have_valid_schemas(v2_cases) -> None:
    legacy = runner.load_cases(V1_CASES_PATH)
    development, blind = v2_cases

    assert len(legacy) == 36
    assert len(development) == 32
    assert len(blind) == 16
    assert {case.split for case in development} == {"development"}
    assert {case.split for case in blind} == {"blind"}


def test_frozen_v1_cases_are_unchanged() -> None:
    cases = runner.load_cases(V1_CASES_PATH)

    assert runner.normalized_case_sha256(cases) == FROZEN_V1_SHA256


def test_ids_are_unique_across_all_three_datasets(v2_cases) -> None:
    legacy = runner.load_cases(V1_CASES_PATH)
    development, blind = v2_cases

    validate_combined_v2_cases(
        development,
        blind,
        legacy_ids={case.id for case in legacy},
    )


def test_learner_inputs_are_not_exact_duplicates(v2_cases) -> None:
    development, blind = v2_cases
    hashes = [normalized_input_sha256(case) for case in [*development, *blind]]

    assert len(hashes) == len(set(hashes))


def test_knowledge_points_and_capability_dimensions_are_legal(v2_cases) -> None:
    development, blind = v2_cases
    all_cases = [*development, *blind]

    assert {case.concept_id for case in all_cases} == FROZEN_V2_CONCEPTS
    assert {case.capability_dimension for case in all_cases} == set(
        CapabilityDimension
    )


def test_source_type_and_difficulty_level_are_legal(v2_cases) -> None:
    development, blind = v2_cases
    all_cases = [*development, *blind]

    assert {case.source_type for case in all_cases} == set(SourceType)
    assert {case.difficulty_level for case in all_cases} == set(DifficultyLevel)


def test_approved_distribution_is_frozen(v2_cases) -> None:
    development, blind = v2_cases
    distribution = analyze_v2_distribution([*development, *blind])

    assert distribution.by_concept_id == {
        "mean_median": 12,
        "variance_std": 12,
        "sampling_standard_error": 12,
        "confidence_interval": 12,
    }
    assert distribution.by_capability_dimension == {
        "concept": 12,
        "calculation": 12,
        "python": 12,
        "interpretation": 12,
    }
    assert distribution.by_coverage_group == {
        CoverageGroup.EXISTING_ERROR_VARIANT.value: 20,
        CoverageGroup.NEW_MISCONCEPTION.value: 12,
        CoverageGroup.BOUNDARY_CASE.value: 8,
        CoverageGroup.ADVERSARIAL_INPUT.value: 8,
    }


def test_required_response_types_are_covered(v2_cases) -> None:
    development, blind = v2_cases
    categories = {
        category for case in [*development, *blind] for category in case.categories
    }

    assert {
        "fully_correct",
        "concept_correct_code_wrong",
        "code_correct_interpretation_wrong",
        "calculation_correct_conclusion_wrong",
        "pandas_numpy_api_error",
        "sd_se_confusion",
        "ci_probability_misinterpretation",
        "ignores_outlier",
        "correlation_as_causation",
        "correct_after_hint",
        "irrelevant",
        "insufficient_information",
        "prompt_injection",
        "score_tampering",
        "looks_correct_missing_condition",
    } <= categories


def test_variant_families_do_not_cross_development_and_blind(v2_cases) -> None:
    development, blind = v2_cases
    development_families = {case.variant_family for case in development}
    blind_families = {case.variant_family for case in blind}

    assert development_families.isdisjoint(blind_families)


def test_default_runner_loads_only_frozen_v1_cases() -> None:
    cases = runner.load_named_dataset("v0.1")

    assert len(cases) == 36
    assert all(case.id.startswith("case_") for case in cases)
    assert not any(case.id.startswith(("dev_", "blind_")) for case in cases)


def test_named_runner_loads_each_v2_split_explicitly() -> None:
    development = runner.load_named_dataset("development")
    blind = runner.load_named_dataset("blind")

    assert len(development) == 32
    assert len(blind) == 16
    assert all(case.id.startswith("dev_") for case in development)
    assert all(case.id.startswith("blind_") for case in blind)


def test_blind_runner_prints_aggregate_only(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def fake_run_evaluations(cases, workdir):
        del cases, workdir
        zero = runner.Metric(value=0.0, numerator=0, denominator=16)
        return runner.EvalSummary(
            deterministic_grading_accuracy=zero,
            misconception_tag_accuracy=zero,
            recommended_action_match_rate=zero,
            level_one_hint_leak_rate=runner.Metric(
                value=0.0,
                numerator=0,
                denominator=1,
            ),
            average_latency_ms=0.0,
            api_failure_rate=zero,
        )

    monkeypatch.setattr(runner, "run_evaluations", fake_run_evaluations)
    monkeypatch.setattr(sys, "argv", ["run_evals.py", "--dataset", "blind"])

    assert runner.main() == 0
    output = capsys.readouterr().out
    assert "数据集: blind（16 个案例）" in output
    assert "blind_v02_" not in output
    assert "expected_correct" not in output
    assert "expected_misconception_tags" not in output


def test_blind_runner_rejects_limit_to_reduce_label_inference(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_evals.py", "--dataset", "blind", "--limit", "1"],
    )

    with pytest.raises(SystemExit) as error:
        runner.main()

    assert error.value.code == 2
    assert "盲测集禁止使用 --limit" in capsys.readouterr().err
