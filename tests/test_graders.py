"""Boundary-focused tests for deterministic graders."""

import math

import pandas as pd
import pytest

from probstat_tutor.graders import (
    grade_dataframe_result,
    grade_multiple_choice,
    grade_numeric,
    grade_text_keywords,
)
from probstat_tutor.schemas import GradeResult


@pytest.mark.parametrize(
    ("actual", "expected"),
    [(5, 5), (5.0, 5), (" 5.0 ", 5), ("25%", 0.25), (" 12.5 % ", 0.125)],
)
def test_grade_numeric_accepts_integer_float_and_percentage(
    actual: int | float | str, expected: int | float
) -> None:
    result = grade_numeric(actual, expected)

    assert isinstance(result, GradeResult)
    assert result.is_correct is True
    assert result.score == 1.0


def test_grade_numeric_supports_absolute_tolerance() -> None:
    result = grade_numeric(10.04, 10.0, absolute_tolerance=0.05)

    assert result.is_correct is True


def test_grade_numeric_supports_relative_tolerance() -> None:
    result = grade_numeric(101, 100, relative_tolerance=0.02)

    assert result.is_correct is True


def test_grade_numeric_rejects_value_outside_tolerance() -> None:
    result = grade_numeric(
        10.2,
        10.0,
        absolute_tolerance=0.1,
        misconception_candidates=["calculation_error"],
    )

    assert result.is_correct is False
    assert result.score == 0.0
    assert result.errors
    assert result.misconception_candidates == ["calculation_error"]


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf, "NaN", "inf"])
def test_grade_numeric_rejects_non_finite_values(invalid: float | str) -> None:
    result = grade_numeric(invalid, 1)

    assert result.is_correct is False
    assert any("NaN 和无穷大" in error for error in result.errors)


@pytest.mark.parametrize("invalid", ["", "abc", True])
def test_grade_numeric_handles_invalid_input(invalid: str | bool) -> None:
    result = grade_numeric(invalid, 1)

    assert result.is_correct is False
    assert result.errors


@pytest.mark.parametrize("tolerance", [-0.1, math.inf, math.nan])
def test_grade_numeric_rejects_invalid_tolerance(tolerance: float) -> None:
    with pytest.raises(ValueError, match="有限数值"):
        grade_numeric(1, 1, absolute_tolerance=tolerance)


def test_grade_multiple_choice_normalizes_case_and_whitespace() -> None:
    result = grade_multiple_choice("  Option   A ", "option a")

    assert result.is_correct is True
    assert result.score == 1.0


def test_grade_multiple_choice_reports_wrong_and_blank_answers() -> None:
    wrong = grade_multiple_choice("B", "A", misconception_candidates=["wrong_choice"])
    blank = grade_multiple_choice("   ", "A")

    assert wrong.is_correct is False
    assert wrong.misconception_candidates == ["wrong_choice"]
    assert blank.is_correct is False
    assert "不能为空" in blank.errors[0]


def test_grade_text_keywords_never_claims_complete_understanding() -> None:
    result = grade_text_keywords("中位数不容易受到异常值影响", ["中位数", "异常值"])

    assert result.score == 0.5
    assert result.is_correct is False
    assert any("辅助证据" in item for item in result.evidence)


def test_grade_text_keywords_returns_partial_evidence() -> None:
    result = grade_text_keywords("我选择中位数", ["中位数", "异常值"])

    assert result.score == 0.25
    assert result.is_correct is False


def test_grade_text_keywords_handles_no_match_blank_and_empty_rules() -> None:
    no_match = grade_text_keywords("我选择另一个统计量", ["中位数"])
    blank = grade_text_keywords(" ", ["中位数"])
    empty_rules = grade_text_keywords("中位数", [])

    assert no_match.score == 0.0
    assert no_match.errors
    assert blank.errors
    assert empty_rules.errors


def test_grade_text_keywords_rejects_full_score_cap() -> None:
    with pytest.raises(ValueError, match="小于 1"):
        grade_text_keywords("中位数", ["中位数"], evidence_score_cap=1.0)


def test_grade_dataframe_result_accepts_matching_values_and_ignores_index() -> None:
    actual = pd.DataFrame({"mean": [2.0, 3.0]}, index=[10, 11])
    expected = pd.DataFrame({"mean": [2, 3]})

    result = grade_dataframe_result(actual, expected)

    assert result.is_correct is True
    assert result.score == 1.0


def test_grade_dataframe_result_supports_float_tolerance() -> None:
    actual = pd.DataFrame({"mean": [1.0001, 2.0]})
    expected = pd.DataFrame({"mean": [1.0, 2.0]})

    result = grade_dataframe_result(actual, expected, absolute_tolerance=0.001)

    assert result.is_correct is True


def test_grade_dataframe_result_detects_column_order() -> None:
    actual = pd.DataFrame({"b": [2], "a": [1]})
    expected = pd.DataFrame({"a": [1], "b": [2]})

    result = grade_dataframe_result(actual, expected)

    assert result.is_correct is False
    assert any("列名或顺序" in error for error in result.errors)


def test_grade_dataframe_result_detects_shape() -> None:
    actual = pd.DataFrame({"value": [1]})
    expected = pd.DataFrame({"value": [1, 2]})

    result = grade_dataframe_result(actual, expected)

    assert result.is_correct is False
    assert any("形状不一致" in error for error in result.errors)


def test_grade_dataframe_result_detects_wrong_values() -> None:
    actual = pd.DataFrame({"value": [1, 99]})
    expected = pd.DataFrame({"value": [1, 2]})

    result = grade_dataframe_result(actual, expected)

    assert result.is_correct is False
    assert result.score == pytest.approx(2 / 3)
    assert any("至少有一个值" in error for error in result.errors)


def test_grade_dataframe_result_rejects_non_dataframe() -> None:
    result = grade_dataframe_result([1, 2], pd.DataFrame({"value": [1, 2]}))

    assert result.is_correct is False
    assert "不是 pandas DataFrame" in result.errors[0]
