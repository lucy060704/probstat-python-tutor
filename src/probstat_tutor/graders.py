"""Deterministic graders that never execute learner-submitted code."""

import math
import re
from collections.abc import Sequence

import pandas as pd

from probstat_tutor.schemas import GradeResult

NumericInput = int | float | str


def grade_numeric(
    actual: NumericInput,
    expected: NumericInput,
    *,
    absolute_tolerance: float = 0.0,
    relative_tolerance: float = 0.0,
    misconception_candidates: Sequence[str] = (),
) -> GradeResult:
    """Compare finite numbers, including learner input written as a percentage."""

    _validate_tolerances(absolute_tolerance, relative_tolerance)

    try:
        actual_value = _parse_number(actual)
    except (TypeError, ValueError) as error:
        return GradeResult(
            score=0.0,
            is_correct=False,
            errors=[f"无法把你的答案识别为有限数值：{error}"],
            misconception_candidates=list(misconception_candidates),
        )

    try:
        expected_value = _parse_number(expected)
    except (TypeError, ValueError) as error:
        return GradeResult(
            score=0.0,
            is_correct=False,
            errors=[f"标准答案配置无效，请联系维护者：{error}"],
        )

    difference = abs(actual_value - expected_value)
    is_correct = math.isclose(
        actual_value,
        expected_value,
        abs_tol=absolute_tolerance,
        rel_tol=relative_tolerance,
    )
    evidence = [
        f"你的数值：{actual_value:g}",
        f"标准数值：{expected_value:g}",
        f"绝对误差：{difference:g}",
        f"允许绝对误差：{absolute_tolerance:g}；允许相对误差：{relative_tolerance:g}",
    ]
    errors = [] if is_correct else ["数值与标准答案的差距超过了允许误差。"]

    return GradeResult(
        score=1.0 if is_correct else 0.0,
        is_correct=is_correct,
        evidence=evidence,
        errors=errors,
        misconception_candidates=[] if is_correct else list(misconception_candidates),
    )


def grade_multiple_choice(
    actual: str,
    expected: str,
    *,
    misconception_candidates: Sequence[str] = (),
) -> GradeResult:
    """Compare choice labels after normalizing case and whitespace."""

    actual_normalized = _normalize_text(actual)
    expected_normalized = _normalize_text(expected)
    if not actual_normalized:
        return GradeResult(
            score=0.0,
            is_correct=False,
            errors=["答案不能为空，请输入一个选项。"],
            misconception_candidates=list(misconception_candidates),
        )
    if not expected_normalized:
        return GradeResult(
            score=0.0,
            is_correct=False,
            errors=["标准选项配置为空，请联系维护者。"],
        )

    is_correct = actual_normalized == expected_normalized
    return GradeResult(
        score=1.0 if is_correct else 0.0,
        is_correct=is_correct,
        evidence=[f"标准化后的作答：{actual_normalized}"],
        errors=[] if is_correct else ["所选答案与标准选项不一致。"],
        misconception_candidates=[] if is_correct else list(misconception_candidates),
    )


def grade_text_keywords(
    actual: str,
    keywords: Sequence[str],
    *,
    evidence_score_cap: float = 0.5,
    misconception_candidates: Sequence[str] = (),
) -> GradeResult:
    """Collect keyword evidence without claiming complete understanding."""

    if not 0.0 <= evidence_score_cap < 1.0:
        raise ValueError("关键词辅助分上限必须大于等于 0 且小于 1。")

    normalized_answer = actual.casefold().strip()
    if not normalized_answer:
        return GradeResult(
            score=0.0,
            is_correct=False,
            errors=["文字答案不能为空，请写出你的判断或理由。"],
            misconception_candidates=list(misconception_candidates),
        )

    normalized_keywords = _unique_nonempty_keywords(keywords)
    if not normalized_keywords:
        return GradeResult(
            score=0.0,
            is_correct=False,
            errors=["关键词评分规则为空，无法形成辅助证据，请联系维护者。"],
        )

    matched = [keyword for keyword in normalized_keywords if keyword in normalized_answer]
    missing = [keyword for keyword in normalized_keywords if keyword not in normalized_answer]
    score = evidence_score_cap * len(matched) / len(normalized_keywords)
    evidence = [f"命中的辅助关键词：{', '.join(matched) if matched else '无'}"]
    if missing:
        evidence.append(f"未观察到的关键词：{', '.join(missing)}")
    evidence.append("关键词命中只能作为辅助证据，不能单独证明已经完全理解。")

    return GradeResult(
        score=score,
        is_correct=False,
        evidence=evidence,
        errors=[] if matched else ["暂未从回答中观察到预设的关键概念。"],
        misconception_candidates=[] if matched else list(misconception_candidates),
    )


def grade_dataframe_result(
    actual: object,
    expected: object,
    *,
    absolute_tolerance: float = 0.0,
    relative_tolerance: float = 0.0,
    misconception_candidates: Sequence[str] = (),
) -> GradeResult:
    """Compare DataFrame columns, shape, and values without comparing the index."""

    _validate_tolerances(absolute_tolerance, relative_tolerance)
    if not isinstance(actual, pd.DataFrame):
        return GradeResult(
            score=0.0,
            is_correct=False,
            errors=["你的结果不是 pandas DataFrame。"],
            misconception_candidates=list(misconception_candidates),
        )
    if not isinstance(expected, pd.DataFrame):
        return GradeResult(
            score=0.0,
            is_correct=False,
            errors=["标准结果不是 pandas DataFrame，请联系维护者。"],
        )

    actual_columns = list(actual.columns)
    expected_columns = list(expected.columns)
    columns_match = actual_columns == expected_columns
    shape_matches = actual.shape == expected.shape
    values_match = False
    evidence: list[str] = []
    errors: list[str] = []

    if columns_match:
        evidence.append(f"列名和顺序正确：{actual_columns}")
    else:
        errors.append(f"列名或顺序不一致；需要 {expected_columns}，得到 {actual_columns}。")

    if shape_matches:
        evidence.append(f"形状正确：{actual.shape}")
    else:
        errors.append(f"形状不一致；需要 {expected.shape}，得到 {actual.shape}。")

    if columns_match and shape_matches:
        try:
            pd.testing.assert_frame_equal(
                actual.reset_index(drop=True),
                expected.reset_index(drop=True),
                check_dtype=False,
                check_exact=False,
                atol=absolute_tolerance,
                rtol=relative_tolerance,
            )
        except AssertionError:
            errors.append("DataFrame 中至少有一个值与标准结果不一致。")
        else:
            values_match = True
            evidence.append("所有数据值都在允许误差范围内。")
    else:
        errors.append("列名和形状正确后，才能继续逐项比较数据值。")

    passed_checks = sum((columns_match, shape_matches, values_match))
    is_correct = passed_checks == 3
    return GradeResult(
        score=passed_checks / 3,
        is_correct=is_correct,
        evidence=evidence,
        errors=errors,
        misconception_candidates=[] if is_correct else list(misconception_candidates),
    )


def _parse_number(value: NumericInput) -> float:
    if isinstance(value, bool):
        raise TypeError("布尔值不能作为数值答案")

    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            raise ValueError("答案为空")
        is_percentage = normalized.endswith("%")
        if is_percentage:
            normalized = normalized[:-1].strip()
        try:
            parsed = float(normalized)
        except ValueError as error:
            raise ValueError(f"“{value}”不是有效数字") from error
        if is_percentage:
            parsed /= 100.0
    else:
        try:
            parsed = float(value)
        except (TypeError, ValueError) as error:
            raise TypeError(f"{value!r} 不是整数、浮点数或百分比") from error

    if not math.isfinite(parsed):
        raise ValueError("NaN 和无穷大不能作为答案")
    return parsed


def _validate_tolerances(absolute_tolerance: float, relative_tolerance: float) -> None:
    if not math.isfinite(absolute_tolerance) or absolute_tolerance < 0:
        raise ValueError("绝对误差必须是大于等于 0 的有限数值。")
    if not math.isfinite(relative_tolerance) or relative_tolerance < 0:
        raise ValueError("相对误差必须是大于等于 0 的有限数值。")


def _normalize_text(value: str) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip().casefold()


def _unique_nonempty_keywords(keywords: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    for keyword in keywords:
        cleaned = keyword.casefold().strip()
        if cleaned and cleaned not in normalized:
            normalized.append(cleaned)
    return normalized
