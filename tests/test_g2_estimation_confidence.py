"""Content, grading, AST, leakage, source, and graph checks for G2.7."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from probstat_tutor.curriculum import load_default_question_bank
from probstat_tutor.curriculum_graph import (
    TextbookId,
    UnitContentStatus,
    load_default_curriculum_catalog,
)
from probstat_tutor.graders import (
    combine_submission_evidence,
    grade_dataframe_result,
    grade_multiple_choice,
)
from probstat_tutor.rag import load_rag_manifest, load_rag_source
from probstat_tutor.schemas import (
    ContentReviewStatus,
    DeepUnitId,
    LearnerSubmission,
    Question,
    QuestionType,
)

ROOT = Path(__file__).resolve().parents[1]
UNIT_ID = DeepUnitId.ESTIMATION_CONFIDENCE_INTERVAL


def _questions() -> tuple[Question, ...]:
    return tuple(
        question
        for question in load_default_question_bank().questions
        if question.unit_id == UNIT_ID
    )


def _question(question_id: str) -> Question:
    return next(question for question in _questions() if question.id == question_id)


def _grade_choice(question: Question, submission: LearnerSubmission):
    return combine_submission_evidence(
        question,
        submission,
        grade_multiple_choice(
            submission.answer,
            str(question.expected_answer),
            accepted_answers=question.accepted_answers,
            misconception_candidates=question.misconception_tags,
        ),
    )


def _grade_interval(question: Question, submission: LearnerSubmission):
    answer_result = grade_dataframe_result(
        actual=pd.DataFrame([[46.08, 53.92]]),
        expected=pd.DataFrame([question.expected_answer]),
        absolute_tolerance=question.numeric_tolerance or 0.0,
    )
    return combine_submission_evidence(question, submission, answer_result)


def _estimator_reasoning() -> str:
    return (
        "估计量是由样本计算并在重复抽样中变化的统计规则，估计值是一次实现。"
        "A无偏，方差和MSE都是6平方等于36；B偏差为103-100=3，"
        "方差是2平方等于4，MSE=4+3平方=13。13小于36，"
        "所以B虽有偏但均方误差更小，无偏不保证MSE最小。"
    )


def _width_reasoning() -> str:
    return (
        "B样本量是A的四倍，20/根号100=2而20/根号400=1，"
        "所以标准误减半且B区间通常比A窄。99%的临界值更大，C通常比B更宽。"
        "D便利样本存在选择偏差，增加样本量或得到窄区间都不能消除；"
        "公式宽度相同不代表无偏，名义95%也不保证有偏设计的实际覆盖。"
    )


def test_unit_has_five_pending_questions_two_sources_and_all_nodes() -> None:
    questions = _questions()
    catalog = load_default_curriculum_catalog()
    unit = next(unit for unit in catalog.units if unit.unit_id == UNIT_ID)

    assert len(questions) == 5
    assert {question.question_type for question in questions} == set(QuestionType)
    assert {question.id for question in questions} == set(unit.question_ids)
    assert all(question.hints is not None for question in questions)
    assert all(question.knowledge_node_ids for question in questions)
    assert set(unit.knowledge_node_ids) == {
        node_id for question in questions for node_id in question.knowledge_node_ids
    }
    assert all(
        question.review_status == ContentReviewStatus.PENDING_TEACHER_REVIEW
        for question in questions
    )
    assert {
        question.id for question in questions if question.python_code_required
    } == {"confidence_interval_python_01"}
    assert unit.content_status == UnitContentStatus.PENDING_TEACHER_REVIEW
    assert set(unit.source_ids) == {
        "point_estimation_bias_variability_core",
        "confidence_interval_core",
    }


def test_frozen_question_ids_answers_and_tolerance_remain_compatible() -> None:
    concept = _question("confidence_interval_concept_01")
    python = _question("confidence_interval_python_01")
    interpretation = _question("confidence_interval_interpretation_01")

    assert concept.expected_answer == (
        "about_95_percent_of_intervals_cover_the_true_parameter"
    )
    assert python.expected_answer == [46.08, 53.92]
    assert python.numeric_tolerance == pytest.approx(0.000001)
    assert interpretation.expected_answer == (
        "no_the_interval_does_not_prove_values_outside_are_impossible"
    )

    legacy_concept = _grade_choice(
        concept,
        LearnerSubmission(
            answer="about_95_percent_of_intervals_cover_the_true_parameter",
            reasoning="重复抽样构造的许多区间中，约 95% 会覆盖真实参数。",
        ),
    )
    assert legacy_concept.is_correct is True

    legacy_interpretation = _grade_choice(
        interpretation,
        LearnerSubmission(
            answer="no_the_interval_does_not_prove_values_outside_are_impossible",
            reasoning="区间不是绝对边界，不能说 48 以下完全不可能。",
        ),
    )
    assert legacy_interpretation.is_correct is True


def test_coverage_requires_repetition_and_long_run_coverage() -> None:
    question = _question("confidence_interval_concept_01")
    correct = _grade_choice(
        question,
        LearnerSubmission(
            answer="A",
            reasoning=(
                "同一方法长期反复抽样，每次区间会随样本改变；"
                "长期覆盖率约为95%，也就是约95%的区间覆盖参数。"
                "不能把它理解成固定参数在这一次区间里的概率。"
            ),
        ),
    )
    assert correct.is_correct is True

    incomplete_cases = (
        "95%的区间覆盖参数。",
        "只要重复抽样就行。",
        "覆盖率。",
    )
    for reasoning in incomplete_cases:
        result = _grade_choice(
            question,
            LearnerSubmission(answer="A", reasoning=reasoning),
        )
        assert result.is_correct is False
        assert "missing_repeated_sampling_condition" in (
            result.misconception_candidates
        )


@pytest.mark.parametrize(
    ("reasoning", "expected_tag"),
    (
        ("参数有95%的概率在本次区间里。", "parameter_has_95_percent_probability"),
        ("区间包含95%的原始数据。", "data_coverage_confusion"),
        ("每100个区间必有95个覆盖。", "exact_coverage_guaranteed"),
    ),
)
def test_coverage_rejects_probability_data_and_exact_batch_claims(
    reasoning: str,
    expected_tag: str,
) -> None:
    question = _question("confidence_interval_concept_01")
    result = _grade_choice(
        question,
        LearnerSubmission(
            answer="A",
            reasoning=(
                "长期重复抽样，约95%的区间覆盖真实参数。" + reasoning
            ),
        ),
    )
    assert result.is_correct is False
    assert expected_tag in result.misconception_candidates


def test_coverage_negation_and_retraction_are_directional() -> None:
    question = _question("confidence_interval_concept_01")
    protected = _grade_choice(
        question,
        LearnerSubmission(
            answer="A",
            reasoning=(
                "长期反复抽样时约95%的区间覆盖真实参数；"
                "不能说参数有95%的概率在本次区间，也不是95%的原始数据。"
            ),
        ),
    )
    assert protected.is_correct is True

    corrected = _grade_choice(
        question,
        LearnerSubmission(
            answer="A",
            reasoning=(
                "参数有95%的概率在本次区间。上句话错误。"
                "相同方法反复抽样时，长期约95%的区间覆盖真实参数。"
            ),
        ),
    )
    assert corrected.is_correct is True

    reversal = _grade_choice(
        question,
        LearnerSubmission(
            answer="A",
            reasoning=(
                "相同方法反复抽样时，长期约95%的区间覆盖真实参数；"
                "不过参数在本次区间中的概率仍是95%。"
            ),
        ),
    )
    assert reversal.is_correct is False
    assert "parameter_has_95_percent_probability" in (
        reversal.misconception_candidates
    )

    terminal_retraction = _grade_choice(
        question,
        LearnerSubmission(
            answer="A",
            reasoning=(
                "重复抽样构造的许多区间中，约95%的区间覆盖真实参数。"
                "上面整句话都不对。"
            ),
        ),
    )
    assert terminal_retraction.is_correct is False
    assert "missing_repeated_sampling_condition" in (
        terminal_retraction.misconception_candidates
    )


def test_judge_natural_coverage_and_reversals_are_deterministic() -> None:
    question = _question("confidence_interval_concept_01")
    natural = (
        "每回重新取一批样本都会产生一对新端点；让这套程序运行很多轮，"
        "跨过那个固定真值的区间占比会趋近95%。"
        "这不是说眼前这一对端点给真值分配了0.95的机会。"
    )
    assert _grade_choice(
        question,
        LearnerSubmission(answer="A", reasoning=natural),
    ).is_correct is True

    base = "重复抽样构造的许多区间中，约95%的区间覆盖真实参数。"
    cases = (
        (
            "不过对眼前这一组端点，真值仍有0.95的机会落在里面。",
            "parameter_has_95_percent_probability",
        ),
        ("随便做100遍都会准确命中95遍。", "exact_coverage_guaranteed"),
        ("这也代表这段范围圈住总体里95%的个体。", "data_coverage_confusion"),
    )
    for reversal_text, expected_tag in cases:
        result = _grade_choice(
            question,
            LearnerSubmission(answer="A", reasoning=base + reversal_text),
        )
        assert result.is_correct is False
        assert expected_tag in result.misconception_candidates

    second_natural = (
        "在同样条件下不断重新采样，每轮得到的区间都不同。做得足够多后，"
        "覆盖那个固定总体参数的比例大约是百分之九十五；"
        "这说的是手续的频率表现，不是给本次参数位置赋概率。"
    )
    assert _grade_choice(
        question,
        LearnerSubmission(answer="A", reasoning=second_natural),
    ).is_correct is True

    second_cases = (
        (
            "然而固定参数落进现有端点的几率仍为九成五。",
            "parameter_has_95_percent_probability",
        ),
        (
            "做一百次时覆盖个数一定不多不少正好95。",
            "exact_coverage_guaranteed",
        ),
    )
    for reversal_text, expected_tag in second_cases:
        result = _grade_choice(
            question,
            LearnerSubmission(answer="A", reasoning=base + reversal_text),
        )
        assert result.is_correct is False
        assert expected_tag in result.misconception_candidates


def test_estimator_question_requires_rule_bias_variance_and_mse() -> None:
    question = _question("estimation_bias_variability_concept_01")
    assert _grade_choice(
        question,
        LearnerSubmission(answer="B", reasoning=_estimator_reasoning()),
    ).is_correct is True

    natural = _grade_choice(
        question,
        LearnerSubmission(
            answer="B",
            reasoning=(
                "估计量是样本的函数，反复取样会得到不同取值；某次得到的数才叫估计值。"
                "A的Bias为0，Var和MSE均为36；B的Bias为3，Var为4，"
                "所以MSE是4+3²=13。虽然B有系统偏移，但13<36，"
                "可见无偏不自动意味着平方误差风险最低。"
            ),
        ),
    )
    assert natural.is_correct is True

    judge_natural = _grade_choice(
        question,
        LearnerSubmission(
            answer="B",
            reasoning=(
                "估计量是把样本映射成数的函数，换一批样本输出会变；"
                "估计值是某批数据下的输出。A偏差0、方差36、MSE36；"
                "B偏差3、方差4、MSE=4+9=13，所以平方损失下B更小，"
                "不能仅凭A无偏就选A。"
            ),
        ),
    )
    assert judge_natural.is_correct is True

    second_judge_natural = _grade_choice(
        question,
        LearnerSubmission(
            answer="B",
            reasoning=(
                "估计量是一条从样本到数字的映射，重抽数据会输出另一个数；"
                "本次输出叫估计值。A的bias为0，variance和MSE都是36；"
                "B的bias为3，variance为4，MSE为13。因此按均方误差B胜出，"
                "尽管它的中心偏离目标。"
            ),
        ),
    )
    assert second_judge_natural.is_correct is True

    missing = _grade_choice(
        question,
        LearnerSubmission(answer="B", reasoning="B的MSE更小。"),
    )
    assert missing.is_correct is False
    assert "bias_variability_boundary_unexplained" in (
        missing.misconception_candidates
    )


@pytest.mark.parametrize(
    ("claim", "expected_tag"),
    (
        ("估计量就是某个固定估计值。", "estimator_confused_with_estimate"),
        ("A无偏所以MSE必然更小。", "unbiased_estimator_always_best"),
        ("MSE=方差+偏差，B的MSE为7。", "mse_omits_squared_bias"),
        ("B方差小所以一定无偏。", "low_variance_proves_unbiased"),
        ("一次估计高于100就证明有偏。", "one_estimate_proves_bias"),
    ),
)
def test_estimator_rejects_core_reversals(claim: str, expected_tag: str) -> None:
    question = _question("estimation_bias_variability_concept_01")
    result = _grade_choice(
        question,
        LearnerSubmission(answer="B", reasoning=f"{_estimator_reasoning()}{claim}"),
    )
    assert result.is_correct is False
    assert expected_tag in result.misconception_candidates


def test_estimator_whole_slot_negation_fails_and_correction_can_pass() -> None:
    question = _question("estimation_bias_variability_concept_01")
    negated = _grade_choice(
        question,
        LearnerSubmission(
            answer="B",
            reasoning=(
                "估计量是随机规则不正确；A偏差为0不成立，A的MSE是36也不对；"
                "B偏差为3错误，B的MSE为13不成立；B有偏但MSE更小也不是事实。"
            ),
        ),
    )
    assert negated.is_correct is False
    assert "bias_variability_boundary_unexplained" in (
        negated.misconception_candidates
    )

    corrected = _grade_choice(
        question,
        LearnerSubmission(
            answer="B",
            reasoning=(
                "无偏一定最好。上句话错误。" + _estimator_reasoning()
            ),
        ),
    )
    assert corrected.is_correct is True


@pytest.mark.parametrize(
    "code",
    (
        "mean + np.array([-1, 1]) * 1.96 * standard_error",
        "mean + 1.96 * standard_error * np.array([-1, 1])",
        "mean + np.array([-1, 1]) * critical_value * standard_error",
        "mean + np.array([-1.0, 1.0]) * critical_value * standard_error",
        (
            "np.array([mean - critical_value * standard_error, "
            "mean + critical_value * standard_error])"
        ),
        (
            "margin = 1.96 * standard_error\n"
            "lower = mean - margin\n"
            "upper = mean + margin\n"
            "np.array([lower, upper])"
        ),
    ),
)
def test_python_interval_accepts_exact_connected_equivalents(code: str) -> None:
    question = _question("confidence_interval_python_01")
    result = _grade_interval(
        question,
        LearnerSubmission(
            answer="[46.08, 53.92]",
            reasoning="误差界限为1.96×2=3.92，下限用减法，上限用加法。",
            python_code=code,
        ),
    )
    assert result.is_correct is True
    assert result.misconception_candidates == []


@pytest.mark.parametrize(
    ("code", "expected_tag"),
    (
        (
            "mean + np.array([1, 1]) * 1.96 * standard_error",
            "adds_margin_both_sides",
        ),
        (
            "mean + np.array([1, -1]) * 1.96 * standard_error",
            "reversed_interval_endpoints",
        ),
        (
            "mean + np.array([-1, 1]) * standard_error",
            "omits_critical_value",
        ),
        (
            "mean + np.array([-1, 1]) * 1.96 * standard_deviation",
            "uses_standard_deviation_as_ci_scale",
        ),
        ("np.array([46.08, 53.92])", "hardcoded_result_not_implementation"),
        (
            "mean + np.array([-1, 1], object) * 1.96 * standard_error",
            "hardcoded_result_not_implementation",
        ),
        (
            "mean + np.array([-1, 1], dtype=float) * 1.96 * standard_error",
            "hardcoded_result_not_implementation",
        ),
        (
            "critical_value = 0\n"
            "mean + np.array([-1, 1]) * critical_value * standard_error",
            "hardcoded_result_not_implementation",
        ),
        (
            "good = mean + np.array([-1, 1]) * 1.96 * standard_error\n"
            "np.array([0, 100])",
            "hardcoded_result_not_implementation",
        ),
        (
            "mean = 0\nmean + np.array([-1, 1]) * 1.96 * standard_error",
            "hardcoded_result_not_implementation",
        ),
        (
            "standard_error = 99\n"
            "mean + np.array([-1, 1]) * 1.96 * standard_error",
            "hardcoded_result_not_implementation",
        ),
        (
            "np = object()\n"
            "mean + np.array([-1, 1]) * critical_value * standard_error",
            "hardcoded_result_not_implementation",
        ),
        (
            "mean + holder.np.array([-1, 1]) * critical_value * standard_error",
            "hardcoded_result_not_implementation",
        ),
        (
            "mean += 100\n"
            "mean + np.array([-1, 1]) * critical_value * standard_error",
            "hardcoded_result_not_implementation",
        ),
        (
            "critical_value *= 0\n"
            "mean + np.array([-1, 1]) * critical_value * standard_error",
            "hardcoded_result_not_implementation",
        ),
        (
            "standard_error += 99\n"
            "mean + np.array([-1, 1]) * critical_value * standard_error",
            "hardcoded_result_not_implementation",
        ),
        (
            "(mean := 0)\n"
            "mean + np.array([-1, 1]) * critical_value * standard_error",
            "hardcoded_result_not_implementation",
        ),
        (
            "del mean\n"
            "mean + np.array([-1, 1]) * critical_value * standard_error",
            "hardcoded_result_not_implementation",
        ),
        (
            "import math as np\n"
            "mean + np.array([-1, 1]) * critical_value * standard_error",
            "hardcoded_result_not_implementation",
        ),
        (
            "for standard_error in [99]:\n"
            "    pass\n"
            "mean + np.array([-1, 1]) * critical_value * standard_error",
            "hardcoded_result_not_implementation",
        ),
        (
            "def mean():\n"
            "    return 0\n"
            "mean + np.array([-1, 1]) * critical_value * standard_error",
            "hardcoded_result_not_implementation",
        ),
        (
            "np.array = lambda values: values\n"
            "mean + np.array([-1, 1]) * critical_value * standard_error",
            "hardcoded_result_not_implementation",
        ),
    ),
)
def test_python_interval_rejects_specific_errors_and_bypasses(
    code: str,
    expected_tag: str,
) -> None:
    question = _question("confidence_interval_python_01")
    result = _grade_interval(
        question,
        LearnerSubmission(
            answer="[46.08, 53.92]",
            reasoning="误差界限为3.92，得到两个端点。",
            python_code=code,
        ),
    )
    assert result.is_correct is False
    assert result.misconception_candidates == [expected_tag]


def test_python_interval_is_static_and_never_calls_numpy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    question = _question("confidence_interval_python_01")
    answer_result = grade_dataframe_result(
        actual=pd.DataFrame([[46.08, 53.92]]),
        expected=pd.DataFrame([question.expected_answer]),
        absolute_tolerance=question.numeric_tolerance or 0.0,
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("学习者代码被执行")

    monkeypatch.setattr(np, "array", fail_if_called)
    result = combine_submission_evidence(
        question,
        LearnerSubmission(
            answer="[46.08, 53.92]",
            python_code=(
                "runner = eval\n"
                "runner('mean + np.array([-1,1]) * 1.96 * standard_error')"
            ),
        ),
        answer_result,
    )
    assert result.is_correct is False
    assert "unsafe_code_execution_request" in result.misconception_candidates


def test_interval_interpretation_accepts_boundary_and_rejects_object_confusion() -> None:
    question = _question("confidence_interval_interpretation_01")
    correct = _grade_choice(
        question,
        LearnerSubmission(
            answer="no_the_interval_does_not_prove_values_outside_are_impossible",
            reasoning=(
                "不能这样断言；区间会随样本变化，是当前数据下与总体均值相容的估计范围，"
                "不是绝对边界，也不是个体观测范围。"
            ),
        ),
    )
    assert correct.is_correct is True

    cases = (
        ("小于48完全不可能。", "outside_interval_impossible"),
        ("参数有95%概率在本区间。", "parameter_has_95_percent_probability"),
        ("48到52包含95%的原始数据。", "interval_contains_data"),
    )
    for claim, expected_tag in cases:
        result = _grade_choice(
            question,
            LearnerSubmission(
                answer="no_the_interval_does_not_prove_values_outside_are_impossible",
                reasoning=(
                    "不能断言，区间不是绝对边界；" + claim
                ),
            ),
        )
        assert result.is_correct is False
        assert expected_tag in result.misconception_candidates

    judge_natural = _grade_choice(
        question,
        LearnerSubmission(
            answer="no_the_interval_does_not_prove_values_outside_are_impossible",
            reasoning=(
                "不可以。48与52只是这份样本按该程序算出的端点，换一批样本就会变；"
                "低于48和当前证据不太相容，却并未被逻辑排除。"
            ),
        ),
    )
    assert judge_natural.is_correct is True

    zero_probability = _grade_choice(
        question,
        LearnerSubmission(
            answer="no_the_interval_does_not_prove_values_outside_are_impossible",
            reasoning=(
                "不能断言，区间不是绝对边界；区间外的参数值概率就是零。"
            ),
        ),
    )
    assert zero_probability.is_correct is False
    assert "outside_interval_impossible" in (
        zero_probability.misconception_candidates
    )

    personal_means = _grade_choice(
        question,
        LearnerSubmission(
            answer="no_the_interval_does_not_prove_values_outside_are_impossible",
            reasoning=(
                "不能断言，区间不是绝对边界；"
                "这也表示95%的个人均值都在48和52之间。"
            ),
        ),
    )
    assert personal_means.is_correct is False
    assert "interval_contains_data" in personal_means.misconception_candidates

    second_judge_natural = _grade_choice(
        question,
        LearnerSubmission(
            answer="no_the_interval_does_not_prove_values_outside_are_impossible",
            reasoning=(
                "不行。上下限由当前样本决定，重抽后会移动；"
                "48以下只与现有证据较不吻合，不能视作从可能集合中被排除。"
            ),
        ),
    )
    assert second_judge_natural.is_correct is True

    second_cases = (
        ("然而低于48的候选已经彻底出局。", "outside_interval_impossible"),
        (
            "也可把它当成未来个人观测的95%预测带。",
            "parameter_interval_treated_as_prediction_interval",
        ),
    )
    for reversal, expected_tag in second_cases:
        result = _grade_choice(
            question,
            LearnerSubmission(
                answer="no_the_interval_does_not_prove_values_outside_are_impossible",
                reasoning="不能断言，区间不是绝对边界；" + reversal,
            ),
        )
        assert result.is_correct is False
        assert expected_tag in result.misconception_candidates


def test_width_question_requires_size_confidence_and_bias_boundaries() -> None:
    question = _question("confidence_interval_width_bias_interpretation_01")
    assert _grade_choice(
        question,
        LearnerSubmission(answer="A", reasoning=_width_reasoning()),
    ).is_correct is True

    incomplete = _grade_choice(
        question,
        LearnerSubmission(answer="A", reasoning="B样本量更大，所以更好。"),
    )
    assert incomplete.is_correct is False
    assert "confidence_interval_width_boundary_unexplained" in (
        incomplete.misconception_candidates
    )

    judge_natural = _grade_choice(
        question,
        LearnerSubmission(
            answer="A",
            reasoning=(
                "B的400是A的100的4倍；A的SE=20/10=2，B=20/20=1，"
                "所以同为95%时B半宽更小。C报99%必须取更大的临界值，因此更宽。"
                "D虽公式宽度同B，却因便利样本中心偏高；增加n只能压低随机误差，"
                "不能把中心搬回真值，也不承诺真实覆盖。"
            ),
        ),
    )
    assert judge_natural.is_correct is True

    second_judge_natural = _grade_choice(
        question,
        LearnerSubmission(
            answer="A",
            reasoning=(
                "n由100变400，根号n翻倍，所以标准误从2降至1，同为95%的B范围更紧；"
                "相同数据改99%需更大分位点，所以C更宽。D的便利样本会把中心向上带，"
                "扩大规模只减少偶然起伏，纠正不了这种偏移，也不能确保命中真值。"
            ),
        ),
    )
    assert second_judge_natural.is_correct is True


@pytest.mark.parametrize(
    ("claim", "expected_tag"),
    (
        (
            "样本量四倍所以标准误变为四分之一。",
            "fourfold_n_quarters_interval_width",
        ),
        ("99%的区间更窄。", "higher_confidence_narrower_interval"),
        ("D区间窄所以选择偏差消失。", "narrow_interval_removes_bias"),
        (
            "标准差相同所以四个区间同样宽。",
            "equal_standard_deviation_means_equal_interval_width",
        ),
        ("标成95%就保证覆盖真值。", "nominal_confidence_guarantees_coverage"),
    ),
)
def test_width_question_rejects_core_reversals(
    claim: str,
    expected_tag: str,
) -> None:
    question = _question("confidence_interval_width_bias_interpretation_01")
    result = _grade_choice(
        question,
        LearnerSubmission(answer="A", reasoning=f"{_width_reasoning()}{claim}"),
    )
    assert result.is_correct is False
    assert expected_tag in result.misconception_candidates


def test_width_negation_protection_and_internal_reversal() -> None:
    question = _question("confidence_interval_width_bias_interpretation_01")
    protected = _grade_choice(
        question,
        LearnerSubmission(
            answer="A",
            reasoning=(
                f"{_width_reasoning()}不能说99%的区间更窄，"
                "也不能说D区间窄所以选择偏差消失。"
            ),
        ),
    )
    assert protected.is_correct is True

    reversal = _grade_choice(
        question,
        LearnerSubmission(
            answer="A",
            reasoning=f"{_width_reasoning()}不过99%的区间其实更窄。",
        ),
    )
    assert reversal.is_correct is False
    assert "higher_confidence_narrower_interval" in (
        reversal.misconception_candidates
    )

    cases = (
        (
            "提高置信度其实会同时缩短区间。",
            "higher_confidence_narrower_interval",
        ),
        (
            "便利抽样只要n足够大，偏移也会被平均掉。",
            "narrow_interval_removes_bias",
        ),
        (
            "公式同宽足以证明D也有95%的真实覆盖率。",
            "nominal_confidence_guarantees_coverage",
        ),
    )
    for reversal_text, expected_tag in cases:
        result = _grade_choice(
            question,
            LearnerSubmission(
                answer="A",
                reasoning=_width_reasoning() + reversal_text,
            ),
        )
        assert result.is_correct is False
        assert expected_tag in result.misconception_candidates

    second_cases = (
        (
            "因此n变成4倍，宽度会压到原来的25%。",
            "fourfold_n_quarters_interval_width",
        ),
        (
            "样本再多一点就能把便利抽样的方向错误稀释掉。",
            "narrow_interval_removes_bias",
        ),
    )
    for reversal_text, expected_tag in second_cases:
        result = _grade_choice(
            question,
            LearnerSubmission(
                answer="A",
                reasoning=_width_reasoning() + reversal_text,
            ),
        )
        assert result.is_correct is False
        assert expected_tag in result.misconception_candidates


def test_first_two_hints_do_not_reveal_g27_answers() -> None:
    forbidden_by_question = {
        "confidence_interval_concept_01": (
            "约 95% 的区间覆盖",
            "about_95_percent",
            "选择 a",
        ),
        "confidence_interval_python_01": (
            "3.92",
            "46.08",
            "53.92",
            "[-1, 1]",
        ),
        "confidence_interval_interpretation_01": (
            "不能断言",
            "不是绝对边界",
            "no_the_interval",
        ),
        "estimation_bias_variability_concept_01": (
            "mse=36",
            "mse=13",
            "13小于36",
            "选择 b",
        ),
        "confidence_interval_width_bias_interpretation_01": (
            "20/10=2",
            "20/20=1",
            "c通常比b更宽",
            "选择 a",
        ),
    }
    for question_id, forbidden_phrases in forbidden_by_question.items():
        hints = _question(question_id).hints
        assert hints is not None
        early_text = f"{hints.for_level(1)}\n{hints.for_level(2)}".casefold()
        assert not any(phrase.casefold() in early_text for phrase in forbidden_phrases)
        complete = hints.complete_explanation
        assert all(
            getattr(complete, field).strip()
            for field in ("concept", "calculation", "python", "interpretation")
        )


def test_g27_sources_are_registered_eligible_and_question_safe() -> None:
    manifest = load_rag_manifest(ROOT / "data" / "rag" / "manifest.yaml")
    entries = {
        entry.source_id: entry
        for entry in manifest.sources
        if entry.metadata.get("unit_id") == UNIT_ID.value
    }
    assert set(entries) == {
        "point_estimation_bias_variability_core",
        "confidence_interval_core",
    }
    forbidden_fingerprints = (
        "θ=100",
        "103-100=3",
        "MSE(A)=36",
        "MSE(B)=13",
        "mean=50",
        "standard_error=2",
        "46.08",
        "53.92",
        "[48,52]",
        "n=100、95%",
        "n=400、95%",
        "系统性高估总体均值的便利样本",
    )
    for entry in entries.values():
        loaded = load_rag_source(entry, ROOT)
        compact = (ROOT / entry.file_path).read_text(encoding="utf-8").replace(" ", "")
        assert loaded.eligibility.eligible_for_chunking is True
        assert entry.answer_leakage_risk.value == "low"
        assert entry.metadata["content_status"] == "pending_teacher_review"
        assert entry.metadata["reviewer_role"] == "course_teacher"
        assert not any(
            fingerprint.replace(" ", "") in compact
            for fingerprint in forbidden_fingerprints
        )


def test_g27_graph_edges_and_chapter_mapping_cover_all_nodes() -> None:
    catalog = load_default_curriculum_catalog()
    edges = {
        (edge.source_node_id, edge.target_node_id, edge.relation.value)
        for edge in catalog.edges
    }
    assert {
        ("si_population_sample", "ec_point_estimation", "prerequisite"),
        ("ec_point_estimation", "ec_bias_variability", "prerequisite"),
        ("ec_point_estimation", "ec_interval_construction", "prerequisite"),
        ("ec_bias_variability", "ec_interval_construction", "supports"),
        ("si_standard_error", "ec_interval_construction", "prerequisite"),
        ("ec_interval_construction", "ec_interval_interpretation", "supports"),
        ("ec_bias_variability", "ec_interval_interpretation", "supports"),
    } <= edges

    chapter = next(
        mapping
        for mapping in catalog.chapter_mappings
        if mapping.textbook_id == TextbookId.PROBABILITY_STATISTICS
        and mapping.chapter_number == 7
    )
    assert set(chapter.knowledge_node_ids) == {
        "ec_point_estimation",
        "ec_bias_variability",
        "ec_interval_construction",
        "ec_interval_interpretation",
    }
