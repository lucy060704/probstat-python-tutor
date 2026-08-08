"""Content, grading, AST, synthetic-data, RAG, and graph checks for G2.8."""

from pathlib import Path

import pandas as pd
import pytest
from scipy import stats

from probstat_tutor.curriculum import load_default_question_bank
from probstat_tutor.curriculum_graph import (
    UnitContentStatus,
    load_default_curriculum_catalog,
)
from probstat_tutor.graders import (
    analyze_python_code,
    combine_submission_evidence,
    grade_multiple_choice,
    grade_numeric,
)
from probstat_tutor.rag import load_rag_manifest, load_rag_source
from probstat_tutor.schemas import (
    ContentReviewStatus,
    DeepUnitId,
    LearnerSubmission,
    PythonMismatchKind,
    PythonStructureKind,
    Question,
    QuestionType,
)

ROOT = Path(__file__).resolve().parents[1]
UNIT_ID = DeepUnitId.HYPOTHESIS_TESTING
QUESTION_IDS = {
    "hypothesis_p_value_concept_01",
    "ab_welch_ttest_python_01",
    "hypothesis_errors_power_concept_01",
    "ab_effect_significance_interpretation_01",
    "ab_design_multiplicity_interpretation_01",
}
SOURCE_IDS = {
    "hypothesis_testing_decisions_core",
    "ab_experiment_analysis_core",
}


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


def _grade_welch(submission: LearnerSubmission):
    question = _question("ab_welch_ttest_python_01")
    return combine_submission_evidence(
        question,
        submission,
        grade_numeric(
            submission.answer,
            question.expected_answer,
            absolute_tolerance=question.numeric_tolerance or 0.0,
            misconception_candidates=question.misconception_tags,
        ),
    )


def _p_value_reasoning() -> str:
    return (
        "H0是两组总体转化率相等，H1是双侧不相等。"
        "假定H0成立时，观察到当前或更极端结果的尾部概率为0.032。"
        "0.032<0.05，所以拒绝H0；p值不是H0为真的概率，"
        "统计显著也不说明效应很大，还要看效应量和区间。"
    )


def _welch_reasoning() -> str:
    return (
        "A组均值是71，B组是74.875，均值差为3.875。"
        "p值约0.002948，小于0.05，因此在0.05阈值下拒绝H0。"
        "这份合成小样本不能代表所有现实用户，因果解释还依赖随机分组完整执行，"
        "也应报告效应量与置信区间。"
    )


def _errors_power_reasoning() -> str:
    return (
        "实际无提升却误上线是H0真而拒绝的一类错误；"
        "存在真实提升却漏掉是H1真而未拒绝H0的二类错误。"
        "功效是在指定真实提升下拒绝H0的概率，也就是1-β。"
        "增加样本量通常提高功效，但不保证显著，也不会改变真实效应或消除抽样偏差。"
    )


def _effect_reasoning() -> str:
    return (
        "p=0.001小于0.05且95%区间排除0，所以差异统计显著。"
        "绝对差为0.15个百分点，相对提升为1.5%，二者口径不同。"
        "点估计和区间上限0.24都低于1个百分点，因此没有达到预设实际阈值。"
        "随机化支持本实验范围内的因果解释，但外推仍受受试人群和实施条件限制。"
    )


def _design_reasoning() -> str:
    return (
        "20次独立真H0都不假阳性的概率为0.95的20次方，"
        "所以至少一次假阳性是1-0.95^20，约64.15%。"
        "事后只挑p=0.03有多重比较和选择性报告风险，应预注册主指标或校正。"
        "用户自选不是随机分组，存在混杂；18%与5%的差异流失需要审计，"
        "不能删除后忽略，还应报告效应量、区间和敏感性分析。"
    )


def test_unit_has_five_pending_questions_two_sources_and_all_nodes() -> None:
    questions = _questions()
    catalog = load_default_curriculum_catalog()
    unit = next(unit for unit in catalog.units if unit.unit_id == UNIT_ID)

    assert len(questions) == 5
    assert {question.id for question in questions} == QUESTION_IDS
    assert {question.question_type for question in questions} == set(QuestionType)
    assert set(unit.question_ids) == QUESTION_IDS
    assert set(unit.source_ids) == SOURCE_IDS
    assert {
        node_id for question in questions for node_id in question.knowledge_node_ids
    } == set(unit.knowledge_node_ids)
    assert unit.content_status == UnitContentStatus.PENDING_TEACHER_REVIEW
    assert all(
        question.review_status == ContentReviewStatus.PENDING_TEACHER_REVIEW
        for question in questions
    )
    assert {
        question.id for question in questions if question.python_code_required
    } == {"ab_welch_ttest_python_01"}


def test_synthetic_dataset_is_small_private_free_and_matches_frozen_statistics() -> None:
    path = ROOT / "data" / "synthetic" / "ab_session_duration.csv"
    frame = pd.read_csv(path)

    assert list(frame.columns) == [
        "synthetic_id",
        "variant",
        "session_duration_minutes",
    ]
    assert len(frame) == 16
    assert frame["synthetic_id"].is_unique
    assert frame.isna().sum().sum() == 0
    assert frame.groupby("variant").size().to_dict() == {"A": 8, "B": 8}
    means = frame.groupby("variant")["session_duration_minutes"].mean()
    assert means["A"] == pytest.approx(71.0)
    assert means["B"] == pytest.approx(74.875)
    assert means["B"] - means["A"] == pytest.approx(3.875)
    assert all(identifier.startswith("SYN-") for identifier in frame.synthetic_id)


def test_maintainer_welch_result_matches_frozen_answer() -> None:
    frame = pd.read_csv(ROOT / "data" / "synthetic" / "ab_session_duration.csv")
    variant_a = frame.loc[frame.variant == "A", "session_duration_minutes"]
    variant_b = frame.loc[frame.variant == "B", "session_duration_minutes"]

    result = stats.ttest_ind(
        variant_b,
        variant_a,
        equal_var=False,
        alternative="two-sided",
    )

    assert float(result.statistic) == pytest.approx(3.6002033768285844)
    assert float(result.pvalue) == pytest.approx(
        _question("ab_welch_ttest_python_01").expected_answer,
        abs=1e-15,
    )


def test_p_value_question_accepts_complete_evidence() -> None:
    result = _grade_choice(
        _question("hypothesis_p_value_concept_01"),
        LearnerSubmission(answer="A", reasoning=_p_value_reasoning()),
    )

    assert result.is_correct is True


@pytest.mark.parametrize(
    "reasoning",
    (
        "0.032<0.05，所以拒绝H0。",
        "H0是两组总体转化率相等，H1是双侧不相等。",
        "p值不是H0为真的概率。",
    ),
)
def test_p_value_question_requires_all_evidence_groups(reasoning: str) -> None:
    result = _grade_choice(
        _question("hypothesis_p_value_concept_01"),
        LearnerSubmission(answer="A", reasoning=reasoning),
    )

    assert result.is_correct is False
    assert "p_value_context_unexplained" in result.misconception_candidates


@pytest.mark.parametrize(
    ("claim", "tag"),
    (
        ("H0为真的概率是3.2%。", "p_value_is_probability_null_true"),
        ("B更好的概率是96.8%。", "p_value_is_probability_treatment_better"),
        ("0.032大于0.05，所以不拒绝。", "alpha_p_value_comparison_reversed"),
        ("p值很小所以效应一定很大。", "statistical_significance_proves_large_effect"),
    ),
)
def test_p_value_question_rejects_core_reversals(claim: str, tag: str) -> None:
    result = _grade_choice(
        _question("hypothesis_p_value_concept_01"),
        LearnerSubmission(answer="A", reasoning=f"{_p_value_reasoning()}{claim}"),
    )

    assert result.is_correct is False
    assert tag in result.misconception_candidates


def test_p_value_negation_and_retraction_are_directional() -> None:
    question = _question("hypothesis_p_value_concept_01")
    protected = _grade_choice(
        question,
        LearnerSubmission(
            answer="A",
            reasoning=(
                _p_value_reasoning()
                + "不能说H0为真的概率是3.2%，也不能说显著就代表效应很大。"
            ),
        ),
    )
    assert protected.is_correct is True

    corrected = _grade_choice(
        question,
        LearnerSubmission(
            answer="A",
            reasoning=(
                "H0为真的概率是3.2%。前面的说法错误。" + _p_value_reasoning()
            ),
        ),
    )
    assert corrected.is_correct is True


@pytest.mark.parametrize(
    "code",
    (
        'stats.ttest_ind(variant_b, variant_a, equal_var=False, alternative="two-sided").pvalue',
        (
            "scipy.stats.ttest_ind(variant_a, variant_b, equal_var=False, "
            'alternative="two-sided").pvalue'
        ),
        (
            "left = variant_b\nright = variant_a\n"
            "answer = stats.ttest_ind(left, right, equal_var=False, "
            'alternative="two-sided").pvalue\n'
            "answer"
        ),
        (
            "result = stats.ttest_ind(variant_b, variant_a, equal_var=False, "
            'alternative="two-sided")\n'
            "result.pvalue"
        ),
    ),
)
def test_welch_python_accepts_exact_connected_equivalents(code: str) -> None:
    result = _grade_welch(
        LearnerSubmission(
            answer="0.002948258250101697",
            reasoning=_welch_reasoning(),
            python_code=code,
        )
    )

    assert result.is_correct is True
    analysis = analyze_python_code(code)
    assert PythonStructureKind.DIRECT_WELCH_TTEST_PVALUE in (
        analysis.result_expressions[0].structure_kinds
    )


@pytest.mark.parametrize(
    ("code", "tag", "mismatch"),
    (
        (
            'stats.ttest_ind(variant_b, variant_a, equal_var=True, alternative="two-sided").pvalue',
            "pooled_ttest_used_when_welch_required",
            PythonMismatchKind.POOLED_TTEST_FOR_WELCH_QUESTION,
        ),
        (
            'stats.ttest_ind(variant_b, variant_a, alternative="two-sided").pvalue',
            "pooled_ttest_used_when_welch_required",
            PythonMismatchKind.POOLED_TTEST_FOR_WELCH_QUESTION,
        ),
        (
            'stats.ttest_ind(variant_b, variant_a, equal_var=False, alternative="greater").pvalue',
            "one_sided_test_used_for_two_sided_question",
            PythonMismatchKind.ONE_SIDED_TTEST_FOR_TWO_SIDED_QUESTION,
        ),
        (
            "stats.ttest_ind(variant_a, variant_a, equal_var=False, "
            'alternative="two-sided").pvalue',
            "ttest_same_group_twice",
            PythonMismatchKind.TTEST_SAME_GROUP_TWICE,
        ),
        (
            "variant_a = variant_b\n"
            "stats.ttest_ind(variant_b, variant_a, equal_var=False, "
            'alternative="two-sided").pvalue',
            "ttest_same_group_twice",
            PythonMismatchKind.TTEST_SAME_GROUP_TWICE,
        ),
        (
            'stats.ttest_ind(variant_b, variant_a, equal_var=False, alternative="two-sided")',
            "ttest_result_not_pvalue",
            PythonMismatchKind.TTEST_RESULT_WITHOUT_PVALUE,
        ),
    ),
)
def test_welch_python_reports_specific_structural_misconceptions(
    code: str,
    tag: str,
    mismatch: PythonMismatchKind,
) -> None:
    result = _grade_welch(
        LearnerSubmission(
            answer="0.002948258250101697",
            reasoning=_welch_reasoning(),
            python_code=code,
        )
    )

    assert result.is_correct is False
    assert tag in result.misconception_candidates
    assert mismatch in analyze_python_code(code).result_expressions[0].mismatch_kinds


@pytest.mark.parametrize(
    "code",
    (
        "0.002948258250101697",
        'stats.ttest_ind(variant_b, variant_a, equal_var=False, alternative="two-sided").statistic',
        (
            "holder.stats.ttest_ind(variant_b, variant_a, equal_var=False, "
            'alternative="two-sided").pvalue'
        ),
        (
            "stats.ttest_ind(variant_b, variant_a, equal_var=False, "
            'alternative="two-sided", nan_policy="omit").pvalue'
        ),
        'stats.ttest_ind(*groups, equal_var=False, alternative="two-sided").pvalue',
        (
            "good = stats.ttest_ind(variant_b, variant_a, equal_var=False, "
            'alternative="two-sided").pvalue\n'
            "999"
        ),
        (
            "stats = holder\n"
            'stats.ttest_ind(variant_b, variant_a, equal_var=False, alternative="two-sided").pvalue'
        ),
        (
            "for item in []:\n    pass\n"
            'stats.ttest_ind(variant_b, variant_a, equal_var=False, alternative="two-sided").pvalue'
        ),
    ),
)
def test_welch_python_rejects_hardcoding_disconnection_and_bypasses(code: str) -> None:
    result = _grade_welch(
        LearnerSubmission(
            answer="0.002948258250101697",
            reasoning=_welch_reasoning(),
            python_code=code,
        )
    )

    assert result.is_correct is False
    assert "welch_ttest_structure_mismatch" in result.misconception_candidates


def test_welch_static_analysis_never_executes_learner_code(monkeypatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("learner code was executed")

    monkeypatch.setattr("builtins.eval", forbidden)
    monkeypatch.setattr("builtins.exec", forbidden)
    monkeypatch.setattr("scipy.stats.ttest_ind", forbidden)
    code = (
        'stats.ttest_ind(variant_b, variant_a, equal_var=False, '
        'alternative="two-sided").pvalue'
    )

    result = _grade_welch(
        LearnerSubmission(
            answer="0.002948258250101697",
            reasoning=_welch_reasoning(),
            python_code=code,
        )
    )

    assert result.is_correct is True


def test_unsafe_code_blocks_otherwise_correct_welch_answer() -> None:
    result = _grade_welch(
        LearnerSubmission(
            answer="0.002948258250101697",
            reasoning=_welch_reasoning(),
            python_code=(
                "runner = eval\nrunner('1+1')\n"
                'stats.ttest_ind(variant_b, variant_a, equal_var=False, '
                'alternative="two-sided").pvalue'
            ),
        )
    )

    assert result.is_correct is False
    assert "unsafe_code_execution_request" in result.misconception_candidates


def test_errors_power_question_accepts_complete_evidence() -> None:
    result = _grade_choice(
        _question("hypothesis_errors_power_concept_01"),
        LearnerSubmission(answer="A", reasoning=_errors_power_reasoning()),
    )

    assert result.is_correct is True


@pytest.mark.parametrize(
    ("claim", "tag"),
    (
        ("一类错误是漏掉真提升。", "type_i_type_ii_reversed"),
        ("功效就是1-α。", "power_definition_reversed"),
        ("样本量够大就一定显著。", "larger_sample_guarantees_significance"),
        ("大样本会自动消除系统偏差。", "sample_size_removes_design_bias"),
    ),
)
def test_errors_power_rejects_reversed_claims(claim: str, tag: str) -> None:
    result = _grade_choice(
        _question("hypothesis_errors_power_concept_01"),
        LearnerSubmission(answer="A", reasoning=f"{_errors_power_reasoning()}{claim}"),
    )

    assert result.is_correct is False
    assert tag in result.misconception_candidates


def test_effect_question_accepts_complete_evidence() -> None:
    result = _grade_choice(
        _question("ab_effect_significance_interpretation_01"),
        LearnerSubmission(answer="A", reasoning=_effect_reasoning()),
    )

    assert result.is_correct is True


@pytest.mark.parametrize(
    ("claim", "tag"),
    (
        ("显著性证明效应很大。", "statistical_significance_proves_large_effect"),
        ("相对提升也是0.15%。", "percentage_point_relative_change_confused"),
        ("区间排除0就说明达到1个百分点。", "confidence_interval_practical_threshold_ignored"),
        ("随机分组保证对所有用户都成立。", "randomization_guarantees_external_validity"),
    ),
)
def test_effect_question_rejects_significance_and_scope_reversals(
    claim: str,
    tag: str,
) -> None:
    result = _grade_choice(
        _question("ab_effect_significance_interpretation_01"),
        LearnerSubmission(answer="A", reasoning=f"{_effect_reasoning()}{claim}"),
    )

    assert result.is_correct is False
    assert tag in result.misconception_candidates


def test_design_question_accepts_complete_evidence() -> None:
    result = _grade_choice(
        _question("ab_design_multiplicity_interpretation_01"),
        LearnerSubmission(answer="A", reasoning=_design_reasoning()),
    )

    assert result.is_correct is True


@pytest.mark.parametrize(
    ("claim", "tag"),
    (
        ("有一个p小于0.05就必有真实改善。", "multiple_comparisons_ignored"),
        ("20次至少一次假阳性仍是5%。", "familywise_error_calculation_wrong"),
        ("自选分组等价于随机化。", "self_selection_treated_as_randomization"),
        ("18%与5%的流失差异可以忽略。", "differential_attrition_ignored"),
    ),
)
def test_design_question_rejects_validity_reversals(claim: str, tag: str) -> None:
    result = _grade_choice(
        _question("ab_design_multiplicity_interpretation_01"),
        LearnerSubmission(answer="A", reasoning=f"{_design_reasoning()}{claim}"),
    )

    assert result.is_correct is False
    assert tag in result.misconception_candidates


@pytest.mark.parametrize(
    ("question_id", "reasoning"),
    (
        (
            "hypothesis_p_value_concept_01",
            "H0说两个版本的总体转化没有差别，双侧H1则说差值不为零。"
            "若H0连同模型条件成立，像本次或更偏离零的统计量出现机会是0.032。"
            "它低于alpha=.05，故否定H0；这个数既不是H0的真假概率，"
            "也没告诉我们差异有多大，仍需效应量和CI。",
        ),
        (
            "hypothesis_errors_power_concept_01",
            "没有提升却宣布上线，犯的是第一类；"
            "真实提升到规定大小却没检出，是第二类。"
            "power就是在该备择点上能拒绝H0的频率。"
            "加样本通常压低随机误差而提高检出率，"
            "但不会改变提升本身，也治不好分组偏差，更不承诺这次一定显著。",
        ),
        (
            "ab_effect_significance_interpretation_01",
            "p=.001且区间0.06到0.24不跨0，故统计上显著。"
            "10.00%到10.15%差0.15个百分点，相对基线是1.5%。"
            "估计值和上界均不到事先要求的1个百分点，所以不能称有实际价值。"
            "随机分派只支撑当前实验里的因果，换人群或时间仍需验证。",
        ),
        (
            "ab_design_multiplicity_interpretation_01",
            "若20项H0都真且检验独立，全都不误报的概率是.95^20，补集约.6415。"
            "看完才挑唯一显著项属于多测与选择报告，应用预注册或校正。"
            "用户自己选版本会混杂，不是随机指派；"
            "18%对5%的退出不平衡会改变留下的人，需披露并做敏感性分析，不能直接删行。",
        ),
        (
            "hypothesis_p_value_concept_01",
            "零假设规定总体差等于零，双边备择允许正负差异。"
            "以零假设与检验前提为条件，统计量达到样本所见或更极端的尾部面积是3.2%。"
            "3.2%比事先5%的门槛低，按方案驳回零假设。"
            "它不等于零假设真伪的可信程度，显著也不量化差异幅度。",
        ),
        (
            "hypothesis_errors_power_concept_01",
            "空效果存在却被我们拒绝并发布，属于第一类误判；"
            "有预先关心的提升却没有检出，属于第二类。"
            "功效是在给定这个提升时成功驳回H0的机会。"
            "增加n一般会提升检出机会，但既不能创造真实提升，"
            "也无法修好选择偏差，更没有必然显著的承诺。",
        ),
        (
            "ab_effect_significance_interpretation_01",
            "p=.001并且0.06至0.24的区间完全位于零上方，因此统计上有差异。"
            "B比A高0.15个百分点，相对A是1.5%。"
            "无论中心还是上端都没有达到事前定下的1个百分点门槛，故不能称为重要收益。"
            "随机指派支持这次试验对象内的因果判断，换市场或时期仍要再验证。",
        ),
        (
            "ab_design_multiplicity_interpretation_01",
            "假定20个零假设均成立且彼此独立，零误报机会为0.95的二十次幂，"
            "因此至少一个误报的补概率约64.15%。"
            "结果出来后再挑最小p属于多指标筛选，应先登记主指标或调节阈值。"
            "自己选择界面不能形成随机组，会有混杂；"
            "新旧两组18%和5%的退出差也会改变可分析人群，应披露并检验敏感性。",
        ),
    ),
)
def test_judge_natural_paraphrases_are_accepted(
    question_id: str,
    reasoning: str,
) -> None:
    result = _grade_choice(
        _question(question_id),
        LearnerSubmission(answer="A", reasoning=reasoning),
    )

    assert result.is_correct is True


@pytest.mark.parametrize(
    ("question_id", "base_reasoning", "claim", "tag"),
    (
        (
            "hypothesis_p_value_concept_01",
            _p_value_reasoning(),
            "不过原假设正确的可信度只剩3.2%。",
            "p_value_is_probability_null_true",
        ),
        (
            "hypothesis_p_value_concept_01",
            _p_value_reasoning(),
            "因此新版获胜的把握达到96.8%。",
            "p_value_is_probability_treatment_better",
        ),
        (
            "hypothesis_p_value_concept_01",
            _p_value_reasoning(),
            "显著便说明收益必然可观。",
            "statistical_significance_proves_large_effect",
        ),
        (
            "hypothesis_errors_power_concept_01",
            _errors_power_reasoning(),
            "换句话说，第一类其实是漏掉有效版本。",
            "type_i_type_ii_reversed",
        ),
        (
            "hypothesis_errors_power_concept_01",
            _errors_power_reasoning(),
            "power等于1-alpha。",
            "power_definition_reversed",
        ),
        (
            "hypothesis_errors_power_concept_01",
            _errors_power_reasoning(),
            "数据量无限就必能显著。",
            "larger_sample_guarantees_significance",
        ),
        (
            "hypothesis_errors_power_concept_01",
            _errors_power_reasoning(),
            "扩大样本会冲淡非随机分组问题。",
            "sample_size_removes_design_bias",
        ),
        (
            "ab_effect_significance_interpretation_01",
            _effect_reasoning(),
            "小p本身已经证明商业影响巨大。",
            "statistical_significance_proves_large_effect",
        ),
        (
            "ab_effect_significance_interpretation_01",
            _effect_reasoning(),
            "0.15个百分点也就是0.15%的相对涨幅。",
            "percentage_point_relative_change_confused",
        ),
        (
            "ab_effect_significance_interpretation_01",
            _effect_reasoning(),
            "CI避开0便自动越过业务门槛。",
            "confidence_interval_practical_threshold_ignored",
        ),
        (
            "ab_effect_significance_interpretation_01",
            _effect_reasoning(),
            "随机分配使结论在任何地区都通用。",
            "randomization_guarantees_external_validity",
        ),
        (
            "ab_design_multiplicity_interpretation_01",
            _design_reasoning(),
            "只要找到一个低于.05，就能肯定至少一个改进是真的。",
            "multiple_comparisons_ignored",
        ),
        (
            "ab_design_multiplicity_interpretation_01",
            _design_reasoning(),
            "家族误报率仍固定在单项5%。",
            "familywise_error_calculation_wrong",
        ),
        (
            "ab_design_multiplicity_interpretation_01",
            _design_reasoning(),
            "让用户自主选择也能实现随机平衡。",
            "self_selection_treated_as_randomization",
        ),
        (
            "ab_design_multiplicity_interpretation_01",
            _design_reasoning(),
            "把退出者去掉即可恢复无偏。",
            "differential_attrition_ignored",
        ),
        (
            "hypothesis_p_value_concept_01",
            _p_value_reasoning(),
            "可是p本质上就是零假设的后验可信度。",
            "p_value_is_probability_null_true",
        ),
        (
            "hypothesis_errors_power_concept_01",
            _errors_power_reasoning(),
            "不过只要观测数无限多，显著性就是百分之百确定的。",
            "larger_sample_guarantees_significance",
        ),
        (
            "ab_effect_significance_interpretation_01",
            _effect_reasoning(),
            "不过随机化以后就拥有跨地区跨时间的普适性。",
            "randomization_guarantees_external_validity",
        ),
        (
            "ab_design_multiplicity_interpretation_01",
            _design_reasoning(),
            "实际上让参与者自愿挑版本同样能消掉混杂。",
            "self_selection_treated_as_randomization",
        ),
    ),
)
def test_judge_rephrased_reversals_are_rejected(
    question_id: str,
    base_reasoning: str,
    claim: str,
    tag: str,
) -> None:
    result = _grade_choice(
        _question(question_id),
        LearnerSubmission(answer="A", reasoning=f"{base_reasoning}{claim}"),
    )

    assert result.is_correct is False
    assert tag in result.misconception_candidates


@pytest.mark.parametrize(
    ("question_id", "reasoning", "retraction", "tag"),
    (
        (
            "hypothesis_p_value_concept_01",
            _p_value_reasoning(),
            "以上整段结论收回。",
            "p_value_context_unexplained",
        ),
        (
            "hypothesis_errors_power_concept_01",
            _errors_power_reasoning(),
            "前面的判断全部作废。",
            "errors_power_boundary_unexplained",
        ),
        (
            "ab_effect_significance_interpretation_01",
            _effect_reasoning(),
            "刚才整段都收回。",
            "effect_practical_boundary_unexplained",
        ),
        (
            "ab_design_multiplicity_interpretation_01",
            _design_reasoning(),
            "以上分析一律撤销。",
            "ab_design_validity_unexplained",
        ),
    ),
)
def test_judge_terminal_retraction_withdraws_all_prior_support(
    question_id: str,
    reasoning: str,
    retraction: str,
    tag: str,
) -> None:
    result = _grade_choice(
        _question(question_id),
        LearnerSubmission(
            answer="A",
            reasoning=f"{reasoning}{retraction}",
        ),
    )

    assert result.is_correct is False
    assert tag in result.misconception_candidates


def test_judge_welch_paraphrase_and_probability_reversal() -> None:
    code = (
        'stats.ttest_ind(variant_b, variant_a, equal_var=False, '
        'alternative="two-sided").pvalue'
    )
    natural = (
        "A的算术平均为71，B为74.875，后者多出3.875分钟。"
        "双侧Welch给出的概率值约.002948，低于5%的门槛，所以否定均值相同。"
        "每组只有8个虚构单位，这只能演示当前随机化设定，"
        "不能代表所有线上用户；还要呈现效应大小和区间。"
    )

    accepted = _grade_welch(
        LearnerSubmission(
            answer="0.002948258250101697",
            reasoning=natural,
            python_code=code,
        )
    )
    assert accepted.is_correct is True

    rejected = _grade_welch(
        LearnerSubmission(
            answer="0.002948258250101697",
            reasoning=(
                _welch_reasoning()
                + "不过均值相等的可信度就是0.2948%。"
            ),
            python_code=code,
        )
    )
    assert rejected.is_correct is False
    assert "p_value_is_probability_null_true" in (
        rejected.misconception_candidates
    )


def test_first_two_hints_do_not_leak_final_results_or_code() -> None:
    forbidden_by_question = {
        "hypothesis_p_value_concept_01": ("0.032<0.05", "选a", "答案是"),
        "ab_welch_ttest_python_01": ("0.002948", "3.875", ".pvalue", "答案是"),
        "hypothesis_errors_power_concept_01": ("一类错误是", "二类错误是", "选a"),
        "ab_effect_significance_interpretation_01": ("0.15个百分点", "1.5%", "选a"),
        "ab_design_multiplicity_interpretation_01": ("0.6415", "64.15%", "选a"),
    }
    for question_id, forbidden_phrases in forbidden_by_question.items():
        hints = _question(question_id).hints
        assert hints is not None
        early = f"{hints.concept_cue}\n{hints.method_cue}".casefold()
        assert not any(phrase.casefold() in early for phrase in forbidden_phrases)
        complete = hints.complete_explanation
        assert all(
            getattr(complete, dimension)
            for dimension in ("concept", "calculation", "python", "interpretation")
        )


def test_two_rag_sources_are_pending_low_risk_eligible_and_match_unit_nodes() -> None:
    catalog = load_default_curriculum_catalog()
    unit = next(unit for unit in catalog.units if unit.unit_id == UNIT_ID)
    manifest = load_rag_manifest(ROOT / "data" / "rag" / "manifest.yaml")
    entries = {entry.source_id: entry for entry in manifest.sources}

    assert set(unit.source_ids) == SOURCE_IDS
    covered_nodes: set[str] = set()
    for source_id in SOURCE_IDS:
        entry = entries[source_id]
        loaded = load_rag_source(entry, ROOT)
        assert entry.answer_leakage_risk.value == "low"
        assert entry.metadata["content_status"] == "pending_teacher_review"
        assert loaded.eligibility.eligible_for_chunking is True
        assert loaded.document.concept_id.value == "hypothesis_testing"
        covered_nodes.update(entry.metadata["knowledge_node_ids"])
    assert covered_nodes == set(unit.knowledge_node_ids)


def test_hypothesis_graph_edges_and_chapter_mapping_are_present_and_acyclic() -> None:
    catalog = load_default_curriculum_catalog()
    edges = {
        (edge.source_node_id, edge.target_node_id, edge.relation.value)
        for edge in catalog.edges
    }

    assert {
        ("ht_hypotheses", "ht_p_value", "prerequisite"),
        ("ht_p_value", "ht_errors_power", "prerequisite"),
        ("ht_hypotheses", "ht_ab_design", "prerequisite"),
        ("ht_p_value", "ht_ab_design", "supports"),
        ("ht_errors_power", "ht_ab_design", "supports"),
        ("jc_grouping", "ht_ab_design", "supports"),
        ("dq_decision_log", "ht_ab_design", "supports"),
    } <= edges
    chapter = next(
        mapping
        for mapping in catalog.chapter_mappings
        if mapping.textbook_id.value == "probability_statistics"
        and mapping.chapter_number == 8
    )
    assert chapter.unit_ids == (UNIT_ID,)
    assert set(chapter.knowledge_node_ids) == {
        "ht_hypotheses",
        "ht_p_value",
        "ht_errors_power",
    }


def test_all_g2_8_question_misconception_rules_are_declared() -> None:
    for question in _questions():
        declared = set(question.misconception_tags)
        assert question.evidence_policy.reasoning_insufficient_rule is not None
        assert (
            question.evidence_policy.reasoning_insufficient_rule.misconception_tag
            in declared
        )
        for rule in question.evidence_policy.text_rules:
            if rule.misconception_tag is not None:
                assert rule.misconception_tag in declared
        static_spec = question.evidence_policy.python_static_spec
        if static_spec is not None:
            assert static_spec.misconception_tag in declared
            assert all(
                rule.misconception_tag in declared
                for rule in static_spec.mismatch_rules
            )
