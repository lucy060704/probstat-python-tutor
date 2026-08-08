"""Content, AST, leakage, source, and graph checks for G2.6 Unit 6."""

from pathlib import Path

import numpy as np
import pytest

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
    Question,
    QuestionType,
)

ROOT = Path(__file__).resolve().parents[1]
UNIT_ID = DeepUnitId.SAMPLING_INFERENCE


def _questions() -> tuple[Question, ...]:
    return tuple(
        question
        for question in load_default_question_bank().questions
        if question.unit_id == UNIT_ID
    )


def _question(question_id: str) -> Question:
    return next(question for question in _questions() if question.id == question_id)


def _grade(question: Question, submission: LearnerSubmission):
    if isinstance(question.expected_answer, (int, float)):
        answer_result = grade_numeric(
            submission.answer,
            float(question.expected_answer),
            absolute_tolerance=question.numeric_tolerance or 0.0,
        )
    else:
        answer_result = grade_multiple_choice(
            submission.answer,
            str(question.expected_answer),
            accepted_answers=question.accepted_answers,
        )
    return combine_submission_evidence(question, submission, answer_result)


def _clt_reasoning() -> str:
    return (
        "反复独立随机抽样，每次只保留一个均值，这些样本均值形成均值统计量的抽样分布。"
        "中心极限定理在有限方差且样本量足够时给近似；其中心为 50，"
        "标准误为 16/8=2。原始总体仍然右偏，不保证有限样本下精确正态。"
    )


def test_unit_has_five_ready_questions_two_sources_and_full_node_coverage() -> None:
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
    } == {"sampling_standard_error_python_01"}
    assert unit.content_status == UnitContentStatus.PENDING_TEACHER_REVIEW
    assert set(unit.source_ids) == {
        "sampling_standard_error_core",
        "sampling_distribution_clt_core",
    }


def test_existing_scaling_case_remains_compatible_and_blocks_reversals() -> None:
    question = _question("sampling_standard_error_concept_01")
    correct = _grade(
        question,
        LearnerSubmission(
            answer="0.5",
            reasoning=(
                "标准误与样本量平方根成反比；样本量从 25 变为 100，"
                "平方根变为两倍，所以标准误减半。"
            ),
        ),
    )
    assert correct.is_correct is True

    cases = (
        ("样本量扩大四倍所以标准误缩小四倍。", "se_divides_by_n"),
        ("样本越多标准误越大。", "larger_sample_larger_se"),
        ("样本量变化不影响标准误。", "sample_size_does_not_affect_se"),
        ("标准误减半就证明没有偏差。", "small_se_proves_unbiased"),
    )
    for reasoning, expected_tag in cases:
        result = _grade(
            question,
            LearnerSubmission(answer="0.5", reasoning=reasoning),
        )
        assert result.is_correct is False
        assert expected_tag in result.misconception_candidates


def test_scaling_reasoning_requires_relation_and_conclusion_and_blocks_bias_claim() -> None:
    question = _question("sampling_standard_error_concept_01")
    cases = (
        ("平方根。", "standard_error_scaling_unexplained"),
        (
            "因为有平方根，所以新标准误反而更大。",
            "standard_error_scaling_unexplained",
        ),
        (
            "样本量增大到4倍，标准误按平方根关系减半；"
            "但这说明选择偏差已经完全消失。",
            "small_se_proves_unbiased",
        ),
    )
    for reasoning, expected_tag in cases:
        result = _grade(
            question,
            LearnerSubmission(answer="0.5", reasoning=reasoning),
        )
        assert result.is_correct is False
        assert expected_tag in result.misconception_candidates

    corrected = _grade(
        question,
        LearnerSubmission(
            answer="0.5",
            reasoning=(
                "选择偏差已经完全消失。上句话错误。"
                "标准误按平方根关系缩小为一半；这不代表无偏。"
            ),
        ),
    )
    assert corrected.is_correct is True

    reversed_relation = _grade(
        question,
        LearnerSubmission(
            answer="0.5",
            reasoning=(
                "标准误与样本量的平方根成正比；"
                "本题按平方根关系计算，标准误减半。"
            ),
        ),
    )
    assert reversed_relation.is_correct is False
    assert "se_scaling_direction_reversed" in (
        reversed_relation.misconception_candidates
    )

    formula_paraphrase = _grade(
        question,
        LearnerSubmission(
            answer="0.5",
            reasoning=(
                "均值标准误公式的分母是根号n，n由25变100，分母从5变10，"
                "因此新旧比值为0.5。"
            ),
        ),
    )
    assert formula_paraphrase.is_correct is True

    symbolic_formula = _grade(
        question,
        LearnerSubmission(
            answer="0.5",
            reasoning="SE=s/sqrt(n)，n从25到100，所以SE_new/SE_old=0.5。",
        ),
    )
    assert symbolic_formula.is_correct is True


def test_clt_question_requires_object_conditions_scale_and_boundary() -> None:
    question = _question("sampling_inference_clt_01")
    assert _grade(
        question,
        LearnerSubmission(answer="A", reasoning=_clt_reasoning()),
    ).is_correct is True

    incomplete = _grade(
        question,
        LearnerSubmission(answer="A", reasoning="样本量大，所以会变成正态。"),
    )
    assert incomplete.is_correct is False
    assert "sampling_distribution_clt_boundary_unexplained" in (
        incomplete.misconception_candidates
    )

    cases = (
        ("中心极限定理把原始数据变成正态。", "raw_data_becomes_normal_under_clt"),
        ("一份样本的直方图就是抽样分布。", "single_sample_is_sampling_distribution"),
        ("总体分布就是抽样分布。", "sampling_distribution_confused_with_population"),
        ("标准误是 16/64。", "se_divides_by_n"),
        ("n=64 就一定正态。", "clt_exact_normality_claim"),
        ("总体均值会随每次抽样改变。", "population_parameter_treated_as_sample_statistic"),
    )
    for reasoning, expected_tag in cases:
        result = _grade(
            question,
            LearnerSubmission(answer="A", reasoning=reasoning),
        )
        assert result.is_correct is False
        assert expected_tag in result.misconception_candidates


def test_clt_reversal_is_blocked_but_explicit_correction_can_pass() -> None:
    question = _question("sampling_inference_clt_01")
    reversal = _grade(
        question,
        LearnerSubmission(
            answer="A",
            reasoning=f"{_clt_reasoning()}不过 CLT 近似的是原始数据分布。",
        ),
    )
    assert reversal.is_correct is False
    assert "raw_data_becomes_normal_under_clt" in reversal.misconception_candidates

    corrected = _grade(
        question,
        LearnerSubmission(
            answer="A",
            reasoning=(
                "中心极限定理把原始数据变成正态。上句话错误。"
                f"{_clt_reasoning()}"
            ),
        ),
    )
    assert corrected.is_correct is True


def test_clt_natural_paraphrase_and_judge_reversals_are_deterministic() -> None:
    question = _question("sampling_inference_clt_01")
    natural = (
        "把过程独立地做两千遍，每批计算平均数，平均数组成均值经验分布；"
        "观察互不依赖、二阶矩存在且规模充分时可作渐近逼近；"
        "分布围绕50，16除以根号64等于2；底层个体仍右偏。"
    )
    assert _grade(
        question,
        LearnerSubmission(answer="A", reasoning=natural),
    ).is_correct is True

    reversals = (
        ("实际被拉成钟形的是每批64个观测本身", "raw_data_becomes_normal_under_clt"),
        ("64这个数确保最后绝对就是正态", "clt_exact_normality_claim"),
        ("标准误该写成16除以64，也就是0.25", "se_divides_by_n"),
        (
            "总体的均值会在每回重新抽取时跟着变化",
            "population_parameter_treated_as_sample_statistic",
        ),
    )
    for reversal, expected_tag in reversals:
        result = _grade(
            question,
            LearnerSubmission(answer="A", reasoning=f"{natural}不过{reversal}。"),
        )
        assert result.is_correct is False
        assert expected_tag in result.misconception_candidates

    condition_reversals = (
        "不过观察之间会互相影响，并不独立。",
        "实际上总体的方差无限大。",
        "其实64个样本在这里根本不足以支持正态近似。",
    )
    for reversal in condition_reversals:
        result = _grade(
            question,
            LearnerSubmission(answer="A", reasoning=f"{natural}{reversal}"),
        )
        assert result.is_correct is False
        assert "clt_conditions_ignored" in result.misconception_candidates

    semantic_condition_reversals = (
        "可是数据在时间上有关联，独立条件并不成立。",
        "可这个总体根本没有有限的二阶矩。",
    )
    for reversal in semantic_condition_reversals:
        result = _grade(
            question,
            LearnerSubmission(answer="A", reasoning=f"{natural}{reversal}"),
        )
        assert result.is_correct is False
        assert "clt_conditions_ignored" in result.misconception_candidates

    withdrawn_all = _grade(
        question,
        LearnerSubmission(answer="A", reasoning=f"{natural}上面整段都不对。"),
    )
    assert withdrawn_all.is_correct is False
    assert "sampling_distribution_clt_boundary_unexplained" in (
        withdrawn_all.misconception_candidates
    )


def test_clt_content_reviewer_reference_sentence_is_accepted() -> None:
    reasoning = (
        "反复按同样方法抽 64 个观测，每次只保留均值，这些均值跨样本的分布才是"
        "样本均值的抽样分布。中心极限定理在独立随机抽样、方差有限且样本量足够时"
        "近似这个均值统计量的分布；中心仍是 50，标准误是 16/8=2。"
        "它不会把原始个体观测变成正态，近似也不等于精确保证。"
    )
    result = _grade(
        _question("sampling_inference_clt_01"),
        LearnerSubmission(answer="A", reasoning=reasoning),
    )
    assert result.is_correct is True


@pytest.mark.parametrize(
    "code",
    (
        "np.std(sample, ddof=1) / np.sqrt(len(sample))",
        "np.std(sample, axis=0, ddof=1) / np.sqrt(sample.size)",
        "np.std(sample, ddof=1) / np.sqrt(sample.shape[0])",
        "np.std(sample, ddof=1) / np.sqrt(np.size(sample))",
        "np.std(sample, 0, ddof=1) / len(sample) ** 0.5",
        "sample.std(ddof=1) / sample.size ** 0.5",
        (
            "sd = np.std(sample, ddof=1)\n"
            "n = len(sample)\n"
            "root_n = np.sqrt(n)\n"
            "se = sd / root_n\n"
            "se"
        ),
    ),
)
def test_standard_error_ast_accepts_reviewed_equivalent_forms(code: str) -> None:
    result = _grade(
        _question("sampling_standard_error_python_01"),
        LearnerSubmission(answer="1.2909944487", python_code=code),
    )
    assert result.is_correct is True, code
    assert result.misconception_candidates == []


@pytest.mark.parametrize(
    ("code", "expected_tag"),
    (
        (
            "np.std(sample, ddof=0) / np.sqrt(len(sample))",
            "wrong_ddof",
        ),
        ("np.std(sample) / np.sqrt(len(sample))", "wrong_ddof"),
        (
            "np.var(sample, ddof=1) / np.sqrt(len(sample))",
            "uses_variance_in_se_formula",
        ),
        ("np.std(sample, ddof=1) / len(sample)", "omits_sqrt"),
        ("np.std(sample, ddof=1)", "returns_standard_deviation"),
        (
            "np.sqrt(len(sample)) / np.std(sample, ddof=1)",
            "python_code_conflicts_with_answer",
        ),
        ("1.2909944487358056", "python_code_conflicts_with_answer"),
        (
            "np.std(other, ddof=1) / np.sqrt(len(sample))",
            "python_code_conflicts_with_answer",
        ),
        (
            "np.std(sample, bogus, ddof=1) / np.sqrt(len(sample))",
            "python_code_conflicts_with_answer",
        ),
        (
            "np.std(sample, ddof=1) / np.sqrt(len(sample), 999)",
            "python_code_conflicts_with_answer",
        ),
        (
            "np.std(sample, ddof=1) / np.sqrt(len(sample, bogus))",
            "python_code_conflicts_with_answer",
        ),
    ),
)
def test_standard_error_ast_rejects_wrong_or_malformed_formulas(
    code: str,
    expected_tag: str,
) -> None:
    result = _grade(
        _question("sampling_standard_error_python_01"),
        LearnerSubmission(answer="1.2909944487", python_code=code),
    )
    assert result.is_correct is False, code
    assert expected_tag in result.misconception_candidates


def test_standard_error_code_must_be_finally_connected_and_safe() -> None:
    question = _question("sampling_standard_error_python_01")
    disconnected = _grade(
        question,
        LearnerSubmission(
            answer="1.2909944487",
            python_code=(
                "good = np.std(sample, ddof=1) / np.sqrt(len(sample))\n"
                "result = 1.2909944487"
            ),
        ),
    )
    assert disconnected.is_correct is False
    assert disconnected.misconception_candidates == ["python_code_conflicts_with_answer"]

    unsafe = _grade(
        question,
        LearnerSubmission(
            answer="1.2909944487",
            python_code=(
                'runner = eval\nrunner("1+1")\n'
                "np.std(sample, ddof=1) / np.sqrt(len(sample))"
            ),
        ),
    )
    assert unsafe.is_correct is False
    assert unsafe.misconception_candidates == ["unsafe_code_execution_request"]


@pytest.mark.parametrize(
    "code",
    (
        (
            'if True:\n    runner = eval\n    runner("1+1")\n'
            "np.std(sample,ddof=1)/np.sqrt(len(sample))"
        ),
        (
            'def helper():\n    runner = eval\n    return runner("1+1")\n'
            "np.std(sample,ddof=1)/np.sqrt(len(sample))"
        ),
        (
            "from builtins import eval as runner\nrunner('1+1')\n"
            "np.std(sample,ddof=1)/np.sqrt(len(sample))"
        ),
    ),
)
def test_nested_or_imported_dangerous_aliases_are_rejected(code: str) -> None:
    analysis = analyze_python_code(code)
    assert analysis.unsafe_features

    result = _grade(
        _question("sampling_standard_error_python_01"),
        LearnerSubmission(answer="1.2909944487", python_code=code),
    )
    assert result.is_correct is False
    assert result.misconception_candidates == ["unsafe_code_execution_request"]


def test_static_checker_never_calls_numpy_or_executes_submission(monkeypatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("学习者代码不应被执行")

    monkeypatch.setattr(np, "std", forbidden)
    monkeypatch.setattr(np, "sqrt", forbidden)
    monkeypatch.setattr("builtins.eval", forbidden)
    monkeypatch.setattr("builtins.exec", forbidden)

    result = _grade(
        _question("sampling_standard_error_python_01"),
        LearnerSubmission(
            answer="1.2909944487",
            python_code="np.std(sample, ddof=1) / np.sqrt(len(sample))",
        ),
    )
    assert result.is_correct is True


def test_precision_interpretation_preserves_old_answer_and_blocks_overclaim() -> None:
    question = _question("sampling_standard_error_interpretation_01")
    correct = _grade(
        question,
        LearnerSubmission(
            answer="研究B",
            reasoning=(
                "两项研究标准差相同，A 的标准误为 10/5=2，B 为 10/10=1；"
                "B 样本量更大所以均值标准误更小、更精确，但这不代表无偏。"
            ),
        ),
    )
    assert correct.is_correct is True

    cases = (
        ("标准差相同所以标准误相同。", "sample_size_does_not_matter"),
        ("标准误就是原始数据标准差。", "confuses_sd_se"),
        ("B 标准误小所以一定无偏。", "small_se_proves_unbiased"),
        (
            "样本更多，所以 B 一定更精确。",
            "precision_claim_ignores_equal_sd_condition",
        ),
    )
    for reasoning, expected_tag in cases:
        result = _grade(
            question,
            LearnerSubmission(answer="研究B", reasoning=reasoning),
        )
        assert result.is_correct is False
        assert expected_tag in result.misconception_candidates


def test_precision_reasoning_requires_all_comparison_evidence() -> None:
    question = _question("sampling_standard_error_interpretation_01")
    cases = (
        ("标准差相同。", "precision_comparison_unexplained"),
        (
            "标准差相同，但标准误描述每个人的起伏，所以B更精确。",
            "confuses_sd_se",
        ),
        (
            "标准差相同，B样本量更大，标准误更小；"
            "不过B的每个学生成绩也一定更集中。",
            "confuses_sd_se",
        ),
    )
    for reasoning, expected_tag in cases:
        result = _grade(
            question,
            LearnerSubmission(answer="研究B", reasoning=reasoning),
        )
        assert result.is_correct is False
        assert expected_tag in result.misconception_candidates

    natural = _grade(
        question,
        LearnerSubmission(
            answer="研究B",
            reasoning="给定 s 相等，B 有更多观测，所以 SE 更低、均值抽样波动更小。",
        ),
    )
    assert natural.is_correct is True

    reversals = (
        ("但样本越多均值的抽样波动反而越大。", "larger_sample_larger_se"),
        ("不过实际A的样本均值才更稳定。", "study_a_claimed_more_precise"),
    )
    base = "标准差相同，B样本量更大，所以标准误更小。"
    for reversal, expected_tag in reversals:
        result = _grade(
            question,
            LearnerSubmission(answer="研究B", reasoning=f"{base}{reversal}"),
        )
        assert result.is_correct is False
        assert expected_tag in result.misconception_candidates

    formula_paraphrase = _grade(
        question,
        LearnerSubmission(
            answer="研究B",
            reasoning=(
                "两边s一样，B有100人而A只有25人，套公式得A的SE为2、B为1，"
                "所以B均值估计抖动更少。"
            ),
        ),
    )
    assert formula_paraphrase.is_correct is True

    symbolic_formula = _grade(
        question,
        LearnerSubmission(
            answer="研究B",
            reasoning="两组s都是10，B的n是A的四倍，因此SE_B=SE_A/2。",
        ),
    )
    assert symbolic_formula.is_correct is True


def test_bias_question_separates_precision_from_systematic_error() -> None:
    question = _question("sampling_inference_bias_interpretation_01")
    correct_reasoning = (
        "A 的标准误为 1.2，B 的标准误为 0.6；标准误衡量样本均值的随机抽样波动。"
        "每次都增加 5 是固定偏移和系统偏差，样本量增加也不会消除；"
        "较小标准误不能证明 B 无偏，也不一定更接近总体真值。"
    )
    assert _grade(
        question,
        LearnerSubmission(answer="A", reasoning=correct_reasoning),
    ).is_correct is True

    cases = (
        ("样本量四倍所以标准误是四分之一。", "fourfold_n_quarters_se"),
        ("标准差相同所以标准误相同。", "equal_sd_means_equal_se"),
        (
            "标准误描述个体读数的离散，B 的个体读数更加集中。",
            "standard_error_treated_as_individual_spread",
        ),
        (
            "A 的标准误为 1.2，B 的标准误为 0.6；样本量大就会消除系统偏差。",
            "large_sample_removes_systematic_bias",
        ),
        (
            "A 的标准误为 1.2，B 的标准误为 0.6；B 标准误小所以一定无偏。",
            "small_se_proves_unbiased",
        ),
    )
    for reasoning, expected_tag in cases:
        result = _grade(
            question,
            LearnerSubmission(answer="A", reasoning=reasoning),
        )
        assert result.is_correct is False
        assert expected_tag in result.misconception_candidates


def test_bias_content_reviewer_reference_sentence_is_accepted() -> None:
    reasoning = (
        "在相同样本标准差下，A 的标准误是 12/10=1.2，B 是 12/20=0.6，"
        "所以 B 的均值随机抽样波动更小。但是仪器每次都加 5 会让均值系统性偏高，"
        "这不是增加样本量能平均掉的随机误差；因此不能由较小标准误断言 B 无偏"
        "或一定更准确。"
    )
    result = _grade(
        _question("sampling_inference_bias_interpretation_01"),
        LearnerSubmission(answer="A", reasoning=reasoning),
    )
    assert result.is_correct is True


def test_bias_reversals_and_retractions_are_deterministic() -> None:
    question = _question("sampling_inference_bias_interpretation_01")
    correct = (
        "A 的标准误为 1.2，B 的标准误为 0.6；标准误衡量样本均值的随机抽样波动。"
        "每次都增加 5 是固定偏移和系统偏差，样本量增加也不会消除；"
        "较小标准误不能证明 B 无偏，也不一定更接近总体真值。"
    )
    reversals = (
        (
            "不过这个加5其实只是随机误差，多收些样本就会趋近于0。",
            "large_sample_removes_systematic_bias",
        ),
        ("但SE小足以说明B的估计更真实。", "small_se_proves_unbiased"),
        (
            "而且标准误其实就是每个人测量值的起伏。",
            "standard_error_treated_as_individual_spread",
        ),
        ("不过B的标准误应当是0.3。", "fourfold_n_quarters_se"),
        (
            "实际上系统性偏移会被增加的样本稀释掉。",
            "large_sample_removes_systematic_bias",
        ),
    )
    for reversal, expected_tag in reversals:
        result = _grade(
            question,
            LearnerSubmission(answer="A", reasoning=f"{correct}{reversal}"),
        )
        assert result.is_correct is False
        assert expected_tag in result.misconception_candidates

    corrected = _grade(
        question,
        LearnerSubmission(
            answer="A",
            reasoning=f"B的标准误应当是0.3。上句话错误。{correct}",
        ),
    )
    assert corrected.is_correct is True

    withdrawn = _grade(
        question,
        LearnerSubmission(answer="A", reasoning=f"{correct}上面整段都不对。"),
    )
    assert withdrawn.is_correct is False
    assert "standard_error_bias_boundary_unexplained" in (
        withdrawn.misconception_candidates
    )

    semantic_reversal = _grade(
        question,
        LearnerSubmission(
            answer="A",
            reasoning=(
                f"{correct}话虽如此，加5这种影响属于偶然噪声，"
                "随着n增大会自行归零。"
            ),
        ),
    )
    assert semantic_reversal.is_correct is False
    assert "large_sample_removes_systematic_bias" in (
        semantic_reversal.misconception_candidates
    )


def test_bias_semantic_slots_accept_formula_and_plain_language() -> None:
    reasoning = (
        "A为12除以根号100等于1.2，B为12除以根号400等于0.6；"
        "标准误是平均数反复取样起伏。仪器造成统一的系统性偏移，"
        "更多样本抵消不了；不能据此说无偏或离真值更近。"
    )
    result = _grade(
        _question("sampling_inference_bias_interpretation_01"),
        LearnerSubmission(answer="A", reasoning=reasoning),
    )
    assert result.is_correct is True

    judge_wording = (
        "A为12除以根号100等于1.2，B为12除以根号400等于0.6。"
        "标准误讲的是平均数在反复取样时的起伏。仪器所有读数加5是统一的系统性偏移，"
        "再收集更多样本也抵消不了。因此B虽更精密，却不能据此说无偏或离真值更近。"
    )
    judge_result = _grade(
        _question("sampling_inference_bias_interpretation_01"),
        LearnerSubmission(answer="A", reasoning=judge_wording),
    )
    assert judge_result.is_correct is True

    second_paraphrase = (
        "研究A均值标准误1.2，研究B均值标准误0.6。标准误表示平均值在重复抽样中的随机变动；"
        "仪器是方向固定测量偏移，全部读数加5，增加人数也冲不掉，"
        "所以随机精度不意味无偏，不能说离真值更近。"
    )
    second_result = _grade(
        _question("sampling_inference_bias_interpretation_01"),
        LearnerSubmission(answer="A", reasoning=second_paraphrase),
    )
    assert second_result.is_correct is True


def test_bias_semantic_slots_reject_sentence_level_negation() -> None:
    reasoning = (
        "A的标准误为1.2并不正确，B的标准误为0.6也不对；"
        "样本均值的随机抽样波动不是标准误描述的对象；"
        "每次都增加5根本不是系统偏差；"
        "样本量增加也不会消除这个说法不成立；"
        "不能证明B无偏也是错误结论。"
    )
    result = _grade(
        _question("sampling_inference_bias_interpretation_01"),
        LearnerSubmission(answer="A", reasoning=reasoning),
    )
    assert result.is_correct is False
    assert "standard_error_bias_boundary_unexplained" in (
        result.misconception_candidates
    )


def test_bias_standard_natural_answer_is_accepted() -> None:
    reasoning = (
        "A均值标准误为1.2，B均值标准误为0.6。"
        "标准误是均值在重复抽样中的随机波动。"
        "加5不是随机误差，而是固定系统偏差；增加样本量不能消除它，"
        "因此小SE不等于无偏。"
    )
    result = _grade(
        _question("sampling_inference_bias_interpretation_01"),
        LearnerSubmission(answer="A", reasoning=reasoning),
    )
    assert result.is_correct is True


def test_first_two_hints_do_not_reveal_sampling_answers() -> None:
    forbidden_by_question = {
        "sampling_standard_error_concept_01": ("0.5", "一半", "5/10"),
        "sampling_inference_clt_01": ("中心为 50", "标准误为 2", "16/8", "选择 A"),
        "sampling_standard_error_python_01": (
            "1.29099",
            "2.58199",
            "np.std(sample,ddof=1)/np.sqrt(len(sample))",
        ),
        "sampling_standard_error_interpretation_01": ("10/5", "10/10", "研究 B"),
        "sampling_inference_bias_interpretation_01": (
            "1.2",
            "0.6",
            "12/√100",
            "12/√400",
            "选择 A",
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


def test_sampling_sources_are_registered_eligible_and_question_safe() -> None:
    manifest = load_rag_manifest(ROOT / "data" / "rag" / "manifest.yaml")
    entries = {
        entry.source_id: entry
        for entry in manifest.sources
        if entry.metadata.get("unit_id") == UNIT_ID.value
    }
    assert set(entries) == {
        "sampling_standard_error_core",
        "sampling_distribution_clt_core",
    }
    forbidden_fingerprints = (
        "μ=50",
        "σ=16",
        "n=64",
        "2000次",
        "[8,10,12,14]",
        "1.2909944487358056",
        "10/√25",
        "10/√100",
        "12/√100",
        "12/√400",
        "每次读数都会系统性增加5",
    )
    for entry in entries.values():
        loaded = load_rag_source(entry, ROOT)
        compact = (ROOT / entry.file_path).read_text(encoding="utf-8").replace(" ", "")
        assert loaded.eligibility.eligible_for_chunking is True
        assert entry.metadata["content_status"] == "pending_teacher_review"
        assert entry.metadata["reviewer_role"] == "course_teacher"
        assert not any(
            fingerprint.replace(" ", "") in compact
            for fingerprint in forbidden_fingerprints
        )


def test_sampling_graph_edges_exist_and_catalog_remains_acyclic() -> None:
    catalog = load_default_curriculum_catalog()
    edges = {
        (edge.source_node_id, edge.target_node_id, edge.relation.value)
        for edge in catalog.edges
    }
    assert {
        ("si_population_sample", "si_sampling_distribution", "prerequisite"),
        ("si_sampling_distribution", "si_clt", "prerequisite"),
        ("si_sampling_distribution", "si_standard_error", "prerequisite"),
        ("si_clt", "si_standard_error", "supports"),
        ("ps_independence", "si_sampling_distribution", "prerequisite"),
        ("ds_outlier_robustness", "si_clt", "supports"),
        ("ds_spread", "si_standard_error", "prerequisite"),
    } <= edges
