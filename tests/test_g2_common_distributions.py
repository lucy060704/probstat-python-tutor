"""Content, grading, AST, leakage, and source checks for G2.4 Unit 4."""

from pathlib import Path

from probstat_tutor.curriculum import load_default_question_bank
from probstat_tutor.curriculum_graph import (
    UnitContentStatus,
    load_default_curriculum_catalog,
)
from probstat_tutor.graders import (
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
UNIT_ID = DeepUnitId.COMMON_DISTRIBUTIONS


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
        ),
    )


def _grade_number(question: Question, submission: LearnerSubmission):
    return combine_submission_evidence(
        question,
        submission,
        grade_numeric(
            submission.answer,
            question.expected_answer,
            absolute_tolerance=question.numeric_tolerance or 0.0,
        ),
    )


def test_unit_has_three_questions_two_sources_and_pending_status() -> None:
    questions = _questions()
    catalog = load_default_curriculum_catalog()
    unit = next(unit for unit in catalog.units if unit.unit_id == UNIT_ID)

    assert len(questions) == 3
    assert {question.question_type for question in questions} == set(QuestionType)
    assert all(question.hints is not None for question in questions)
    assert all(question.evidence_policy.reasoning_support_groups for question in questions)
    assert all(
        question.review_status == ContentReviewStatus.PENDING_TEACHER_REVIEW
        for question in questions
    )
    assert {
        question.id for question in questions if question.python_code_required
    } == {"common_distributions_python_01"}
    assert unit.content_status == UnitContentStatus.PENDING_TEACHER_REVIEW
    assert set(unit.source_ids) == {
        "discrete_distributions_core",
        "continuous_distributions_core",
    }


def test_binomial_model_requires_conditions_support_and_expectation_meaning() -> None:
    question = _question("common_distributions_concept_01")
    correct = _grade_choice(
        question,
        LearnerSubmission(
            answer="A",
            reasoning=(
                "这是固定 6 次独立同概率二分类试验，符合二项分布条件；"
                "支持集是 0 到 6 的整数；E(X)=6×0.8=4.8，期望是长期平均位置。"
            ),
        ),
    )
    assert correct.is_correct is True

    natural_correct = _grade_choice(
        question,
        LearnerSubmission(
            answer="A",
            reasoning=(
                "共有6次试验，彼此独立，每次只有通过或不通过两种结果，而且通过率都为0.8；"
                "X可能取0、1、2、3、4、5、6；均值为6乘0.8等于4.8，"
                "表示大量重复检测时通过台数的平均水平，单次仍然只能是整数。"
            ),
        ),
    )
    assert natural_correct.is_correct is True

    cases = (
        ("总数 X 也是伯努利变量。", "total_count_treated_as_bernoulli"),
        ("X 可以取任意实数。", "discrete_count_given_continuous_support"),
        ("二项分布不需要独立。", "binomial_independence_ignored"),
        ("期望 4.8 不是整数所以模型错误。", "expectation_must_be_observable_integer"),
        ("期望通过数量是 0.8 个。", "probability_parameter_confused_with_expected_count"),
        ("符合二项分布条件。", "binomial_model_boundary_unexplained"),
        (
            "不知道是否符合二项分布条件；支持集是 0 到 6 的整数；"
            "E(X)=6×0.8=4.8，期望是长期平均位置。",
            "binomial_model_boundary_unexplained",
        ),
    )
    for reasoning, expected_tag in cases:
        result = _grade_choice(
            question,
            LearnerSubmission(answer="A", reasoning=reasoning),
        )
        assert result.is_correct is False
        assert expected_tag in result.misconception_candidates


def test_binomial_pmf_accepts_exact_connected_scipy_forms() -> None:
    question = _question("common_distributions_python_01")
    codes = (
        "stats.binom.pmf(2, 4, 0.5)",
        "stats.binom.pmf(k=2, n=4, p=0.5)",
        "stats.binom.pmf(2, n=4, p=0.5)",
        "stats.binom.pmf(2, 4, p=0.5)",
        "scipy.stats.binom.pmf(2, 4, 0.5)",
        "value = stats.binom.pmf(2, 4, 0.5)\nvalue",
        "k = 2\nn = 4\np = 0.5\nstats.binom.pmf(k, n, p)",
    )
    for code in codes:
        result = _grade_number(
            question,
            LearnerSubmission(
                answer="37.5%",
                reasoning=(
                    "PMF 是离散单点概率；"
                    "k 是成功次数 n 是试验次数 p 是成功概率。"
                ),
                python_code=code,
            ),
        )
        assert result.is_correct is True
        assert result.misconception_candidates == []

    natural_correct = _grade_number(
        question,
        LearnerSubmission(
            answer="0.375",
            reasoning=(
                "pmf 给出恰好取得2次成功的概率；第一个参数2表示目标成功数，"
                "第二个参数4表示总试验数，0.5表示单次成功率。"
            ),
            python_code="stats.binom.pmf(2, 4, 0.5)",
        ),
    )
    assert natural_correct.is_correct is True


def test_binomial_pmf_rejects_wrong_api_parameters_and_disconnected_results() -> None:
    question = _question("common_distributions_python_01")
    cases = (
        ("stats.binom.cdf(2, 4, 0.5)", "cdf_used_for_exact_probability"),
        ("stats.binom.pmf(1, 4, 0.5)", "binomial_pmf_code_conflict"),
        ("stats.binom.pmf(2, 5, 0.5)", "binomial_pmf_code_conflict"),
        ("stats.binom.pmf(2, 4, 0.25)", "binomial_pmf_code_conflict"),
        ("stats.binom.pmf(4, 2, 0.5)", "binomial_pmf_code_conflict"),
        ("stats.norm.pdf(2, loc=4, scale=0.5)", "binomial_pmf_code_conflict"),
        ("stats.binom.rvs(4, 0.5)", "binomial_pmf_code_conflict"),
        ("0.375", "binomial_pmf_code_conflict"),
        ("Stats.binom.pmf(2, 4, 0.5)", "binomial_pmf_code_conflict"),
        ("binom.pmf(2, 4, 0.5)", "binomial_pmf_code_conflict"),
        ("stats.binom.pmf(2, 4, 0.5, loc=0)", "binomial_pmf_code_conflict"),
        (
            "k = 2\nn = 4\np = 0.5\nk = 1\nstats.binom.pmf(k, n, p)",
            "binomial_pmf_code_conflict",
        ),
        (
            "correct = stats.binom.pmf(2, 4, 0.5)\n"
            "stats.binom.cdf(2, 4, 0.5)",
            "cdf_used_for_exact_probability",
        ),
    )
    for code, expected_tag in cases:
        result = _grade_number(
            question,
            LearnerSubmission(
                answer="0.375",
                reasoning=(
                    "PMF 是离散单点概率；"
                    "k 是成功次数 n 是试验次数 p 是成功概率。"
                ),
                python_code=code,
            ),
        )
        assert result.is_correct is False
        assert result.misconception_candidates == [expected_tag]


def test_binomial_pmf_reasoning_rejects_api_and_parameter_reversals() -> None:
    question = _question("common_distributions_python_01")
    code = "stats.binom.pmf(2, 4, 0.5)"
    cases = (
        ("cdf 就是恰好等于 2。", "cdf_used_for_exact_probability"),
        ("随机抽一次就得到理论概率。", "random_draw_used_as_theoretical_probability"),
        ("pmf 返回成功次数。", "pmf_output_treated_as_count"),
        ("k 和 n 的位置可以交换。", "binomial_parameter_roles_reversed"),
        ("离散变量应该用 pdf。", "pmf_cdf_confused"),
        ("PMF 是离散单点概率。", "binomial_pmf_roles_unexplained"),
        (
            "PMF 是离散单点概率；6/16=0.375。",
            "binomial_pmf_roles_unexplained",
        ),
    )
    for reasoning, expected_tag in cases:
        result = _grade_number(
            question,
            LearnerSubmission(answer="0.375", reasoning=reasoning, python_code=code),
        )
        assert result.is_correct is False
        assert expected_tag in result.misconception_candidates


def test_continuous_interpretation_requires_interval_density_and_support() -> None:
    question = _question("common_distributions_interpretation_01")
    correct = _grade_choice(
        question,
        LearnerSubmission(
            answer="B",
            reasoning=(
                "0.841345-0.158655=0.682690；PDF 是密度且不是点概率；"
                "理想连续模型下 P(X=30)=0；正态模型的数学支持集为全部实数。"
            ),
        ),
    )
    assert correct.is_correct is True

    natural_correct = _grade_choice(
        question,
        LearnerSubmission(
            answer="B",
            reasoning=(
                "用上端的累计概率减去下端的累计概率，结果约为0.68269；"
                "密度函数的高度不是某个精确时刻的概率，"
                "连续模型在单个点上的概率质量为零；普通正态分布可以取任意实数。"
            ),
        ),
    )
    assert natural_correct.is_correct is True

    cases = (
        ("pdf(30) 是点概率。", "density_treated_as_point_probability"),
        ("恰好 30 分钟的概率是 0.079788。", "continuous_point_given_positive_mass"),
        ("区间概率用两个 PDF 值相减。", "pdf_difference_used_for_interval"),
        ("区间概率就是 0.841345。", "upper_cdf_used_without_lower_bound"),
        ("支持集只有 [25,35]。", "normal_support_truncated_to_observed_range"),
        ("使用正态 API 就证明数据服从正态分布。", "model_choice_treated_as_proof"),
        ("区间概率用两个端点的 CDF 作差。", "continuous_distribution_boundary_unexplained"),
        (
            "不知道区间概率是否用两个端点的 CDF 作差；"
            "PDF 是密度且不是点概率；正态模型的数学支持集为全部实数。",
            "continuous_distribution_boundary_unexplained",
        ),
    )
    for reasoning, expected_tag in cases:
        result = _grade_choice(
            question,
            LearnerSubmission(answer="B", reasoning=reasoning),
        )
        assert result.is_correct is False
        assert expected_tag in result.misconception_candidates


def test_support_retractions_and_internal_reversals_cannot_pass() -> None:
    concept = _grade_choice(
        _question("common_distributions_concept_01"),
        LearnerSubmission(
            answer="A",
            reasoning=(
                "固定 6 次独立同概率二分类试验，但这些设备其实并不独立；"
                "支持集是 0 到 6 的整数；E(X)=6×0.8=4.8；期望是长期平均位置。"
            ),
        ),
    )
    assert concept.is_correct is False
    assert "binomial_independence_ignored" in concept.misconception_candidates

    python = _grade_number(
        _question("common_distributions_python_01"),
        LearnerSubmission(
            answer="0.375",
            reasoning=(
                "PMF 是离散单点概率；k 是成功次数，n 是试验次数，p 是成功概率；"
                "不过实际上 k 是总次数，n 是成功次数。"
            ),
            python_code="stats.binom.pmf(2, 4, 0.5)",
        ),
    )
    assert python.is_correct is False
    assert "binomial_parameter_roles_reversed" in python.misconception_candidates

    interpretation = _grade_choice(
        _question("common_distributions_interpretation_01"),
        LearnerSubmission(
            answer="B",
            reasoning=(
                "区间概率用两个端点的 CDF 作差；PDF 是密度且不是点概率；"
                "理想连续模型下 P(X=30)=0；正态模型的数学支持集为全部实数；"
                "不过实际上它只允许非负数。"
            ),
        ),
    )
    assert interpretation.is_correct is False
    assert "normal_support_truncated_to_observed_range" in (
        interpretation.misconception_candidates
    )

    equivalent_reversals = (
        (
            "common_distributions_concept_01",
            "A",
            (
                "固定 6 次独立同概率二分类试验；支持集是 0 到 6 的整数；"
                "E(X)=6×0.8=4.8；期望是长期平均位置；"
                "不过这些检测结果其实彼此相关，并不相互独立。"
            ),
            "",
            "binomial_independence_ignored",
        ),
        (
            "common_distributions_python_01",
            "0.375",
            (
                "PMF 是离散单点概率；k 是成功次数，n 是试验次数，p 是成功概率；"
                "不过 k 其实代表试验总数，n 才代表目标成功数。"
            ),
            "stats.binom.pmf(2, 4, 0.5)",
            "binomial_parameter_roles_reversed",
        ),
        (
            "common_distributions_interpretation_01",
            "B",
            (
                "区间概率用两个端点的 CDF 作差；PDF 是密度且不是点概率；"
                "理想连续模型下 P(X=30)=0；正态模型的数学支持集为全部实数；"
                "不过正态变量的取值不能小于 0。"
            ),
            "",
            "normal_support_truncated_to_observed_range",
        ),
    )
    for question_id, answer, reasoning, python_code, expected_tag in equivalent_reversals:
        question = _question(question_id)
        submission = LearnerSubmission(
            answer=answer,
            reasoning=reasoning,
            python_code=python_code,
        )
        result = (
            _grade_number(question, submission)
            if question.question_type == QuestionType.PYTHON
            else _grade_choice(question, submission)
        )
        assert result.is_correct is False
        assert expected_tag in result.misconception_candidates

    adjacent_natural_reversals = (
        (
            "common_distributions_concept_01",
            "A",
            (
                "固定 6 次独立同概率二分类试验；支持集是 0 到 6 的整数；"
                "E(X)=6×0.8=4.8；期望是长期平均位置；"
                "不过各次检测彼此有依赖关系。"
            ),
            "",
            "binomial_independence_ignored",
        ),
        (
            "common_distributions_python_01",
            "0.375",
            (
                "PMF 是离散单点概率；k 是成功次数，n 是试验次数，p 是成功概率；"
                "不过第一个参数是总试验数，第二个参数才是成功数。"
            ),
            "stats.binom.pmf(2, 4, 0.5)",
            "binomial_parameter_roles_reversed",
        ),
        (
            "common_distributions_interpretation_01",
            "B",
            (
                "区间概率用两个端点的 CDF 作差；PDF 是密度且不是点概率；"
                "理想连续模型下 P(X=30)=0；正态模型的数学支持集为全部实数；"
                "但我认为正态分布只能产生大于等于零的值。"
            ),
            "",
            "normal_support_truncated_to_observed_range",
        ),
    )
    for question_id, answer, reasoning, python_code, expected_tag in (
        adjacent_natural_reversals
    ):
        question = _question(question_id)
        submission = LearnerSubmission(
            answer=answer,
            reasoning=reasoning,
            python_code=python_code,
        )
        result = (
            _grade_number(question, submission)
            if question.question_type == QuestionType.PYTHON
            else _grade_choice(question, submission)
        )
        assert result.is_correct is False
        assert expected_tag in result.misconception_candidates

    common_word_form_reversals = (
        (
            "common_distributions_concept_01",
            "A",
            (
                "固定 6 次独立同概率二分类试验；支持集是 0 到 6 的整数；"
                "E(X)=6×0.8=4.8；期望是长期平均位置；"
                "然而每次试验互相相关，不满足独立性。"
            ),
            "",
            "binomial_independence_ignored",
        ),
        (
            "common_distributions_python_01",
            "0.375",
            (
                "PMF 是离散单点概率；k 是成功次数，n 是试验次数，p 是成功概率；"
                "但 k 其实表示总试验次数，n 才表示成功次数。"
            ),
            "stats.binom.pmf(2, 4, 0.5)",
            "binomial_parameter_roles_reversed",
        ),
        (
            "common_distributions_interpretation_01",
            "B",
            (
                "区间概率用两个端点的 CDF 作差；PDF 是密度且不是点概率；"
                "理想连续模型下 P(X=30)=0；正态模型的数学支持集为全部实数；"
                "不过正态分布不可能取负值。"
            ),
            "",
            "normal_support_truncated_to_observed_range",
        ),
    )
    for question_id, answer, reasoning, python_code, expected_tag in (
        common_word_form_reversals
    ):
        question = _question(question_id)
        submission = LearnerSubmission(
            answer=answer,
            reasoning=reasoning,
            python_code=python_code,
        )
        result = (
            _grade_number(question, submission)
            if question.question_type == QuestionType.PYTHON
            else _grade_choice(question, submission)
        )
        assert result.is_correct is False
        assert expected_tag in result.misconception_candidates


def test_retracted_or_incomplete_reasoning_is_not_accepted() -> None:
    cases = (
        (
            "common_distributions_concept_01",
            "A",
            (
                "固定 6 次独立同概率二分类试验；支持集是 0 到 6 的整数；"
                "E(X)=6×0.8=4.8；期望是长期平均位置。上句话错误。"
            ),
            "",
        ),
        (
            "common_distributions_python_01",
            "0.375",
            (
                "PMF 是离散单点概率；k 是成功次数，n 是试验次数，"
                "p 是成功概率。上句话错误。"
            ),
            "stats.binom.pmf(2, 4, 0.5)",
        ),
        (
            "common_distributions_interpretation_01",
            "B",
            (
                "区间概率用两个端点的 CDF 作差；PDF 是密度且不是点概率；"
                "理想连续模型下 P(X=30)=0；"
                "正态模型的数学支持集为全部实数。上句话错误。"
            ),
            "",
        ),
    )
    for question_id, answer, reasoning, python_code in cases:
        question = _question(question_id)
        submission = LearnerSubmission(
            answer=answer,
            reasoning=reasoning,
            python_code=python_code,
        )
        result = (
            _grade_number(question, submission)
            if question.question_type == QuestionType.PYTHON
            else _grade_choice(question, submission)
        )
        assert result.is_correct is False

    corrected_after_retraction = _grade_choice(
        _question("common_distributions_concept_01"),
        LearnerSubmission(
            answer="A",
            reasoning=(
                "上句话错误。固定 6 次独立同概率二分类试验；"
                "支持集是 0 到 6 的整数；E(X)=6×0.8=4.8；"
                "期望是长期平均位置。"
            ),
        ),
    )
    assert corrected_after_retraction.is_correct is True

    missing_density_and_support = _grade_choice(
        _question("common_distributions_interpretation_01"),
        LearnerSubmission(
            answer="B",
            reasoning=(
                "区间概率用两个端点的 CDF 作差；理想连续模型下 P(X=30)=0；"
                "选择正态模型不等于证明数据服从正态分布。"
            ),
        ),
    )
    assert missing_density_and_support.is_correct is False
    assert "continuous_distribution_boundary_unexplained" in (
        missing_density_and_support.misconception_candidates
    )


def test_reverse_rules_preserve_explicit_corrections() -> None:
    concept = _grade_choice(
        _question("common_distributions_concept_01"),
        LearnerSubmission(
            answer="A",
            reasoning=(
                "固定 6 次独立同概率二分类试验；支持集是 0 到 6 的整数；"
                "E(X)=6×0.8=4.8；期望是长期平均位置；"
                "题目采用独立模型，但不能据此断言现实试验互相依赖。"
            ),
        ),
    )
    assert concept.is_correct is True

    python = _grade_number(
        _question("common_distributions_python_01"),
        LearnerSubmission(
            answer="0.375",
            reasoning=(
                "PMF 是离散单点概率；k 是成功次数，n 是试验次数，p 是成功概率；"
                "k 并不代表试验总次数。"
            ),
            python_code="stats.binom.pmf(2, 4, 0.5)",
        ),
    )
    assert python.is_correct is True

    interpretation = _grade_choice(
        _question("common_distributions_interpretation_01"),
        LearnerSubmission(
            answer="B",
            reasoning=(
                "区间概率用两个端点的 CDF 作差；PDF 是密度且不是点概率；"
                "理想连续模型下 P(X=30)=0；正态模型的数学支持集为全部实数；"
                "不能说正态分布不可能取负值。"
            ),
        ),
    )
    assert interpretation.is_correct is True


def test_unsafe_callable_alias_is_rejected_but_safe_rebinding_is_allowed() -> None:
    question = _question("common_distributions_python_01")
    reasoning = (
        "PMF 是离散单点概率；k 是成功次数，n 是试验次数，p 是成功概率。"
    )
    unsafe = _grade_number(
        question,
        LearnerSubmission(
            answer="0.375",
            reasoning=reasoning,
            python_code=(
                'runner = eval\nrunner("1+1")\nstats.binom.pmf(2, 4, 0.5)'
            ),
        ),
    )
    assert unsafe.is_correct is False
    assert unsafe.misconception_candidates == ["unsafe_code_execution_request"]

    safe_rebound = _grade_number(
        question,
        LearnerSubmission(
            answer="0.375",
            reasoning=reasoning,
            python_code=(
                "runner = eval\nrunner = len\nrunner([1])\n"
                "stats.binom.pmf(2, 4, 0.5)"
            ),
        ),
    )
    assert safe_rebound.is_correct is True


def test_first_two_hints_do_not_reveal_distribution_answers() -> None:
    forbidden_by_question = {
        "common_distributions_concept_01": ("应选 A", "二项分布", "4.8"),
        "common_distributions_python_01": ("0.375", "3/8", "6/16", "pmf(2"),
        "common_distributions_interpretation_01": ("应选 B", "0.68269", "点概率为 0"),
    }
    for question_id, forbidden_phrases in forbidden_by_question.items():
        hints = _question(question_id).hints
        assert hints is not None
        early_text = f"{hints.for_level(1)}\n{hints.for_level(2)}"
        assert not any(phrase in early_text for phrase in forbidden_phrases)


def test_distribution_sources_are_registered_without_formal_question_leakage() -> None:
    manifest = load_rag_manifest(ROOT / "data" / "rag" / "manifest.yaml")
    entries = {
        entry.source_id: entry
        for entry in manifest.sources
        if entry.metadata.get("unit_id") == UNIT_ID.value
    }

    assert set(entries) == {
        "discrete_distributions_core",
        "continuous_distributions_core",
    }
    for entry in entries.values():
        loaded = load_rag_source(entry, ROOT)
        source_text = (ROOT / entry.file_path).read_text(encoding="utf-8")
        assert loaded.eligibility.eligible_for_chunking is True
        assert entry.metadata["content_status"] == "pending_teacher_review"
        assert entry.metadata["reviewer_role"] == "course_teacher"
        assert "6 个设备" not in source_text
        assert "0.375" not in source_text
        assert "0.682690" not in source_text
        assert "pdf(30)" not in source_text
