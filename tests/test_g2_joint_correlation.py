"""Content, grading, AST, leakage, source, and graph checks for G2.5 Unit 5."""

from pathlib import Path

import pytest

from probstat_tutor.curriculum import load_default_question_bank
from probstat_tutor.curriculum_graph import (
    UnitContentStatus,
    load_default_curriculum_catalog,
)
from probstat_tutor.graders import combine_submission_evidence, grade_multiple_choice
from probstat_tutor.rag import load_rag_manifest, load_rag_source
from probstat_tutor.schemas import (
    ContentReviewStatus,
    DeepUnitId,
    LearnerSubmission,
    Question,
    QuestionType,
)

ROOT = Path(__file__).resolve().parents[1]
UNIT_ID = DeepUnitId.JOINT_CORRELATION


def _questions() -> tuple[Question, ...]:
    return tuple(
        question
        for question in load_default_question_bank().questions
        if question.unit_id == UNIT_ID
    )


def _question(question_id: str) -> Question:
    return next(question for question in _questions() if question.id == question_id)


def _grade(question: Question, submission: LearnerSubmission):
    return combine_submission_evidence(
        question,
        submission,
        grade_multiple_choice(
            submission.answer,
            str(question.expected_answer),
            accepted_answers=question.accepted_answers,
        ),
    )


def _python_reasoning() -> str:
    return (
        "按 group 分组；mean_score 对非缺失 score 求均值；"
        "count 排除目标列缺失，valid_n 是均值分母；"
        "甲组均值为 69 且有效数为 2；乙组均值为 82 且有效数为 3。"
    )


def _correct_reasoning(question_id: str) -> str:
    if question_id == "joint_correlation_concept_01":
        return (
            "24/60=0.4；24/30=0.8；45/60=0.75；"
            "0.8 与 0.75 不同所以不独立。"
        )
    if question_id == "joint_correlation_python_01":
        return _python_reasoning()
    return (
        "18/60=0.3；相关系数在正比例换算下保持 0.72；"
        "协方差有乘积量纲且相关系数无量纲；只支持样本中的正向线性关联；"
        "不能证明因果且可能有混杂。"
    )


def _correct_code(question_id: str) -> str:
    if question_id != "joint_correlation_python_01":
        return ""
    return (
        'df.groupby("group").agg(mean_score=("score", "mean"), '
        'valid_n=("score", "count"))'
    )


CORE_REVERSAL_CASES = (
    (
        "joint_correlation_concept_01",
        "交集概率应当拿24除以30",
        "joint_probability_wrong_denominator",
    ),
    (
        "joint_correlation_concept_01",
        "条件比例应该仍用60个学生作分母",
        "conditional_probability_wrong_denominator",
    ),
    (
        "joint_correlation_concept_01",
        "B在A条件下应是24/45",
        "conditional_direction_reversed",
    ),
    (
        "joint_correlation_python_01",
        "count也会把score里的空值算在内",
        "count_assumed_to_include_missing",
    ),
    (
        "joint_correlation_python_01",
        "应当按成绩列而不是组别列来分组",
        "wrong_groupby_key",
    ),
    (
        "joint_correlation_python_01",
        "均值计算时应该先把空白补成零",
        "missing_treated_as_zero_in_group_mean",
    ),
    (
        "joint_correlation_interpretation_01",
        "换成小时后r也应该缩小60倍",
        "correlation_scaled_with_units",
    ),
    (
        "joint_correlation_interpretation_01",
        "0.72就是说有72%的可能性",
        "correlation_treated_as_probability",
    ),
    (
        "joint_correlation_interpretation_01",
        "这个趋势对每一位学生都成立",
        "sample_correlation_universalized",
    ),
)


def test_unit_has_three_questions_two_sources_and_pending_status() -> None:
    questions = _questions()
    catalog = load_default_curriculum_catalog()
    unit = next(unit for unit in catalog.units if unit.unit_id == UNIT_ID)

    assert len(questions) == 3
    assert {question.question_type for question in questions} == set(QuestionType)
    assert {question.id for question in questions} == set(unit.question_ids)
    assert all(question.hints is not None for question in questions)
    assert all(question.evidence_policy.reasoning_support_groups for question in questions)
    assert all(
        question.review_status == ContentReviewStatus.PENDING_TEACHER_REVIEW
        for question in questions
    )
    assert {
        question.id for question in questions if question.python_code_required
    } == {"joint_correlation_python_01"}
    assert unit.content_status == UnitContentStatus.PENDING_TEACHER_REVIEW
    assert set(unit.source_ids) == {
        "joint_conditionals_covariance_core",
        "correlation_groupby_core",
    }


@pytest.mark.parametrize(
    ("question_id", "reversal", "expected_tag"),
    CORE_REVERSAL_CASES,
)
def test_declared_core_misconception_synonyms_are_blocked(
    question_id: str,
    reversal: str,
    expected_tag: str,
) -> None:
    result = _grade(
        _question(question_id),
        LearnerSubmission(
            answer="A",
            reasoning=f"{_correct_reasoning(question_id)}不过{reversal}。",
            python_code=_correct_code(question_id),
        ),
    )

    assert result.is_correct is False
    assert expected_tag in result.misconception_candidates


@pytest.mark.parametrize(
    ("question_id", "reversal", "_expected_tag"),
    CORE_REVERSAL_CASES,
)
def test_explicit_negation_of_core_misconception_is_preserved(
    question_id: str,
    reversal: str,
    _expected_tag: str,
) -> None:
    result = _grade(
        _question(question_id),
        LearnerSubmission(
            answer="A",
            reasoning=f"{_correct_reasoning(question_id)}不能说{reversal}。",
            python_code=_correct_code(question_id),
        ),
    )

    assert result.is_correct is True


def test_joint_conditional_question_requires_all_denominators_and_independence() -> None:
    question = _question("joint_correlation_concept_01")
    correct = _grade(
        question,
        LearnerSubmission(
            answer="A",
            reasoning=(
                "联合事件在全部 60 人中计数，所以 24/60=0.4；"
                "给定甲组后分母缩为甲组 30 人，所以 24/30=0.8；"
                "按时完成的边缘概率是全体中的比例，45/60=0.75；"
                "0.8 与 0.75 不同所以不独立。"
            ),
        ),
    )
    assert correct.is_correct is True

    natural_correct = _grade(
        question,
        LearnerSubmission(
            answer="A",
            reasoning=(
                "交集24人占全体60人；已知甲组后30人为分母；"
                "全部60人中45人按时完成；条件与边缘不相等。"
            ),
        ),
    )
    assert natural_correct.is_correct is True

    judge_natural_correct = _grade(
        question,
        LearnerSubmission(
            answer="A",
            reasoning=(
                "交集中的24人占全体60人的0.40；"
                "已知是甲组后，30名甲组学生是分母；"
                "不分组看全部60人，45人按时所以边缘比例是0.75；"
                "条件比例和边缘比例不相等，因此二者不独立。"
            ),
        ),
    )
    assert judge_natural_correct.is_correct is True

    cases = (
        ("联合概率是 24/30。", "joint_probability_wrong_denominator"),
        ("条件概率是 24/60。", "conditional_probability_wrong_denominator"),
        ("P(B|A)=24/45。", "conditional_direction_reversed"),
        ("两个事件都出现过所以独立。", "independence_asserted_without_comparison"),
        (
            "联合事件在全部 60 人中计数；给定甲组后分母缩为甲组 30 人；"
            "按时完成的边缘概率是全体中的比例。",
            "joint_conditional_boundary_unexplained",
        ),
    )
    for reasoning, expected_tag in cases:
        result = _grade(question, LearnerSubmission(answer="A", reasoning=reasoning))
        assert result.is_correct is False
        assert expected_tag in result.misconception_candidates


def test_groupby_named_aggregation_accepts_reviewed_equivalent_forms() -> None:
    question = _question("joint_correlation_python_01")
    codes = (
        'df.groupby("group").agg(mean_score=("score", "mean"), '
        'valid_n=("score", "count"))',
        'df.groupby(by="group").agg(valid_n=("score", "count"), '
        'mean_score=("score", "mean"))',
        'df.groupby("group", as_index=True).agg('
        'mean_score=("score", "mean"), valid_n=("score", "count"))',
        'df.groupby("group", sort=False).agg('
        'mean_score=("score", "mean"), valid_n=("score", "count"))',
        'df.groupby("group", sort=True).agg('
        'mean_score=("score", "mean"), valid_n=("score", "count"))',
        'df.groupby(by=["group"]).agg('
        'mean_score=("score", "mean"), valid_n=("score", "count"))',
        'summary = df.groupby(["group"]).agg(mean_score=("score", "mean"), '
        'valid_n=("score", "count"))\nsummary',
        'df.groupby("group").agg('
        'mean_score=pd.NamedAgg(column="score", aggfunc="mean"), '
        'valid_n=pd.NamedAgg(column="score", aggfunc="count"))',
        'df.groupby("group").agg('
        'valid_n=pandas.NamedAgg(aggfunc="count", column="score"), '
        'mean_score=pandas.NamedAgg(aggfunc="mean", column="score"))',
        'key = "group"\ncolumn = "score"\navg = "mean"\nn = "count"\n'
        'df.groupby(key).agg(mean_score=(column, avg), valid_n=(column, n))',
        'col = "score"\navg = "mean"\nn = "count"\n'
        'df.groupby("group").agg('
        'mean_score=pd.NamedAgg(column=col, aggfunc=avg), '
        'valid_n=pd.NamedAgg(column=col, aggfunc=n))',
    )
    for code in codes:
        result = _grade(
            question,
            LearnerSubmission(
                answer="A",
                reasoning=_python_reasoning(),
                python_code=code,
            ),
        )
        assert result.is_correct is True, code
        assert result.misconception_candidates == []

    natural_reasoning = _grade(
        question,
        LearnerSubmission(
            answer="A",
            reasoning=(
                "以group为键拆组；只平均有记录score；count参与均值个数；"
                "甲62和76平均69共2；乙平均82共3。"
            ),
            python_code=(
                'df.groupby("group").agg(mean_score=("score", "mean"), '
                'valid_n=("score", "count"))'
            ),
        ),
    )
    assert natural_reasoning.is_correct is True

    judge_natural_reasoning = _grade(
        question,
        LearnerSubmission(
            answer="A",
            reasoning=(
                "先以group作为键拆分各组；"
                "mean_score 只平均 score 中有记录的值；"
                "valid_n 用 count 记录参与平均的个数；"
                "甲组的62和76平均为69且共有2个有效值；"
                "乙组70、85、91平均为82且有3个有效值。"
            ),
            python_code=(
                'df.groupby("group").agg(mean_score=("score", "mean"), '
                'valid_n=("score", "count"))'
            ),
        ),
    )
    assert judge_natural_reasoning.is_correct is True


def test_groupby_named_aggregation_rejects_wrong_or_disconnected_structures() -> None:
    question = _question("joint_correlation_python_01")
    cases = (
        (
            'df.groupby("group").agg(mean_score=("score", "mean"), '
            'valid_n=("score", "size"))',
            "groupby_size_used_for_valid_count",
        ),
        (
            'df.groupby("score").agg(mean_score=("score", "mean"), '
            'valid_n=("score", "count"))',
            "groupby_agg_code_conflict",
        ),
        (
            'df.groupby("group").agg(mean_score=("value", "mean"), '
            'valid_n=("value", "count"))',
            "groupby_agg_code_conflict",
        ),
        (
            'df.groupby("group").agg(mean_score=("score", "median"), '
            'valid_n=("score", "count"))',
            "groupby_agg_code_conflict",
        ),
        (
            'df.groupby("group").agg(mean=("score", "mean"), '
            'count=("score", "count"))',
            "groupby_agg_code_conflict",
        ),
        (
            'df.groupby("group").agg(mean_score=("score", "mean"), '
            'valid_n=("score", "count"), total=("score", "sum"))',
            "groupby_agg_code_conflict",
        ),
        ('df.groupby("group")["score"].agg(["mean", "count"])', "groupby_agg_code_conflict"),
        (
            'df.groupby("group").agg(mean_score=("score", "mean"), '
            'valid_n=("score", "count")).reset_index()',
            "groupby_agg_code_conflict",
        ),
        (
            'good = df.groupby("group").agg(mean_score=("score", "mean"), '
            'valid_n=("score", "count"))\n'
            'df.groupby("score").agg(mean_score=("score", "mean"), '
            'valid_n=("score", "count"))',
            "groupby_agg_code_conflict",
        ),
        (
            'df.groupby("Group").agg(mean_score=("score", "mean"), '
            'valid_n=("score", "count"))',
            "groupby_agg_code_conflict",
        ),
        (
            'df = other\ndf.groupby("group").agg('
            'mean_score=("score", "mean"), valid_n=("score", "count"))',
            "groupby_agg_code_conflict",
        ),
        (
            'df, spare = other, 1\ndf.groupby("group").agg('
            'mean_score=("score", "mean"), valid_n=("score", "count"))',
            "groupby_agg_code_conflict",
        ),
        (
            'df.groupby("group").agg(mean_score=["score", "mean"], '
            'valid_n=["score", "count"])',
            "groupby_agg_code_conflict",
        ),
    )
    for code, expected_tag in cases:
        result = _grade(
            question,
            LearnerSubmission(
                answer="A",
                reasoning=_python_reasoning(),
                python_code=code,
            ),
        )
        assert result.is_correct is False, code
        assert result.misconception_candidates == [expected_tag]


def test_groupby_reasoning_rejects_missing_and_grouping_misconceptions() -> None:
    question = _question("joint_correlation_python_01")
    code = (
        'df.groupby("group").agg(mean_score=("score", "mean"), '
        'valid_n=("score", "count"))'
    )
    cases = (
        ("缺失值按 0 参与均值。", "missing_treated_as_zero_in_group_mean"),
        ("size 就是非缺失数量。", "groupby_size_used_for_valid_count"),
        ("按 score 分组。", "wrong_groupby_key"),
        ("按 group 分组并使用 mean 和 count。", "groupby_aggregation_boundary_unexplained"),
    )
    for reasoning, expected_tag in cases:
        result = _grade(
            question,
            LearnerSubmission(answer="A", reasoning=reasoning, python_code=code),
        )
        assert result.is_correct is False
        assert expected_tag in result.misconception_candidates


def test_covariance_correlation_interpretation_requires_scale_and_boundaries() -> None:
    question = _question("joint_correlation_interpretation_01")
    correct = _grade(
        question,
        LearnerSubmission(
            answer="A",
            reasoning=(
                "18/60=0.3，协方差按单位比例缩放；"
                "相关系数在正比例换算下保持 0.72；"
                "协方差有乘积量纲且相关系数无量纲；"
                "只支持样本中的正向线性关联；"
                "不能证明因果且可能有基础能力这一混杂因素。"
            ),
        ),
    )
    assert correct.is_correct is True

    natural_correct = _grade(
        question,
        LearnerSubmission(
            answer="A",
            reasoning=(
                "X除60故协方差除60；r不变；"
                "前者有小时×分量纲后者无单位；只描述80人；"
                "基础能力混杂不能因果。"
            ),
        ),
    )
    assert natural_correct.is_correct is True

    judge_natural_correct = _grade(
        question,
        LearnerSubmission(
            answer="A",
            reasoning=(
                "X 除以60后协方差同样除以60得到0.3；"
                "正比例换单位不会改变 r，所以仍是0.72；"
                "前者带小时乘分的量纲而后者没有单位；"
                "这只描述80人样本中的正向线性关系；"
                "观察记录还可能受基础能力混杂，不能推出因果。"
            ),
        ),
    )
    assert judge_natural_correct.is_correct is True

    cases = (
        ("协方差无量纲。", "covariance_treated_as_unitless"),
        ("相关系数也除以 60。", "correlation_scaled_with_units"),
        ("0.72 表示 72% 的概率。", "correlation_treated_as_probability"),
        ("证明练习时间导致分数提高。", "observational_correlation_treated_as_causation"),
        ("对所有学生都成立。", "sample_correlation_universalized"),
        (
            "18/60=0.3；相关系数在正比例换算下保持 0.72；"
            "协方差有乘积量纲且相关系数无量纲；只支持样本中的正向线性关联。",
            "covariance_correlation_boundary_unexplained",
        ),
    )
    for reasoning, expected_tag in cases:
        result = _grade(question, LearnerSubmission(answer="A", reasoning=reasoning))
        assert result.is_correct is False
        assert expected_tag in result.misconception_candidates


def test_retractions_reversals_and_unsafe_code_cannot_pass() -> None:
    concept = _grade(
        _question("joint_correlation_concept_01"),
        LearnerSubmission(
            answer="A",
            reasoning=(
                "24/60=0.4；24/30=0.8；45/60=0.75；"
                "0.8 与 0.75 不同所以不独立。上句话错误。"
            ),
        ),
    )
    assert concept.is_correct is False

    concept_reversal = _grade(
        _question("joint_correlation_concept_01"),
        LearnerSubmission(
            answer="A",
            reasoning=(
                "24/60=0.4；24/30=0.8；45/60=0.75；"
                "0.8 与 0.75 不同所以不独立；不过这两个事件其实相互独立。"
            ),
        ),
    )
    assert concept_reversal.is_correct is False
    assert "independence_asserted_without_comparison" in (
        concept_reversal.misconception_candidates
    )

    python_reversal = _grade(
        _question("joint_correlation_python_01"),
        LearnerSubmission(
            answer="A",
            reasoning=(
                f"{_python_reasoning()}不过缺失的那一行也应该计入均值分母。"
            ),
            python_code=(
                'df.groupby("group").agg(mean_score=("score", "mean"), '
                'valid_n=("score", "count"))'
            ),
        ),
    )
    assert python_reversal.is_correct is False
    assert "count_assumed_to_include_missing" in (
        python_reversal.misconception_candidates
    )

    interpretation = _grade(
        _question("joint_correlation_interpretation_01"),
        LearnerSubmission(
            answer="A",
            reasoning=(
                "18/60=0.3；相关系数在正比例换算下保持 0.72；"
                "协方差有乘积量纲且相关系数无量纲；只支持样本中的正向线性关联；"
                "不能证明因果且可能有混杂；不过相关系数最终证明练习导致高分。"
            ),
        ),
    )
    assert interpretation.is_correct is False
    assert "observational_correlation_treated_as_causation" in (
        interpretation.misconception_candidates
    )

    unit_reversal = _grade(
        _question("joint_correlation_interpretation_01"),
        LearnerSubmission(
            answer="A",
            reasoning=(
                "18/60=0.3；相关系数在正比例换算下保持 0.72；"
                "协方差有乘积量纲且相关系数无量纲；只支持样本中的正向线性关联；"
                "不能证明因果且可能有混杂；"
                "不过相关系数本身也带着分钟和分数的乘积单位。"
            ),
        ),
    )
    assert unit_reversal.is_correct is False
    assert "correlation_treated_as_unitful" in unit_reversal.misconception_candidates

    causal_reversal = _grade(
        _question("joint_correlation_interpretation_01"),
        LearnerSubmission(
            answer="A",
            reasoning=(
                "18/60=0.3；相关系数在正比例换算下保持 0.72；"
                "协方差有乘积量纲且相关系数无量纲；只支持样本中的正向线性关联；"
                "不能证明因果且可能有混杂；"
                "不过这足以说明练习时间确实造成了成绩提高。"
            ),
        ),
    )
    assert causal_reversal.is_correct is False
    assert "observational_correlation_treated_as_causation" in (
        causal_reversal.misconception_candidates
    )

    unsafe = _grade(
        _question("joint_correlation_python_01"),
        LearnerSubmission(
            answer="A",
            reasoning=_python_reasoning(),
            python_code=(
                'runner = eval\nrunner("1+1")\n'
                'df.groupby("group").agg(mean_score=("score", "mean"), '
                'valid_n=("score", "count"))'
            ),
        ),
    )
    assert unsafe.is_correct is False
    assert unsafe.misconception_candidates == ["unsafe_code_execution_request"]

    unsafe_aliases = (
        (
            "from builtins import eval as runner\n"
            'runner("1+1")\n'
            'df.groupby("group").agg(mean_score=("score", "mean"), '
            'valid_n=("score", "count"))'
        ),
        (
            "runner, spare = eval, len\n"
            'runner("1+1")\n'
            'df.groupby("group").agg(mean_score=("score", "mean"), '
            'valid_n=("score", "count"))'
        ),
        (
            'runner = getattr(__builtins__, "eval")\nrunner("1+1")\n'
            'df.groupby("group").agg(mean_score=("score", "mean"), '
            'valid_n=("score", "count"))'
        ),
        (
            '(runner := eval)("1+1")\n'
            'df.groupby("group").agg(mean_score=("score", "mean"), '
            'valid_n=("score", "count"))'
        ),
        (
            'runner = __builtins__["eval"]\nrunner("1+1")\n'
            'df.groupby("group").agg(mean_score=("score", "mean"), '
            'valid_n=("score", "count"))'
        ),
    )
    for code in unsafe_aliases:
        result = _grade(
            _question("joint_correlation_python_01"),
            LearnerSubmission(
                answer="A",
                reasoning=_python_reasoning(),
                python_code=code,
            ),
        )
        assert result.is_correct is False
        assert result.misconception_candidates == ["unsafe_code_execution_request"]


def test_explicit_corrections_discard_withdrawn_claims() -> None:
    concept = _grade(
        _question("joint_correlation_concept_01"),
        LearnerSubmission(
            answer="A",
            reasoning=(
                "联合概率是 24/30。上句话错误。"
                "24/60=0.4；24/30=0.8；45/60=0.75；"
                "0.8 与 0.75 不同所以不独立。"
            ),
        ),
    )
    assert concept.is_correct is True

    python = _grade(
        _question("joint_correlation_python_01"),
        LearnerSubmission(
            answer="A",
            reasoning=(
                "缺失那一行也应计入均值分母。上句话错误。"
                f"{_python_reasoning()}"
            ),
            python_code=(
                'df.groupby("group").agg(mean_score=("score", "mean"), '
                'valid_n=("score", "count"))'
            ),
        ),
    )
    assert python.is_correct is True

    interpretation = _grade(
        _question("joint_correlation_interpretation_01"),
        LearnerSubmission(
            answer="A",
            reasoning=(
                "相关系数也带分钟和分数乘积单位。上句话错误。"
                "18/60=0.3；相关系数在正比例换算下保持 0.72；"
                "协方差有乘积量纲且相关系数无量纲；只支持样本中的正向线性关联；"
                "不能证明因果且可能有混杂。"
            ),
        ),
    )
    assert interpretation.is_correct is True


def test_first_two_hints_do_not_reveal_formal_answers() -> None:
    forbidden_by_question = {
        "joint_correlation_concept_01": (
            "应选 A",
            "24/60",
            "24/30",
            "45/60",
            "不独立",
        ),
        "joint_correlation_python_01": (
            "应选 A",
            "69.0",
            "82.0",
            'mean_score=("score", "mean")',
            'valid_n=("score", "count")',
        ),
        "joint_correlation_interpretation_01": (
            "应选 A",
            "18/60",
            "0.3",
            "相关系数仍为 0.72",
        ),
    }
    for question_id, forbidden_phrases in forbidden_by_question.items():
        hints = _question(question_id).hints
        assert hints is not None
        early_text = f"{hints.for_level(1)}\n{hints.for_level(2)}"
        assert not any(phrase in early_text for phrase in forbidden_phrases)


def test_joint_sources_are_registered_eligible_and_question_safe() -> None:
    manifest = load_rag_manifest(ROOT / "data" / "rag" / "manifest.yaml")
    entries = {
        entry.source_id: entry
        for entry in manifest.sources
        if entry.metadata.get("unit_id") == UNIT_ID.value
    }

    assert set(entries) == {
        "joint_conditionals_covariance_core",
        "correlation_groupby_core",
    }
    forbidden_fingerprints = (
        "24/60",
        "24/30",
        "45/60",
        "62,76",
        "62, 76",
        "18/60",
        "0.3 小时·分",
    )
    for entry in entries.values():
        loaded = load_rag_source(entry, ROOT)
        source_text = (ROOT / entry.file_path).read_text(encoding="utf-8")
        compact = source_text.replace(" ", "")
        assert loaded.eligibility.eligible_for_chunking is True
        assert entry.metadata["content_status"] == "pending_teacher_review"
        assert entry.metadata["reviewer_role"] == "course_teacher"
        assert not any(
            fingerprint.replace(" ", "") in compact
            for fingerprint in forbidden_fingerprints
        )


def test_joint_graph_edges_exist_and_catalog_remains_acyclic() -> None:
    catalog = load_default_curriculum_catalog()
    edges = {
        (edge.source_node_id, edge.target_node_id, edge.relation.value)
        for edge in catalog.edges
    }
    assert {
        ("jc_joint_conditionals", "jc_covariance", "prerequisite"),
        ("jc_covariance", "jc_correlation_causation", "prerequisite"),
        ("jc_joint_conditionals", "jc_grouping", "supports"),
        ("jc_grouping", "jc_correlation_causation", "supports"),
        ("ds_spread", "jc_covariance", "prerequisite"),
    } <= edges


def test_python_checker_never_calls_pandas_or_executes_submission(monkeypatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("学习者代码不应被执行")

    monkeypatch.setattr("builtins.eval", forbidden)
    monkeypatch.setattr("builtins.exec", forbidden)
    monkeypatch.setattr("pandas.DataFrame.groupby", forbidden)
    result = _grade(
        _question("joint_correlation_python_01"),
        LearnerSubmission(
            answer="A",
            reasoning=_python_reasoning(),
            python_code=(
                'df.groupby("group").agg(mean_score=("score", "mean"), '
                'valid_n=("score", "count"))'
            ),
        ),
    )
    assert result.is_correct is True
