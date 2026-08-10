"""Deterministic, evidence-linked learner recommendations."""

from probstat_tutor.schemas import (
    CapabilityDimension,
    EvidenceFinding,
    EvidenceVerdict,
    GradeResult,
    NextQuestionDecision,
    PolicyStatus,
    Question,
    RecommendationDecision,
    RecommendationKind,
)

_VERDICT_PRIORITY = {
    EvidenceVerdict.UNSAFE: 0,
    EvidenceVerdict.IRRELEVANT: 1,
    EvidenceVerdict.CONTRADICTS: 2,
    EvidenceVerdict.INSUFFICIENT: 3,
    EvidenceVerdict.SUPPORTS: 4,
}

_TAG_ACTIONS = {
    "prompt_injection_attempt": "请只回答当前统计题，不要要求修改判题规则或泄露隐藏内容。",
    "score_tampering_attempt": "请重新提交题目相关答案；分数只能由可观察作答证据决定。",
    "unsafe_code_execution_request": (
        "请移除文件、网络、环境或系统访问代码，再提交与本题有关的计算表达式。"
    ),
    "irrelevant_response": "请回到当前题目，提交你的答案，并说明与题目数据有关的理由。",
    "mean_always_best": "请重新比较异常值对均值与中位数的影响，再说明哪一个更能代表典型水平。",
    "ignores_outlier": "请指出极端观测会怎样改变均值，并据此修正你的理由。",
    "uses_middle_without_averaging": (
        "请检查偶数个数据时需要处理几个中间位置，再修改代码或计算过程。"
    ),
    "python_method_not_called": (
        "请检查代码中的统计方法是否使用括号真正调用，并重新提交结果表达式。"
    ),
    "same_mean_same_spread": "请不要只比较均值；重新观察各组数据偏离均值的程度。",
    "range_ignored": "请比较两组观测的波动或偏离情况，再判断离散程度。",
    "returns_variance": "请区分方差与标准差，并检查结果表达式调用的统计方法。",
    "uses_population_ddof": "请检查 NumPy 标准差的自由度参数是否符合样本标准差口径。",
    "insufficient_statistical_interpretation": (
        "请补充较小标准差为何对应更集中、更稳定，而不只是比较两个数字。"
    ),
    "sample_size_does_not_affect_se": "请重新检查样本量出现在标准误公式的哪个位置，以及变化方向。",
    "uses_variance_in_se_formula": "请检查标准误公式分子需要的是方差还是标准差，再修改实现。",
    "missing_repeated_sampling_condition": "请补充重复抽样和长期覆盖率条件，再解释置信水平。",
    "insufficient_evidence": "请补充支持答案的关键统计关系或适用条件，再提交一次。",
    "python_syntax_error": "请先修正 Python 语法；系统只会静态检查，不会执行你的代码。",
    "python_code_conflicts_with_answer": "请检查最终结果表达式是否真正实现了题目要求的方法。",
    "hardcoded_result_not_implementation": (
        "请用题目给出的变量和公式构造结果，不要只填写硬编码端点。"
    ),
}

_DIMENSION_ACTIONS = {
    CapabilityDimension.CONCEPT: "请重新说明决定方法选择的统计概念，再提交一次。",
    CapabilityDimension.CALCULATION: "请逐步检查公式、代入值和计算方向，再提交一次。",
    CapabilityDimension.PYTHON: "请检查最终代码表达式的函数、参数和运算结构，再提交一次。",
    CapabilityDimension.INTERPRETATION: "请结合题目情境说明这个结果能支持什么、不能支持什么。",
}


def recommend_from_findings(
    question: Question,
    grade: GradeResult,
    next_question: NextQuestionDecision,
) -> RecommendationDecision:
    """Choose one immediate action from deterministic findings and policy output."""

    if grade.answer_is_correct:
        return _correct_recommendation(next_question)

    primary_dimension = _primary_dimension(question)
    primary_finding = _primary_blocking_finding(question, grade.findings)
    if primary_finding is None:
        return RecommendationDecision(
            kind=RecommendationKind.RETRY_ANSWER,
            action_zh=_DIMENSION_ACTIONS[primary_dimension],
            target_dimension=primary_dimension,
        )

    target_dimension = primary_finding.dimension or primary_dimension
    action = _TAG_ACTIONS.get(
        primary_finding.misconception_tag or "",
        _DIMENSION_ACTIONS[target_dimension],
    )
    return RecommendationDecision(
        kind=_retry_kind(primary_finding.verdict),
        action_zh=action,
        target_dimension=target_dimension,
        source_rule_id=primary_finding.rule_id,
    )


def _correct_recommendation(
    decision: NextQuestionDecision,
) -> RecommendationDecision:
    kind_by_status = {
        PolicyStatus.QUESTION: RecommendationKind.NEXT_QUESTION,
        PolicyStatus.COMPLETE: RecommendationKind.COMPLETE,
        PolicyStatus.BLOCKED: RecommendationKind.BLOCKED,
    }
    return RecommendationDecision(
        kind=kind_by_status[decision.status],
        action_zh=decision.reason,
        next_question_id=(
            decision.question_id
            if decision.status == PolicyStatus.QUESTION
            else None
        ),
        target_dimension=decision.target_dimension,
    )


def _primary_blocking_finding(
    question: Question,
    findings: list[EvidenceFinding],
) -> EvidenceFinding | None:
    blocking = [
        (index, finding)
        for index, finding in enumerate(findings)
        if finding.verdict != EvidenceVerdict.SUPPORTS
    ]
    if not blocking:
        return None
    return min(
        blocking,
        key=lambda item: (
            _VERDICT_PRIORITY[item[1].verdict],
            -_dimension_weight(question, item[1].dimension),
            item[0],
        ),
    )[1]


def _retry_kind(verdict: EvidenceVerdict) -> RecommendationKind:
    return {
        EvidenceVerdict.UNSAFE: RecommendationKind.RETRY_UNSAFE,
        EvidenceVerdict.IRRELEVANT: RecommendationKind.RETRY_IRRELEVANT,
        EvidenceVerdict.CONTRADICTS: RecommendationKind.RETRY_CONTRADICTION,
        EvidenceVerdict.INSUFFICIENT: RecommendationKind.RETRY_INSUFFICIENT,
        EvidenceVerdict.SUPPORTS: RecommendationKind.RETRY_ANSWER,
    }[verdict]


def _primary_dimension(question: Question) -> CapabilityDimension:
    return max(
        CapabilityDimension,
        key=lambda dimension: _dimension_weight(question, dimension),
    )


def _dimension_weight(
    question: Question,
    dimension: CapabilityDimension | None,
) -> float:
    if dimension is None:
        return 0.0
    return getattr(question.dimension_weights, dimension.value)
