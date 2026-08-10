"""Deterministic graders that never execute learner-submitted code."""

import ast
import math
import re
import unicodedata
from collections.abc import Iterable, Sequence

import pandas as pd

from probstat_tutor.schemas import (
    MAX_PYTHON_CODE_LENGTH,
    CapabilityDimension,
    EvidenceFinding,
    EvidenceVerdict,
    GradeResult,
    LearnerSubmission,
    PythonCallFeature,
    PythonExpressionFeatures,
    PythonMismatchKind,
    PythonMismatchRule,
    PythonStaticAnalysis,
    PythonStaticVariant,
    PythonStructureKind,
    Question,
    ReasoningAssessment,
    SubmissionField,
    TextEvidenceRule,
)

NumericInput = int | float | str
AssignmentVersions = dict[str, tuple[tuple[int, ast.AST], ...]]
ScopedExpression = tuple[ast.AST, int]
MAX_AST_NODES = 500
MAX_AST_DEPTH = 40
MAX_NEGATION_DISTANCE = 8
_BOUNDED_FORMULA_UNSUPPORTED_STATEMENT = "__unsupported_bounded_formula_statement__"
_CLAUSE_SEPARATORS = "。！？；，,.!?;\n"
_ATTACK_NEGATION_GUARDS = ("不要", "不能", "不应", "拒绝")

_PROMPT_INJECTION_PHRASES = (
    "忽略题目",
    "忽略判题器",
    "隐藏 rubric",
    "隐藏rubric",
    "最高优先级系统规则",
)
_SCORE_TAMPERING_PHRASES = (
    "overall_correctness",
    "标记正确",
    "修改分数",
    "分数设为 1",
    "分数设为1",
    "无论我的回答是什么都给满分",
    "overwrite score",
)
_IRRELEVANT_PHRASES = (
    "今天天气",
    "没有回答统计问题",
    "与统计问题无关",
    "drop table",
    "😀😀",
)
_INSUFFICIENT_ANSWERS = frozenset(
    {"不知道", "不会", "不清楚", "信息不足", "无法判断", "不能确定"}
)
_UNSAFE_MODULE_ROOTS = frozenset({"os", "requests", "socket", "subprocess"})
_UNSAFE_CALLS = frozenset(
    {
        "__import__",
        "compile",
        "eval",
        "exec",
        "getattr",
        "open",
        "os.popen",
        "os.system",
        "pathlib.path.write_bytes",
        "pathlib.path.write_text",
        "requests.get",
        "requests.post",
        "socket.socket",
        "subprocess.call",
        "subprocess.popen",
        "subprocess.run",
        "write_bytes",
        "write_text",
    }
)
_UNSAFE_ATTRIBUTES = frozenset({"os.environ"})
_UNSAFE_NAMES = frozenset({"__builtins__"})
_REASONING_RETRACTION_PHRASES = (
    "上句话错误",
    "上一句话错误",
    "前面说法错误",
    "前面的说法错误",
    "上述说法错误",
    "上面说法错误",
    "前述说法错误",
    "前面的结论不成立",
    "以上说法不对",
    "前面说得不对",
    "上面整段都不对",
    "上面整句话都不对",
    "上面这句话不对",
    "以上整段都不对",
    "以上整句话都不对",
    "前面整段都不对",
    "前面整句话不对",
    "这整句话不对",
    "前面全部不对",
    "以上整段结论收回",
    "前面的判断全部作废",
    "刚才整段都收回",
    "以上分析一律撤销",
)
_EVIDENCE_CANONICAL_REPLACEMENTS = (
    ("百分之九十五", "95%"),
    ("九成五", "95%"),
    ("一百", "100"),
    ("采样", "抽样"),
    ("平均数", "均值"),
    ("不可以作为标准", "不适合代表"),
    ("不能作为标准", "不适合代表"),
    ("不适合作为标准", "不适合代表"),
    ("不具有代表性", "不适合代表"),
    ("不太敏感", "不容易受影响"),
    ("不敏感", "不容易受影响"),
    ("带偏", "受影响"),
    ("拖高", "拉高"),
    ("抬高", "拉高"),
    ("取样", "抽样"),
    ("根号", "√"),
    ("除以", "/"),
    ("等于", "="),
    ("抖动", "波动"),
    ("起伏", "波动"),
    ("变动", "波动"),
    ("系统性偏移", "系统偏差"),
    ("系统性偏高", "系统偏差"),
    ("统一的系统偏差", "固定系统偏差"),
    ("统一系统偏差", "固定系统偏差"),
    ("偶然噪声", "随机误差"),
    ("抵消不了", "不能消除"),
    ("冲不掉", "不能消除"),
    ("平均不掉", "不能消除"),
    ("自行归零", "消失"),
    ("更真实", "更准确"),
    ("不能据此说", "不能说"),
    ("手续", "方法"),
    ("几率", "概率"),
    ("正好", "恰好"),
    ("上下限", "区间端点"),
    ("不吻合", "不相容"),
    ("从可能集合中被排除", "逻辑排除"),
    ("彻底出局", "绝对不可能"),
    ("预测带", "预测区间"),
    ("variance", "var"),
    ("均方误差", "mse"),
    ("分位点", "临界值"),
    ("中心偏离目标", "有偏"),
    ("向上带", "系统性高估"),
    ("偶然", "随机"),
    ("纠正不了", "不能消除"),
    ("命中真值", "覆盖真值"),
    ("压到", "缩为"),
    ("稀释掉", "消除"),
)


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
    accepted_answers: Sequence[str] = (),
    misconception_candidates: Sequence[str] = (),
) -> GradeResult:
    """Compare a label or a controlled learner-facing alias after normalization."""

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

    normalized_answers = {
        expected_normalized,
        *(
            normalized
            for answer in accepted_answers
            if (normalized := _normalize_text(answer))
        ),
    }
    is_correct = actual_normalized in normalized_answers
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


def combine_submission_evidence(
    question: Question,
    submission: LearnerSubmission,
    answer_result: GradeResult,
) -> GradeResult:
    """Combine answer correctness with bounded text rules and AST-only code facts."""

    findings: list[EvidenceFinding] = []
    python_analysis = (
        analyze_python_code(submission.python_code)
        if submission.python_code.strip()
        else None
    )

    unsafe_finding = _unsafe_code_finding(submission.python_code, python_analysis)
    if unsafe_finding is not None:
        return _finalize_evidence_grade(answer_result, [unsafe_finding])

    attack_finding = _adversarial_text_finding(submission)
    if attack_finding is not None:
        return _finalize_evidence_grade(answer_result, [attack_finding])

    generic_finding = _generic_text_finding(submission, answer_result)
    if generic_finding is not None:
        return _finalize_evidence_grade(answer_result, [generic_finding])

    for rule in question.evidence_policy.text_rules:
        finding = _apply_text_rule(rule, submission)
        if finding is not None:
            findings.append(finding)

    if not findings:
        relevance_finding = _policy_relevance_finding(
            question,
            submission,
            answer_result,
        )
        if relevance_finding is not None:
            return _finalize_evidence_grade(answer_result, [relevance_finding])

    reasoning_has_conflict = any(
        finding.source == SubmissionField.REASONING
        and finding.verdict == EvidenceVerdict.CONTRADICTS
        for finding in findings
    )
    if not reasoning_has_conflict:
        support_finding = _reasoning_support_finding(question, submission)
        if support_finding is not None:
            findings.append(support_finding)

    if python_analysis is not None:
        python_findings = _python_evidence_findings(
            question,
            submission.python_code,
            python_analysis,
        )
        findings.extend(python_findings)
    elif question.python_code_required:
        findings.append(
            EvidenceFinding(
                rule_id="python_code_required",
                source=SubmissionField.PYTHON_CODE,
                dimension=CapabilityDimension.PYTHON,
                verdict=EvidenceVerdict.INSUFFICIENT,
                message_zh="本题要求提供 Python 代码，但代码栏为空；系统没有执行任何代码。",
                misconception_tag="python_code_missing",
            )
        )

    return _finalize_evidence_grade(answer_result, findings)


def assess_reasoning(
    question: Question,
    submission: LearnerSubmission,
    findings: Sequence[EvidenceFinding],
) -> ReasoningAssessment:
    """Summarize reasoning quality without changing answer correctness."""

    provided = bool(submission.reasoning.strip())
    reasoning_findings = [
        finding
        for finding in findings
        if finding.source == SubmissionField.REASONING
    ]
    if reasoning_findings:
        verdict_priority = {
            EvidenceVerdict.UNSAFE: 0,
            EvidenceVerdict.IRRELEVANT: 1,
            EvidenceVerdict.CONTRADICTS: 2,
            EvidenceVerdict.INSUFFICIENT: 3,
            EvidenceVerdict.SUPPORTS: 4,
        }
        selected = min(
            enumerate(reasoning_findings),
            key=lambda item: (verdict_priority[item[1].verdict], item[0]),
        )[1]
        return ReasoningAssessment(
            required=question.evidence_policy.reasoning_required,
            provided=provided,
            verdict=selected.verdict,
            message_zh=selected.message_zh,
        )

    if not provided:
        message = (
            "题目要求说明理由，但本次没有提供思考过程。"
            if question.evidence_policy.reasoning_required
            else "本题未要求提供理由。"
        )
    else:
        message = "已记录思考过程；本题未配置可单独判定的理由规则。"
    return ReasoningAssessment(
        required=question.evidence_policy.reasoning_required,
        provided=provided,
        message_zh=message,
    )


def contains_model_attack_text(submission: LearnerSubmission) -> bool:
    """Conservatively isolate any raw attack phrase from the optional model."""

    return _locate_phrase(
        submission,
        (*_PROMPT_INJECTION_PHRASES, *_SCORE_TAMPERING_PHRASES),
    ) is not None


def analyze_python_code(code: str) -> PythonStaticAnalysis:
    """Extract bounded AST features without compiling, importing, or running code."""

    if len(code) > MAX_PYTHON_CODE_LENGTH:
        return PythonStaticAnalysis(
            syntax_valid=False,
            node_count=0,
            error_zh="Python 代码超过静态分析长度限制，请缩短后重试。",
        )

    try:
        tree = ast.parse(code, mode="exec")
    except (SyntaxError, ValueError):
        return PythonStaticAnalysis(
            syntax_valid=False,
            node_count=0,
            error_zh="Python 代码存在语法错误；系统只做静态分析，没有执行代码。",
        )
    except (MemoryError, RecursionError):
        return PythonStaticAnalysis(
            syntax_valid=False,
            node_count=0,
            error_zh="Python 代码结构过于复杂，无法安全完成静态分析。",
        )

    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_AST_NODES or _ast_depth(tree) > MAX_AST_DEPTH:
        return PythonStaticAnalysis(
            syntax_valid=False,
            node_count=len(nodes),
            error_zh="Python 代码结构超过静态分析限制，请简化后重试。",
        )

    calls: list[PythonCallFeature] = []
    attributes: set[str] = set()
    names = {node.id.casefold() for node in nodes if isinstance(node, ast.Name)}
    operators = {
        type(node.op).__name__
        for node in nodes
        if isinstance(node, (ast.BinOp, ast.UnaryOp))
    }
    constants = {
        token
        for node in nodes
        if (token := _literal_token(node)) is not None
    }
    unsafe_features: set[str] = set()
    top_level_statement_ids = {id(statement) for statement in tree.body}

    for name in names & _UNSAFE_NAMES:
        unsafe_features.add(f"敏感名称：{name}")

    for node in nodes:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module_names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            for module_name in module_names:
                root = module_name.casefold().split(".", maxsplit=1)[0]
                if root in _UNSAFE_MODULE_ROOTS:
                    unsafe_features.add(f"危险导入：{module_name}")
            if (
                isinstance(node, ast.ImportFrom)
                and id(node) not in top_level_statement_ids
                and (node.module or "").casefold() == "builtins"
            ):
                for alias in node.names:
                    imported_path = alias.name.casefold()
                    if _path_in_set(imported_path, _UNSAFE_CALLS):
                        unsafe_features.add(f"危险内置函数导入：{imported_path}")
        if isinstance(node, ast.Call):
            path = _expression_path(node.func)
            if path:
                normalized_path = path.casefold()
                keyword_constants = tuple(
                    sorted(
                        f"{keyword.arg}={token}"
                        for keyword in node.keywords
                        if keyword.arg is not None
                        and (token := _literal_token(keyword.value)) is not None
                    )
                )
                calls.append(
                    PythonCallFeature(
                        path=normalized_path,
                        keyword_constants=keyword_constants,
                    )
                )
                if _path_in_set(normalized_path, _UNSAFE_CALLS):
                    unsafe_features.add(f"危险调用：{normalized_path}")
        if isinstance(node, ast.NamedExpr):
            assigned_path = _expression_path(node.value)
            if assigned_path is not None and _path_in_set(
                assigned_path.casefold(),
                _UNSAFE_CALLS,
            ):
                unsafe_features.add(
                    f"危险赋值表达式：{assigned_path.casefold()}"
                )
        if isinstance(node, ast.Attribute):
            path = _expression_path(node)
            if path:
                normalized_path = path.casefold()
                attributes.add(normalized_path)
                if _path_in_set(normalized_path, _UNSAFE_ATTRIBUTES):
                    unsafe_features.add(f"敏感属性：{normalized_path}")

    unsafe_features.update(_unsafe_callable_assignment_features(tree))
    unsafe_features.update(_unsafe_alias_call_features(tree))

    return PythonStaticAnalysis(
        syntax_valid=True,
        node_count=len(nodes),
        calls=tuple(sorted(calls, key=lambda item: (item.path, item.keyword_constants))),
        attributes=tuple(sorted(attributes)),
        names=tuple(sorted(names)),
        operators=tuple(sorted(operators)),
        constants=tuple(sorted(constants)),
        result_expressions=_result_expression_features(tree),
        unsafe_features=tuple(sorted(unsafe_features)),
    )


def _finalize_evidence_grade(
    answer_result: GradeResult,
    findings: list[EvidenceFinding],
) -> GradeResult:
    blocking = [
        finding
        for finding in findings
        if finding.verdict != EvidenceVerdict.SUPPORTS
    ]
    answer_blocking = [finding for finding in findings if _finding_blocks_answer(finding)]
    is_correct = answer_result.is_correct and not blocking
    answer_is_correct = bool(answer_result.answer_is_correct) and not answer_blocking
    tags = _unique_strings(
        finding.misconception_tag
        for finding in findings
        if finding.misconception_tag is not None
    )
    evidence = [
        *answer_result.evidence,
        *(
            f"规则 {finding.rule_id}（{finding.source.value}/"
            f"{finding.verdict.value}）：{finding.message_zh}"
            for finding in findings
        ),
    ]
    errors = _unique_strings(
        [
            *answer_result.errors,
            *(finding.message_zh for finding in blocking),
        ]
    )
    return GradeResult(
        score=1.0 if is_correct else 0.0,
        is_correct=is_correct,
        answer_score=1.0 if answer_is_correct else 0.0,
        answer_is_correct=answer_is_correct,
        evidence=evidence,
        errors=errors,
        misconception_candidates=tags,
        findings=findings,
    )


def _finding_blocks_answer(finding: EvidenceFinding) -> bool:
    """Keep pedagogical reasoning feedback separate from the answer verdict."""

    if finding.verdict == EvidenceVerdict.SUPPORTS:
        return False
    return not (
        finding.source == SubmissionField.REASONING
        and finding.verdict
        in {EvidenceVerdict.CONTRADICTS, EvidenceVerdict.INSUFFICIENT}
    )


def _unsafe_code_finding(
    code: str,
    analysis: PythonStaticAnalysis | None,
) -> EvidenceFinding | None:
    if analysis is None or not analysis.unsafe_features:
        return None
    return EvidenceFinding(
        rule_id="unsafe_code_execution_request",
        source=SubmissionField.PYTHON_CODE,
        dimension=CapabilityDimension.PYTHON,
        verdict=EvidenceVerdict.UNSAFE,
        message_zh=(
            "代码包含动态执行、文件、环境、网络或系统访问结构；"
            "系统已拒绝，且从未执行该代码。"
        ),
        quote=_excerpt(code),
        misconception_tag="unsafe_code_execution_request",
    )


def _adversarial_text_finding(
    submission: LearnerSubmission,
) -> EvidenceFinding | None:
    located = _locate_phrase(
        submission,
        _PROMPT_INJECTION_PHRASES,
        negation_guards=_ATTACK_NEGATION_GUARDS,
    )
    if located is not None:
        source, text = located
        return EvidenceFinding(
            rule_id="prompt_injection_attempt",
            source=source,
            verdict=EvidenceVerdict.UNSAFE,
            message_zh="检测到要求忽略题目、判题规则或泄露隐藏内容的指令。",
            quote=_excerpt(text),
            misconception_tag="prompt_injection_attempt",
        )

    located = _locate_phrase(
        submission,
        _SCORE_TAMPERING_PHRASES,
        negation_guards=_ATTACK_NEGATION_GUARDS,
    )
    if located is None:
        return None
    source, text = located
    return EvidenceFinding(
        rule_id="score_tampering_attempt",
        source=source,
        verdict=EvidenceVerdict.UNSAFE,
        message_zh="检测到要求修改正确性或掌握度分数的指令；确定性分数不会被覆盖。",
        quote=_excerpt(text),
        misconception_tag="score_tampering_attempt",
    )


def _generic_text_finding(
    submission: LearnerSubmission,
    answer_result: GradeResult,
) -> EvidenceFinding | None:
    if answer_result.is_correct:
        return None
    normalized_answer = _normalize_evidence_text(submission.answer)
    if normalized_answer in _INSUFFICIENT_ANSWERS:
        return EvidenceFinding(
            rule_id="insufficient_evidence",
            source=SubmissionField.ANSWER,
            verdict=EvidenceVerdict.INSUFFICIENT,
            message_zh="当前回答没有提供足以判断统计理解的内容。",
            quote=_excerpt(submission.answer),
            misconception_tag="insufficient_evidence",
        )
    located = _locate_phrase(submission, _IRRELEVANT_PHRASES)
    if located is None:
        return None
    source, text = located
    return EvidenceFinding(
        rule_id="irrelevant_response",
        source=source,
        verdict=EvidenceVerdict.IRRELEVANT,
        message_zh="当前内容与题目要求的统计任务无关。",
        quote=_excerpt(text),
        misconception_tag="irrelevant_response",
    )


def _apply_text_rule(
    rule: TextEvidenceRule,
    submission: LearnerSubmission,
) -> EvidenceFinding | None:
    text = _submission_field_text(submission, rule.source)
    normalized = _normalize_evidence_text(text)
    if rule.source == SubmissionField.REASONING:
        normalized = _reasoning_after_latest_retraction(normalized)
    if not normalized:
        return None
    if not any(
        _contains_unguarded_phrase(
            normalized,
            phrase,
            rule.negation_guards,
        )
        for phrase in rule.phrases
    ):
        return None
    return EvidenceFinding(
        rule_id=rule.rule_id,
        source=rule.source,
        dimension=rule.dimension,
        verdict=rule.verdict,
        message_zh=rule.message_zh,
        quote=_excerpt(text),
        misconception_tag=rule.misconception_tag,
    )


def _policy_relevance_finding(
    question: Question,
    submission: LearnerSubmission,
    answer_result: GradeResult,
) -> EvidenceFinding | None:
    terms = question.evidence_policy.relevance_terms
    if answer_result.is_correct or not terms:
        return None
    combined = _normalize_evidence_text(
        "\n".join(
            (submission.answer, submission.reasoning, submission.python_code)
        )
    )
    if any(_normalize_evidence_text(term) in combined for term in terms):
        return None
    normalized_answer = _normalize_evidence_text(submission.answer)
    if re.fullmatch(r"[a-z][a-z0-9_]*", normalized_answer):
        return None
    try:
        _parse_number(submission.answer)
    except (TypeError, ValueError):
        pass
    else:
        return None
    if normalized_answer.startswith("[") and normalized_answer.endswith("]"):
        return None
    return EvidenceFinding(
        rule_id="irrelevant_response",
        source=SubmissionField.ANSWER,
        verdict=EvidenceVerdict.IRRELEVANT,
        message_zh="三个提交字段中都没有观察到与本题相关的统计内容。",
        quote=_excerpt(submission.answer),
        misconception_tag="irrelevant_response",
    )


def _reasoning_support_finding(
    question: Question,
    submission: LearnerSubmission,
) -> EvidenceFinding | None:
    policy = question.evidence_policy
    reasoning = submission.reasoning
    normalized = _normalize_evidence_text(reasoning)
    if not normalized:
        if not policy.reasoning_required:
            return None
        return EvidenceFinding(
            rule_id=f"{question.id}_reasoning_required",
            source=SubmissionField.REASONING,
            dimension=_primary_dimension(question),
            verdict=EvidenceVerdict.INSUFFICIENT,
            message_zh="题目要求说明理由，但本次没有提供思考过程。",
            misconception_tag="insufficient_evidence",
        )

    support_groups = policy.reasoning_support_groups
    support_text = _reasoning_after_latest_retraction(normalized)
    support_observed = (
        all(
            any(
                _contains_unguarded_phrase(
                    support_text,
                    phrase,
                    policy.reasoning_support_negation_guards,
                )
                for phrase in group
            )
            for group in support_groups
        )
        if support_groups
        else any(
            _contains_unguarded_phrase(
                support_text,
                phrase,
                policy.reasoning_support_negation_guards,
            )
            for phrase in policy.reasoning_support_any
        )
    )
    if not policy.reasoning_support_any and not support_groups:
        return None
    if support_observed:
        return EvidenceFinding(
            rule_id=f"{question.id}_reasoning_support",
            source=SubmissionField.REASONING,
            dimension=_primary_dimension(question),
            verdict=EvidenceVerdict.SUPPORTS,
            message_zh="思考过程中观察到与本题关键概念一致的依据。",
            quote=_excerpt(reasoning),
        )
    if not policy.reasoning_required:
        return None
    insufficient_rule = policy.reasoning_insufficient_rule
    return EvidenceFinding(
        rule_id=(
            insufficient_rule.rule_id
            if insufficient_rule is not None
            else f"{question.id}_reasoning_insufficient"
        ),
        source=SubmissionField.REASONING,
        dimension=_primary_dimension(question),
        verdict=EvidenceVerdict.INSUFFICIENT,
        message_zh=(
            insufficient_rule.message_zh
            if insufficient_rule is not None
            else "已提供理由，但尚未观察到本题要求的关键统计依据。"
        ),
        quote=_excerpt(reasoning),
        misconception_tag=(
            insufficient_rule.misconception_tag
            if insufficient_rule is not None
            else "insufficient_evidence"
        ),
    )


def _python_evidence_findings(
    question: Question,
    code: str,
    analysis: PythonStaticAnalysis,
) -> list[EvidenceFinding]:
    if not analysis.syntax_valid:
        return [
            EvidenceFinding(
                rule_id="python_syntax_error",
                source=SubmissionField.PYTHON_CODE,
                dimension=CapabilityDimension.PYTHON,
                verdict=EvidenceVerdict.CONTRADICTS,
                message_zh=analysis.error_zh or "Python 代码无法完成静态分析。",
                quote=_excerpt(code),
                misconception_tag="python_syntax_error",
            )
        ]

    spec = question.evidence_policy.python_static_spec
    if spec is None:
        return []
    if any(
        _matches_python_variant(expression, variant, spec.structure_kind)
        for expression in analysis.result_expressions
        for variant in spec.variants
    ):
        return [
            EvidenceFinding(
                rule_id=f"{question.id}_python_structure_support",
                source=SubmissionField.PYTHON_CODE,
                dimension=CapabilityDimension.PYTHON,
                verdict=EvidenceVerdict.SUPPORTS,
                message_zh="静态 AST 中观察到本题所需的函数、参数和运算结构。",
                quote=_excerpt(code),
            )
        ]

    specific_findings = [
        EvidenceFinding(
            rule_id=rule.rule_id,
            source=SubmissionField.PYTHON_CODE,
            dimension=CapabilityDimension.PYTHON,
            verdict=EvidenceVerdict.CONTRADICTS,
            message_zh=rule.message_zh,
            quote=_excerpt(code),
            misconception_tag=rule.misconception_tag,
        )
        for rule in spec.mismatch_rules
        if any(
            _matches_python_mismatch_rule(expression, rule)
            for expression in analysis.result_expressions
        )
    ]
    if specific_findings:
        return specific_findings
    return [
        EvidenceFinding(
            rule_id=spec.mismatch_rule_id,
            source=SubmissionField.PYTHON_CODE,
            dimension=CapabilityDimension.PYTHON,
            verdict=EvidenceVerdict.CONTRADICTS,
            message_zh=spec.mismatch_message_zh,
            quote=_excerpt(code),
            misconception_tag=spec.misconception_tag,
        )
    ]


def _matches_python_variant(
    expression: PythonExpressionFeatures,
    variant: PythonStaticVariant,
    structure_kind: PythonStructureKind,
) -> bool:
    call_paths = tuple(call.path for call in expression.calls)
    keywords = {
        keyword
        for call in expression.calls
        for keyword in call.keyword_constants
    }
    return (
        structure_kind in expression.structure_kinds
        and all(
            any(_paths_match(actual, required) for actual in call_paths)
            for required in variant.required_calls
        )
        and set(variant.required_names).issubset(expression.names)
        and set(variant.required_operators).issubset(expression.operators)
        and (
            not variant.allowed_root_kinds
            or expression.root_kind in variant.allowed_root_kinds
        )
        and (
            variant.allowed_operators is None
            or set(expression.operators).issubset(variant.allowed_operators)
        )
        and set(variant.required_constants).issubset(expression.constants)
        and set(variant.required_keywords).issubset(keywords)
    )


def _matches_python_mismatch_rule(
    expression: PythonExpressionFeatures,
    rule: PythonMismatchRule,
) -> bool:
    return rule.kind in expression.mismatch_kinds


def _ast_depth(tree: ast.AST) -> int:
    maximum = 0
    stack = [(tree, 1)]
    while stack:
        node, depth = stack.pop()
        maximum = max(maximum, depth)
        if maximum > MAX_AST_DEPTH:
            return maximum
        stack.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
    return maximum


def _unsafe_alias_call_features(tree: ast.Module) -> set[str]:
    """Find simple ordered aliases of blocked callables without executing code."""

    unsafe_aliases: dict[str, str] = {}
    features: set[str] = set()
    for statement in tree.body:
        for node in ast.walk(statement):
            if not isinstance(node, ast.Call):
                continue
            path = _expression_path(node.func)
            if path is None:
                continue
            normalized_path = path.casefold()
            if normalized_path.partition(".")[0] not in unsafe_aliases:
                continue
            resolved = _resolve_unsafe_alias_path(normalized_path, unsafe_aliases)
            if resolved is not None and _path_in_set(resolved, _UNSAFE_CALLS):
                features.add(f"危险调用别名：{path.casefold()}→{resolved}")

        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                _update_unsafe_alias_target(
                    unsafe_aliases,
                    target,
                    statement.value,
                )
        elif isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            assigned_path = (
                _expression_path(statement.value)
                if statement.value is not None
                else None
            )
            resolved = (
                _resolve_unsafe_alias_path(assigned_path.casefold(), unsafe_aliases)
                if assigned_path is not None
                else None
            )
            _update_unsafe_alias(
                unsafe_aliases,
                statement.target.id.casefold(),
                resolved,
            )
        elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            unsafe_aliases.pop(statement.name.casefold(), None)
        elif isinstance(statement, ast.ImportFrom):
            module = (statement.module or "").casefold()
            for alias in statement.names:
                bound_name = (alias.asname or alias.name).casefold()
                imported_path = alias.name.casefold() if module == "builtins" else None
                _update_unsafe_alias(
                    unsafe_aliases,
                    bound_name,
                    imported_path,
                )
        elif isinstance(statement, ast.Import):
            for alias in statement.names:
                bound_name = (
                    alias.asname or alias.name.split(".", maxsplit=1)[0]
                ).casefold()
                unsafe_aliases.pop(bound_name, None)

    return features


def _unsafe_callable_assignment_features(tree: ast.Module) -> set[str]:
    """Reject dangerous callable aliases in every scope without executing code."""

    features: set[str] = set()
    top_level_statement_ids = {id(statement) for statement in tree.body}
    for node in ast.walk(tree):
        if id(node) in top_level_statement_ids:
            continue
        if isinstance(node, ast.Assign):
            for target in node.targets:
                features.update(_unsafe_target_value_features(target, node.value))
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            features.update(_unsafe_target_value_features(node.target, node.value))
        elif isinstance(node, ast.NamedExpr):
            features.update(_unsafe_target_value_features(node.target, node.value))
    return features


def _unsafe_target_value_features(target: ast.AST, value: ast.AST) -> set[str]:
    """Return conservative facts for direct or tuple-unpacked unsafe aliases."""

    if isinstance(target, ast.Name):
        assigned_path = _expression_path(value)
        if assigned_path is None or not _path_in_set(
            assigned_path.casefold(),
            _UNSAFE_CALLS,
        ):
            return set()
        return {
            "危险可调用对象赋值："
            f"{target.id.casefold()}→{assigned_path.casefold()}"
        }
    if not isinstance(target, (ast.Tuple, ast.List)) or not isinstance(
        value,
        (ast.Tuple, ast.List),
    ):
        return set()
    if len(target.elts) != len(value.elts):
        return set()
    features: set[str] = set()
    for child_target, child_value in zip(target.elts, value.elts, strict=True):
        features.update(_unsafe_target_value_features(child_target, child_value))
    return features


def _resolve_unsafe_alias_path(
    path: str,
    unsafe_aliases: dict[str, str],
) -> str | None:
    if _path_in_set(path, _UNSAFE_CALLS):
        return path
    root, separator, suffix = path.partition(".")
    resolved_root = unsafe_aliases.get(root)
    if resolved_root is None:
        return None
    return f"{resolved_root}.{suffix}" if separator else resolved_root


def _update_unsafe_alias(
    unsafe_aliases: dict[str, str],
    name: str,
    resolved_path: str | None,
) -> None:
    if resolved_path is not None and _path_in_set(resolved_path, _UNSAFE_CALLS):
        unsafe_aliases[name] = resolved_path
    else:
        unsafe_aliases.pop(name, None)


def _update_unsafe_alias_target(
    unsafe_aliases: dict[str, str],
    target: ast.AST,
    value: ast.AST,
) -> None:
    """Track simple names and pairwise tuple unpacking of blocked callables."""

    if isinstance(target, ast.Name):
        assigned_path = _expression_path(value)
        resolved = (
            _resolve_unsafe_alias_path(assigned_path.casefold(), unsafe_aliases)
            if assigned_path is not None
            else None
        )
        _update_unsafe_alias(unsafe_aliases, target.id.casefold(), resolved)
        return
    if not isinstance(target, (ast.Tuple, ast.List)) or not isinstance(
        value,
        (ast.Tuple, ast.List),
    ):
        return
    if len(target.elts) != len(value.elts):
        return
    for child_target, child_value in zip(target.elts, value.elts, strict=True):
        _update_unsafe_alias_target(unsafe_aliases, child_target, child_value)


def _result_expression_features(
    tree: ast.Module,
) -> tuple[PythonExpressionFeatures, ...]:
    for index in range(len(tree.body) - 1, -1, -1):
        statement = tree.body[index]
        if isinstance(statement, ast.Expr):
            assignments = _assignment_versions_before(tree.body, index)
            return (_expression_features(statement.value, assignments, index),)
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            if statement.value is None:
                return ()
            assignments = _assignment_versions_before(tree.body, index)
            return (_expression_features(statement.value, assignments, index),)
    return ()


def _assignment_versions_before(
    statements: list[ast.stmt],
    end_index: int,
) -> AssignmentVersions:
    versions: dict[str, list[tuple[int, ast.AST]]] = {}
    for statement_index, statement in enumerate(statements[:end_index]):
        if not _is_supported_bounded_formula_prelude(statement):
            versions.setdefault(
                _BOUNDED_FORMULA_UNSUPPORTED_STATEMENT,
                [],
            ).append((statement_index, statement))
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                _record_assignment_versions(
                    versions,
                    target,
                    statement.value,
                    statement_index,
                )
        elif isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            if statement.value is not None:
                versions.setdefault(statement.target.id, []).append(
                    (statement_index, statement.value)
                )
        elif isinstance(statement, ast.AugAssign) and isinstance(
            statement.target,
            ast.Name,
        ):
            # The bounded analyzer does not execute or symbolically evaluate
            # in-place updates. Recording the statement as an opaque new
            # definition prevents a learner from mutating a system-provided
            # variable while the backward slice still treats it as pristine.
            versions.setdefault(statement.target.id, []).append(
                (statement_index, statement)
            )
    return {name: tuple(definitions) for name, definitions in versions.items()}


def _is_supported_bounded_formula_prelude(statement: ast.stmt) -> bool:
    """Allow only auditable, straight-line definitions before a formula result."""

    if isinstance(statement, ast.Assign):
        return (
            all(_is_plain_assignment_target(target) for target in statement.targets)
            and not any(isinstance(node, ast.NamedExpr) for node in ast.walk(statement.value))
        )
    if isinstance(statement, ast.AnnAssign):
        return (
            statement.value is not None
            and _is_plain_assignment_target(statement.target)
            and not any(
                isinstance(node, ast.NamedExpr)
                for node in ast.walk(statement.value)
            )
        )
    return False


def _is_plain_assignment_target(target: ast.AST) -> bool:
    if isinstance(target, ast.Name):
        return True
    return isinstance(target, (ast.Tuple, ast.List)) and all(
        _is_plain_assignment_target(child) for child in target.elts
    )


def _has_unsupported_bounded_formula_prelude(
    assignments: AssignmentVersions,
    use_before_index: int,
) -> bool:
    return (
        _latest_assignment(
            assignments,
            _BOUNDED_FORMULA_UNSUPPORTED_STATEMENT,
            use_before_index,
        )
        is not None
    )


def _record_assignment_versions(
    versions: dict[str, list[tuple[int, ast.AST]]],
    target: ast.AST,
    value: ast.AST,
    statement_index: int,
) -> None:
    """Record simple assignments, including pairwise tuple/list unpacking."""

    if isinstance(target, ast.Name):
        versions.setdefault(target.id, []).append((statement_index, value))
        return
    if not isinstance(target, (ast.Tuple, ast.List)) or not isinstance(
        value,
        (ast.Tuple, ast.List),
    ):
        return
    if len(target.elts) != len(value.elts):
        return
    for child_target, child_value in zip(target.elts, value.elts, strict=True):
        _record_assignment_versions(
            versions,
            child_target,
            child_value,
            statement_index,
        )


def _expression_features(
    expression: ast.AST,
    assignments: AssignmentVersions,
    use_before_index: int,
) -> PythonExpressionFeatures:
    nodes = _backward_slice_nodes(expression, assignments, use_before_index)
    calls: list[PythonCallFeature] = []
    for node in nodes:
        if not isinstance(node, ast.Call):
            continue
        path = _expression_path(node.func)
        if path is None:
            continue
        keyword_constants = tuple(
            sorted(
                f"{keyword.arg}={token}"
                for keyword in node.keywords
                if keyword.arg is not None
                and (token := _literal_token(keyword.value)) is not None
            )
        )
        calls.append(
            PythonCallFeature(
                path=path,
                keyword_constants=keyword_constants,
            )
        )
    resolved_root, resolved_root_index = _resolve_assigned_name(
        expression,
        assignments,
        use_before_index,
    )
    return PythonExpressionFeatures(
        root_kind=type(resolved_root).__name__,
        structure_kinds=_structure_kinds(
            resolved_root,
            assignments,
            resolved_root_index,
        ),
        mismatch_kinds=_mismatch_kinds(
            resolved_root,
            assignments,
            resolved_root_index,
        ),
        calls=tuple(sorted(calls, key=lambda item: (item.path, item.keyword_constants))),
        attributes=tuple(
            sorted(
                {
                    path
                    for node in nodes
                    if isinstance(node, ast.Attribute)
                    and (path := _expression_path(node)) is not None
                }
            )
        ),
        names=tuple(
            sorted({node.id for node in nodes if isinstance(node, ast.Name)})
        ),
        operators=tuple(
            sorted(
                {
                    type(node.op).__name__
                    for node in nodes
                    if isinstance(node, (ast.BinOp, ast.UnaryOp))
                }
            )
        ),
        constants=tuple(
            sorted(
                {
                    token
                    for node in nodes
                    if (token := _literal_token(node)) is not None
                }
            )
        ),
    )


def _backward_slice_nodes(
    expression: ast.AST,
    assignments: AssignmentVersions,
    use_before_index: int,
) -> list[ast.AST]:
    nodes: list[ast.AST] = []
    expanded_definitions: set[tuple[str, int]] = set()
    stack: list[ScopedExpression] = [(expression, use_before_index)]
    while stack:
        node, before_index = stack.pop()
        nodes.append(node)
        if isinstance(node, ast.Name):
            normalized_name = node.id
            definition = _latest_assignment(
                assignments,
                normalized_name,
                before_index,
            )
            if definition is not None:
                definition_index, assigned = definition
                definition_key = (normalized_name, definition_index)
                if definition_key not in expanded_definitions:
                    expanded_definitions.add(definition_key)
                    stack.append((assigned, definition_index))
            continue
        stack.extend((child, before_index) for child in ast.iter_child_nodes(node))
    return nodes


def _resolve_assigned_name(
    expression: ast.AST,
    assignments: AssignmentVersions,
    use_before_index: int,
) -> ScopedExpression:
    current = expression
    current_before_index = use_before_index
    visited_definitions: set[tuple[str, int]] = set()
    while isinstance(current, ast.Name):
        normalized_name = current.id
        definition = _latest_assignment(
            assignments,
            normalized_name,
            current_before_index,
        )
        if definition is None:
            break
        definition_index, assigned = definition
        definition_key = (normalized_name, definition_index)
        if definition_key in visited_definitions:
            break
        visited_definitions.add(definition_key)
        current = assigned
        current_before_index = definition_index
    return current, current_before_index


def _latest_assignment(
    assignments: AssignmentVersions,
    normalized_name: str,
    before_index: int,
) -> tuple[int, ast.AST] | None:
    for definition_index, value in reversed(assignments.get(normalized_name, ())):
        if definition_index < before_index:
            return definition_index, value
    return None


def _structure_kinds(
    resolved_root: ast.AST,
    assignments: AssignmentVersions,
    root_before_index: int,
) -> tuple[PythonStructureKind, ...]:
    kinds: list[PythonStructureKind] = []
    if _is_supported_median_call(resolved_root):
        kinds.append(PythonStructureKind.DIRECT_MEDIAN_CALL)
    if _is_supported_sample_std_call(resolved_root):
        kinds.append(PythonStructureKind.DIRECT_STD_CALL)
    if _is_missing_count_chain(resolved_root, assignments, root_before_index):
        kinds.append(PythonStructureKind.DIRECT_MISSING_COUNT_CHAIN)
    if _is_standard_error_formula(resolved_root, assignments, root_before_index):
        kinds.append(PythonStructureKind.STANDARD_ERROR_FORMULA)
    if _is_confidence_interval_formula(
        resolved_root,
        assignments,
        root_before_index,
    ):
        kinds.append(PythonStructureKind.CONFIDENCE_INTERVAL_FORMULA)
    if _is_seeded_binomial_proportion(
        resolved_root,
        assignments,
        root_before_index,
    ):
        kinds.append(PythonStructureKind.SEEDED_BINOMIAL_PROPORTION)
    if _is_direct_binomial_probability_call(
        resolved_root,
        method="pmf",
        assignments=assignments,
        use_before_index=root_before_index,
    ):
        kinds.append(PythonStructureKind.DIRECT_BINOMIAL_PMF)
    groupby_details = _groupby_named_agg_details(
        resolved_root,
        assignments,
        root_before_index,
    )
    if groupby_details == (
        "group",
        (
            ("mean_score", "score", "mean"),
            ("valid_n", "score", "count"),
        ),
    ):
        kinds.append(PythonStructureKind.DIRECT_GROUPBY_NAMED_AGG)
    welch_details = _welch_ttest_details(
        resolved_root,
        assignments,
        root_before_index,
    )
    if welch_details == (True, frozenset({"variant_a", "variant_b"}), "false", "two-sided"):
        kinds.append(PythonStructureKind.DIRECT_WELCH_TTEST_PVALUE)
    return tuple(kinds)


def _is_supported_median_call(node: ast.AST) -> bool:
    """Accept only a zero-argument median call on the question's `df.value` column."""

    if not isinstance(node, ast.Call) or node.args or node.keywords:
        return False
    if not isinstance(node.func, ast.Attribute) or node.func.attr != "median":
        return False
    target = node.func.value
    if isinstance(target, ast.Attribute):
        return (
            isinstance(target.value, ast.Name)
            and target.value.id == "df"
            and target.attr == "value"
        )
    if not isinstance(target, ast.Subscript):
        return False
    return (
        isinstance(target.value, ast.Name)
        and target.value.id == "df"
        and isinstance(target.slice, ast.Constant)
        and target.slice.value == "value"
    )


def _is_supported_sample_std_call(node: ast.AST) -> bool:
    """Accept the two reviewed expressions for the sample standard deviation."""

    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return False
    path = _expression_path(node.func)
    if path is None:
        return False
    if path == "s.std":
        return not node.args and not node.keywords
    if path != "np.std":
        return False
    if len(node.args) != 1 or not isinstance(node.args[0], ast.Name):
        return False
    if node.args[0].id != "s" or len(node.keywords) != 1:
        return False
    keyword = node.keywords[0]
    return keyword.arg == "ddof" and _literal_token(keyword.value) == "1"


def _is_seeded_binomial_proportion(
    node: ast.AST,
    assignments: AssignmentVersions,
    use_before_index: int,
) -> bool:
    """Recognize one seeded Bernoulli simulation connected to a final proportion."""

    generator_call = _binomial_proportion_generator_call(
        node,
        assignments,
        use_before_index,
    )
    return generator_call is not None and _is_default_rng_call(
        generator_call,
        seed_required=True,
    )


def _binomial_proportion_generator_call(
    node: ast.AST,
    assignments: AssignmentVersions,
    use_before_index: int,
) -> ast.Call | None:
    draws_target: ast.AST | None = None
    draws_use_before_index = use_before_index
    denominator_target: ast.AST | None = None
    denominator_use_before_index = use_before_index
    if (
        isinstance(node, ast.Call)
        and not node.args
        and not node.keywords
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "mean"
    ):
        draws_target = node.func.value
    elif (
        isinstance(node, ast.Call)
        and not node.keywords
        and len(node.args) == 1
        and _expression_path(node.func) == "np.mean"
    ):
        draws_target = node.args[0]
    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        sum_expression, sum_before_index = _resolve_assigned_name(
            node.left,
            assignments,
            use_before_index,
        )
        if (
            isinstance(sum_expression, ast.Call)
            and not sum_expression.args
            and not sum_expression.keywords
            and isinstance(sum_expression.func, ast.Attribute)
            and sum_expression.func.attr == "sum"
        ):
            if _literal_token(node.right) == "1000":
                draws_target = sum_expression.func.value
                draws_use_before_index = sum_before_index
            elif (
                isinstance(node.right, ast.Call)
                and _expression_path(node.right.func) == "len"
                and len(node.right.args) == 1
                and not node.right.keywords
            ):
                draws_target = sum_expression.func.value
                draws_use_before_index = sum_before_index
                denominator_target = node.right.args[0]
    if draws_target is None:
        return None

    draws_expression, draws_before_index = _resolve_assigned_name(
        draws_target,
        assignments,
        draws_use_before_index,
    )
    if (
        not isinstance(draws_expression, ast.Call)
        or not isinstance(draws_expression.func, ast.Attribute)
        or draws_expression.func.attr != "binomial"
    ):
        return None
    if not _has_reviewed_bernoulli_arguments(draws_expression):
        return None
    if denominator_target is not None:
        denominator_expression, _ = _resolve_assigned_name(
            denominator_target,
            assignments,
            denominator_use_before_index,
        )
        if denominator_expression is not draws_expression:
            return None

    generator_expression, _ = _resolve_assigned_name(
        draws_expression.func.value,
        assignments,
        draws_before_index,
    )
    return generator_expression if isinstance(generator_expression, ast.Call) else None


def _has_reviewed_bernoulli_arguments(call: ast.Call) -> bool:
    """Accept positional or keyword n/p, but require size=1000 exactly once."""

    if len(call.args) not in {0, 2}:
        return False
    keyword_values = {
        keyword.arg: _literal_token(keyword.value)
        for keyword in call.keywords
        if keyword.arg is not None
    }
    if len(keyword_values) != len(call.keywords):
        return False
    if call.args:
        if _literal_token(call.args[0]) != "1" or _literal_token(call.args[1]) != "0.5":
            return False
        return keyword_values == {"size": "1000"}
    return keyword_values == {"n": "1", "p": "0.5", "size": "1000"}


def _is_default_rng_call(node: ast.Call, *, seed_required: bool) -> bool:
    if _expression_path(node.func) != "np.random.default_rng":
        return False
    if not seed_required:
        return not node.args and not node.keywords
    return (
        len(node.args) == 1
        and _literal_token(node.args[0]) == "2026"
        and not node.keywords
    ) or (
        not node.args
        and len(node.keywords) == 1
        and node.keywords[0].arg == "seed"
        and _literal_token(node.keywords[0].value) == "2026"
    )


def _is_direct_binomial_probability_call(
    node: ast.AST,
    *,
    method: str,
    assignments: AssignmentVersions,
    use_before_index: int,
) -> bool:
    """Match one reviewed SciPy binomial probability call without executing it."""

    if not isinstance(node, ast.Call):
        return False
    path = _expression_path(node.func)
    if path not in {f"stats.binom.{method}", f"scipy.stats.binom.{method}"}:
        return False

    def resolved_token(value: ast.AST) -> str | None:
        resolved, _ = _resolve_assigned_name(
            value,
            assignments,
            use_before_index,
        )
        return _literal_token(resolved)

    keyword_values = {
        keyword.arg: resolved_token(keyword.value)
        for keyword in node.keywords
        if keyword.arg is not None
    }
    if len(keyword_values) != len(node.keywords):
        return False
    if len(node.args) == 3 and not keyword_values:
        return tuple(resolved_token(argument) for argument in node.args) == (
            "2",
            "4",
            "0.5",
        )
    if len(node.args) == 2:
        return tuple(resolved_token(argument) for argument in node.args) == (
            "2",
            "4",
        ) and keyword_values == {"p": "0.5"}
    if len(node.args) == 1 and resolved_token(node.args[0]) == "2":
        return keyword_values == {"n": "4", "p": "0.5"}
    if not node.args:
        return keyword_values == {"k": "2", "n": "4", "p": "0.5"}
    return False


def _welch_ttest_details(
    node: ast.AST,
    assignments: AssignmentVersions,
    use_before_index: int,
) -> tuple[bool, frozenset[str], str | None, str | None] | None:
    """Read one exact independent-samples t-test call without executing it."""

    if _has_unsupported_bounded_formula_prelude(assignments, use_before_index):
        return None
    root, root_before_index = _resolve_assigned_name(
        node,
        assignments,
        use_before_index,
    )
    has_pvalue = False
    candidate = root
    candidate_before_index = root_before_index
    if isinstance(root, ast.Attribute) and root.attr == "pvalue":
        has_pvalue = True
        candidate, candidate_before_index = _resolve_assigned_name(
            root.value,
            assignments,
            root_before_index,
        )
    if not isinstance(candidate, ast.Call):
        return None
    path = _expression_path(candidate.func)
    if path not in {"stats.ttest_ind", "scipy.stats.ttest_ind"}:
        return None
    package_name = "scipy" if path.startswith("scipy.") else "stats"
    if _latest_assignment(assignments, package_name, candidate_before_index) is not None:
        return None
    if len(candidate.args) != 2 or any(
        isinstance(argument, ast.Starred) for argument in candidate.args
    ):
        return None
    if any(keyword.arg is None for keyword in candidate.keywords):
        return None
    keyword_values = {
        keyword.arg: _literal_token(keyword.value)
        for keyword in candidate.keywords
        if keyword.arg is not None
    }
    if len(keyword_values) != len(candidate.keywords) or not set(keyword_values) <= {
        "equal_var",
        "alternative",
    }:
        return None

    groups: list[str] = []
    for argument in candidate.args:
        if _resolved_name_is(
            argument,
            "variant_a",
            assignments,
            candidate_before_index,
        ):
            groups.append("variant_a")
        elif _resolved_name_is(
            argument,
            "variant_b",
            assignments,
            candidate_before_index,
        ):
            groups.append("variant_b")
        else:
            return None
    return (
        has_pvalue,
        frozenset(groups),
        keyword_values.get("equal_var"),
        keyword_values.get("alternative"),
    )


def _groupby_named_agg_details(
    node: ast.AST,
    assignments: AssignmentVersions,
    use_before_index: int,
) -> tuple[str, tuple[tuple[str, str, str], ...]] | None:
    """Read one exact pandas named aggregation without executing it."""

    root, root_before_index = _resolve_assigned_name(
        node,
        assignments,
        use_before_index,
    )
    if not isinstance(root, ast.Call) or not isinstance(
        root.func,
        ast.Attribute,
    ):
        return None
    if root.func.attr != "agg" or root.args:
        return None
    if any(keyword.arg is None for keyword in root.keywords):
        return None

    groupby_call, groupby_before_index = _resolve_assigned_name(
        root.func.value,
        assignments,
        root_before_index,
    )
    if not isinstance(groupby_call, ast.Call) or not isinstance(
        groupby_call.func,
        ast.Attribute,
    ):
        return None
    if (
        groupby_call.func.attr != "groupby"
        or not isinstance(groupby_call.func.value, ast.Name)
        or groupby_call.func.value.id != "df"
    ):
        return None
    if _latest_assignment(assignments, "df", groupby_before_index) is not None:
        return None
    if any(keyword.arg is None for keyword in groupby_call.keywords):
        return None
    groupby_keywords = {keyword.arg: keyword.value for keyword in groupby_call.keywords}
    if len(groupby_keywords) != len(groupby_call.keywords):
        return None
    if set(groupby_keywords) - {"by", "as_index", "sort"}:
        return None
    as_index = groupby_keywords.get("as_index")
    if as_index is not None and _literal_token(as_index) != "true":
        return None
    sort = groupby_keywords.get("sort")
    if sort is not None and _literal_token(sort) not in {"false", "true"}:
        return None
    if len(groupby_call.args) == 1 and "by" not in groupby_keywords:
        group_argument, _ = _resolve_assigned_name(
            groupby_call.args[0],
            assignments,
            groupby_before_index,
        )
    elif not groupby_call.args and "by" in groupby_keywords:
        group_argument, _ = _resolve_assigned_name(
            groupby_keywords["by"],
            assignments,
            groupby_before_index,
        )
    else:
        return None
    if isinstance(group_argument, ast.List) and len(group_argument.elts) == 1:
        group_argument = group_argument.elts[0]
    group_column = _exact_string_literal(group_argument)
    if group_column is None:
        return None

    aggregations: list[tuple[str, str, str]] = []
    for keyword in root.keywords:
        if keyword.arg is None:
            return None
        value, value_before_index = _resolve_assigned_name(
            keyword.value,
            assignments,
            root_before_index,
        )
        parsed = _named_aggregation_pair(
            value,
            assignments,
            value_before_index,
        )
        if parsed is None:
            return None
        source_column, aggregation = parsed
        aggregations.append((keyword.arg, source_column, aggregation))
    return group_column, tuple(sorted(aggregations))


def _named_aggregation_pair(
    node: ast.AST,
    assignments: AssignmentVersions,
    use_before_index: int,
) -> tuple[str, str] | None:
    if isinstance(node, ast.Tuple) and len(node.elts) == 2:
        column_node, _ = _resolve_assigned_name(
            node.elts[0],
            assignments,
            use_before_index,
        )
        aggregation_node, _ = _resolve_assigned_name(
            node.elts[1],
            assignments,
            use_before_index,
        )
        column = _exact_string_literal(column_node)
        aggregation = _exact_string_literal(aggregation_node)
        return (column, aggregation) if column and aggregation else None
    if not isinstance(node, ast.Call):
        return None
    if _expression_path(node.func) not in {"pd.NamedAgg", "pandas.NamedAgg"}:
        return None
    if node.args or len(node.keywords) != 2:
        return None
    values: dict[str | None, str | None] = {}
    for keyword in node.keywords:
        resolved_value, _ = _resolve_assigned_name(
            keyword.value,
            assignments,
            use_before_index,
        )
        values[keyword.arg] = _exact_string_literal(resolved_value)
    if set(values) != {"column", "aggfunc"}:
        return None
    column = values["column"]
    aggregation = values["aggfunc"]
    return (column, aggregation) if column and aggregation else None


def _exact_string_literal(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _mismatch_kinds(
    resolved_root: ast.AST,
    assignments: AssignmentVersions,
    root_before_index: int,
) -> tuple[PythonMismatchKind, ...]:
    kinds: list[PythonMismatchKind] = []
    if _is_direct_attribute(resolved_root, "median"):
        kinds.append(PythonMismatchKind.DIRECT_MEDIAN_ATTRIBUTE_REFERENCE)
    if _is_scalar_iloc_selection(resolved_root):
        kinds.append(PythonMismatchKind.SCALAR_ILOC_SELECTION)
    if _is_direct_call(resolved_root, "var"):
        kinds.append(PythonMismatchKind.DIRECT_VAR_CALL)
    if _is_numpy_std_missing_ddof(resolved_root):
        kinds.append(PythonMismatchKind.NUMPY_STD_MISSING_DDOF)
    if _is_variance_standard_error_formula(
        resolved_root,
        assignments,
        root_before_index,
    ):
        kinds.append(PythonMismatchKind.VARIANCE_STANDARD_ERROR_FORMULA)
    if _is_population_std_standard_error_formula(
        resolved_root,
        assignments,
        root_before_index,
    ):
        kinds.append(PythonMismatchKind.POPULATION_STD_STANDARD_ERROR_FORMULA)
    if _is_linear_n_standard_error_formula(
        resolved_root,
        assignments,
        root_before_index,
    ):
        kinds.append(PythonMismatchKind.LINEAR_N_STANDARD_ERROR_FORMULA)
    if _is_raw_std_as_standard_error(
        resolved_root,
        assignments,
        root_before_index,
    ):
        kinds.append(PythonMismatchKind.RAW_STD_AS_STANDARD_ERROR)
    if _is_vector_confidence_interval_formula(
        resolved_root,
        assignments,
        root_before_index,
        offset_tokens=("1", "1"),
    ):
        kinds.append(PythonMismatchKind.ADDS_CONFIDENCE_MARGIN_BOTH_SIDES)
    if _is_vector_confidence_interval_formula(
        resolved_root,
        assignments,
        root_before_index,
        offset_tokens=("1", "-1"),
    ):
        kinds.append(PythonMismatchKind.REVERSED_CONFIDENCE_INTERVAL_ENDPOINTS)
    if _is_confidence_interval_without_critical_value(
        resolved_root,
        assignments,
        root_before_index,
    ):
        kinds.append(PythonMismatchKind.OMITS_CONFIDENCE_CRITICAL_VALUE)
    if _is_confidence_interval_with_standard_deviation(
        resolved_root,
        assignments,
        root_before_index,
    ):
        kinds.append(PythonMismatchKind.USES_STANDARD_DEVIATION_AS_CI_SCALE)
    if _is_missing_method_not_called(resolved_root):
        kinds.append(PythonMismatchKind.MISSING_METHOD_NOT_CALLED)
    if _is_direct_call(resolved_root, "len") or _is_direct_call(resolved_root, "count"):
        kinds.append(PythonMismatchKind.NON_MISSING_ROW_COUNT)
    generator_call = _binomial_proportion_generator_call(
        resolved_root,
        assignments,
        root_before_index,
    )
    if generator_call is not None and _is_default_rng_call(
        generator_call,
        seed_required=False,
    ):
        kinds.append(PythonMismatchKind.UNSEEDED_BINOMIAL_PROPORTION)
    if _is_direct_binomial_probability_call(
        resolved_root,
        method="cdf",
        assignments=assignments,
        use_before_index=root_before_index,
    ):
        kinds.append(PythonMismatchKind.BINOMIAL_CDF_FOR_EXACT_PROBABILITY)
    groupby_details = _groupby_named_agg_details(
        resolved_root,
        assignments,
        root_before_index,
    )
    if groupby_details == (
        "group",
        (
            ("mean_score", "score", "mean"),
            ("valid_n", "score", "size"),
        ),
    ):
        kinds.append(PythonMismatchKind.GROUPBY_SIZE_FOR_VALID_COUNT)
    welch_details = _welch_ttest_details(
        resolved_root,
        assignments,
        root_before_index,
    )
    if welch_details is not None:
        has_pvalue, groups, equal_var, alternative = welch_details
        if (
            has_pvalue
            and groups == frozenset({"variant_a", "variant_b"})
            and equal_var in {None, "true"}
            and alternative == "two-sided"
        ):
            kinds.append(PythonMismatchKind.POOLED_TTEST_FOR_WELCH_QUESTION)
        if (
            has_pvalue
            and groups == frozenset({"variant_a", "variant_b"})
            and equal_var == "false"
            and alternative in {"less", "greater"}
        ):
            kinds.append(PythonMismatchKind.ONE_SIDED_TTEST_FOR_TWO_SIDED_QUESTION)
        if (
            has_pvalue
            and len(groups) == 1
            and groups <= {"variant_a", "variant_b"}
            and equal_var == "false"
            and alternative == "two-sided"
        ):
            kinds.append(PythonMismatchKind.TTEST_SAME_GROUP_TWICE)
        if (
            not has_pvalue
            and groups == frozenset({"variant_a", "variant_b"})
            and equal_var == "false"
            and alternative == "two-sided"
        ):
            kinds.append(PythonMismatchKind.TTEST_RESULT_WITHOUT_PVALUE)
    return tuple(kinds)


def _is_missing_count_chain(
    node: ast.AST,
    assignments: AssignmentVersions,
    use_before_index: int,
) -> bool:
    """Recognize `series.isna().sum()` or `series.isnull().sum()` exactly."""

    if not _is_direct_call(node, "sum") or not isinstance(node, ast.Call):
        return False
    if node.args or node.keywords:
        return False
    if not isinstance(node.func, ast.Attribute):
        return False
    missing_call = node.func.value
    if not (
        _is_direct_call(missing_call, "isna")
        or _is_direct_call(missing_call, "isnull")
    ):
        return False
    if not isinstance(missing_call, ast.Call) or not isinstance(
        missing_call.func,
        ast.Attribute,
    ):
        return False
    if missing_call.args or missing_call.keywords:
        return False
    target, _ = _resolve_assigned_name(
        missing_call.func.value,
        assignments,
        use_before_index,
    )
    return _is_score_series_reference(target)


def _is_score_series_reference(node: ast.AST) -> bool:
    if isinstance(node, ast.Subscript):
        path = _expression_path(node.value)
        return (
            path is not None
            and _paths_match(path, "df")
            and _literal_token(node.slice) == "score"
        )
    if isinstance(node, ast.Attribute):
        path = _expression_path(node)
        return path is not None and _paths_match(path, "df.score")
    return False


def _is_missing_method_not_called(node: ast.AST) -> bool:
    """Recognize `series.isna.sum()` where the missing mask was never created."""

    if not _is_direct_call(node, "sum") or not isinstance(node, ast.Call):
        return False
    if not isinstance(node.func, ast.Attribute):
        return False
    missing_attribute = node.func.value
    if not isinstance(missing_attribute, ast.Attribute):
        return False
    path = _expression_path(missing_attribute)
    return path is not None and (
        _paths_match(path, "isna") or _paths_match(path, "isnull")
    )


def _is_direct_attribute(node: ast.AST, required_path: str) -> bool:
    if not isinstance(node, ast.Attribute):
        return False
    path = _expression_path(node)
    return path is not None and _paths_match(path, required_path)


def _is_scalar_iloc_selection(node: ast.AST) -> bool:
    if not isinstance(node, ast.Subscript):
        return False
    path = _expression_path(node.value)
    if path is None or not _paths_match(path, "iloc"):
        return False
    index = node.slice
    if isinstance(index, ast.Constant):
        return isinstance(index.value, int) and not isinstance(index.value, bool)
    if not isinstance(index, ast.BinOp) or not isinstance(index.op, ast.FloorDiv):
        return False
    return (
        _is_call_with_path(index.left, "len")
        and _literal_token(index.right) is not None
    )


def _is_numpy_std_missing_ddof(node: ast.AST) -> bool:
    if not _is_call_with_path(node, "np.std") or not isinstance(node, ast.Call):
        return False
    if len(node.args) != 1:
        return False
    return all(
        keyword.arg is not None and keyword.arg != "ddof"
        for keyword in node.keywords
    )


def _is_direct_call(node: ast.AST, required_path: str) -> bool:
    if not isinstance(node, ast.Call):
        return False
    path = _expression_path(node.func)
    return path is not None and _paths_match(path, required_path)


def _is_standard_error_formula(
    node: ast.AST,
    assignments: AssignmentVersions,
    use_before_index: int,
) -> bool:
    root, root_before_index = _resolve_assigned_name(
        node,
        assignments,
        use_before_index,
    )
    if not isinstance(root, ast.BinOp) or not isinstance(root.op, ast.Div):
        return False
    numerator, numerator_before_index = _resolve_assigned_name(
        root.left,
        assignments,
        root_before_index,
    )
    if not _is_reviewed_sample_moment_call(
        numerator,
        assignments,
        numerator_before_index,
        function_name="std",
        ddof_token="1",
    ):
        return False
    return _is_sample_size_sqrt(
        root.right,
        assignments,
        root_before_index,
    )


def _is_variance_standard_error_formula(
    node: ast.AST,
    assignments: AssignmentVersions,
    use_before_index: int,
) -> bool:
    root, root_before_index = _resolve_assigned_name(
        node,
        assignments,
        use_before_index,
    )
    if not isinstance(root, ast.BinOp) or not isinstance(root.op, ast.Div):
        return False
    numerator, numerator_before_index = _resolve_assigned_name(
        root.left,
        assignments,
        root_before_index,
    )
    return (
        _is_reviewed_sample_moment_call(
            numerator,
            assignments,
            numerator_before_index,
            function_name="var",
            ddof_token="1",
        )
        and _is_sample_size_sqrt(
            root.right,
            assignments,
            root_before_index,
        )
    )


def _is_population_std_standard_error_formula(
    node: ast.AST,
    assignments: AssignmentVersions,
    use_before_index: int,
) -> bool:
    """Recognize an otherwise valid SE formula using a population-SD convention."""

    root, root_before_index = _resolve_assigned_name(
        node,
        assignments,
        use_before_index,
    )
    if not isinstance(root, ast.BinOp) or not isinstance(root.op, ast.Div):
        return False
    numerator, numerator_before_index = _resolve_assigned_name(
        root.left,
        assignments,
        root_before_index,
    )
    uses_population_convention = any(
        _is_reviewed_sample_moment_call(
            numerator,
            assignments,
            numerator_before_index,
            function_name="std",
            ddof_token=ddof_token,
        )
        for ddof_token in (None, "0")
    )
    return uses_population_convention and _is_sample_size_sqrt(
        root.right,
        assignments,
        root_before_index,
    )


def _is_linear_n_standard_error_formula(
    node: ast.AST,
    assignments: AssignmentVersions,
    use_before_index: int,
) -> bool:
    """Recognize sample SD divided by n rather than by sqrt(n)."""

    root, root_before_index = _resolve_assigned_name(
        node,
        assignments,
        use_before_index,
    )
    if not isinstance(root, ast.BinOp) or not isinstance(root.op, ast.Div):
        return False
    numerator, numerator_before_index = _resolve_assigned_name(
        root.left,
        assignments,
        root_before_index,
    )
    return _is_reviewed_sample_moment_call(
        numerator,
        assignments,
        numerator_before_index,
        function_name="std",
        ddof_token="1",
    ) and _is_sample_size_value(root.right, assignments, root_before_index)


def _is_raw_std_as_standard_error(
    node: ast.AST,
    assignments: AssignmentVersions,
    use_before_index: int,
) -> bool:
    """Recognize returning the sample SD without converting it to an SE."""

    root, root_before_index = _resolve_assigned_name(
        node,
        assignments,
        use_before_index,
    )
    return _is_reviewed_sample_moment_call(
        root,
        assignments,
        root_before_index,
        function_name="std",
        ddof_token="1",
    )


def _is_reviewed_sample_moment_call(
    node: ast.AST,
    assignments: AssignmentVersions,
    use_before_index: int,
    *,
    function_name: str,
    ddof_token: str | None,
) -> bool:
    """Validate one exact NumPy/sample moment call without executing it."""

    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return False
    path = _expression_path(node.func)
    if path == f"np.{function_name}":
        if len(node.args) not in {1, 2}:
            return False
        if not _resolved_name_is(node.args[0], "sample", assignments, use_before_index):
            return False
        positional_axis = _literal_token(node.args[1]) if len(node.args) == 2 else None
        if len(node.args) == 2 and positional_axis != "0":
            return False
    elif path == f"sample.{function_name}":
        if not _resolved_name_is(node.func.value, "sample", assignments, use_before_index):
            return False
        if len(node.args) > 1:
            return False
        positional_axis = _literal_token(node.args[0]) if node.args else None
        if node.args and positional_axis != "0":
            return False
    else:
        return False

    keyword_values: dict[str, str | None] = {}
    for keyword in node.keywords:
        if keyword.arg is None or keyword.arg in keyword_values:
            return False
        keyword_values[keyword.arg] = _literal_token(keyword.value)
    if set(keyword_values) - {"axis", "ddof"}:
        return False
    if positional_axis is not None and "axis" in keyword_values:
        return False
    if "axis" in keyword_values and keyword_values["axis"] not in {"0", "none"}:
        return False
    if ddof_token is None:
        return "ddof" not in keyword_values
    return keyword_values.get("ddof") == ddof_token


def _is_sample_size_sqrt(
    node: ast.AST,
    assignments: AssignmentVersions,
    use_before_index: int,
) -> bool:
    square_root, square_root_before_index = _resolve_assigned_name(
        node,
        assignments,
        use_before_index,
    )
    if isinstance(square_root, ast.Call):
        return (
            _expression_path(square_root.func) == "np.sqrt"
            and len(square_root.args) == 1
            and not square_root.keywords
            and _is_sample_size_value(
                square_root.args[0],
                assignments,
                square_root_before_index,
            )
        )
    return (
        isinstance(square_root, ast.BinOp)
        and isinstance(square_root.op, ast.Pow)
        and _literal_token(square_root.right) == "0.5"
        and _is_sample_size_value(
            square_root.left,
            assignments,
            square_root_before_index,
        )
    )


def _is_sample_size_value(
    node: ast.AST,
    assignments: AssignmentVersions,
    use_before_index: int,
) -> bool:
    """Recognize exactly len(sample) or sample.size, including simple assignments."""

    value, value_before_index = _resolve_assigned_name(
        node,
        assignments,
        use_before_index,
    )
    if isinstance(value, ast.Call):
        if (
            _expression_path(value.func) == "len"
            and len(value.args) == 1
            and not value.keywords
            and _resolved_name_is(
                value.args[0],
                "sample",
                assignments,
                value_before_index,
            )
        ):
            return True
        return (
            _expression_path(value.func) == "np.size"
            and len(value.args) == 1
            and not value.keywords
            and _resolved_name_is(
                value.args[0],
                "sample",
                assignments,
                value_before_index,
            )
        )
    if (
        isinstance(value, ast.Attribute)
        and value.attr == "size"
        and _resolved_name_is(
            value.value,
            "sample",
            assignments,
            value_before_index,
        )
    ):
        return True
    if not isinstance(value, ast.Subscript) or _literal_token(value.slice) != "0":
        return False
    shape = value.value
    return (
        isinstance(shape, ast.Attribute)
        and shape.attr == "shape"
        and _resolved_name_is(
            shape.value,
            "sample",
            assignments,
            value_before_index,
        )
    )


def _is_confidence_interval_formula(
    node: ast.AST,
    assignments: AssignmentVersions,
    use_before_index: int,
) -> bool:
    if _is_vector_confidence_interval_formula(
        node,
        assignments,
        use_before_index,
        offset_tokens=("-1", "1"),
    ):
        return True
    return _is_explicit_confidence_interval_array(
        node,
        assignments,
        use_before_index,
    )


def _is_vector_confidence_interval_formula(
    node: ast.AST,
    assignments: AssignmentVersions,
    use_before_index: int,
    *,
    offset_tokens: tuple[str, str],
) -> bool:
    if _has_unsupported_bounded_formula_prelude(assignments, use_before_index):
        return False
    root, root_before_index = _resolve_assigned_name(
        node,
        assignments,
        use_before_index,
    )
    if not isinstance(root, ast.BinOp) or not isinstance(root.op, ast.Add):
        return False
    for mean_side, margin_side in ((root.left, root.right), (root.right, root.left)):
        if not _resolved_name_is(
            mean_side,
            "mean",
            assignments,
            root_before_index,
        ):
            continue
        factors = _flatten_mult_factors(
            margin_side,
            assignments,
            root_before_index,
        )
        if len(factors) != 3:
            continue
        has_offsets = any(
            _is_offset_array(
                factor,
                assignments,
                before_index,
                offset_tokens=offset_tokens,
            )
            for factor, before_index in factors
        )
        has_critical_value = any(
            _is_confidence_critical_value(
                factor,
                assignments,
                before_index,
            )
            for factor, before_index in factors
        )
        has_standard_error = any(
            _resolved_name_is(
                factor,
                "standard_error",
                assignments,
                before_index,
            )
            for factor, before_index in factors
        )
        if has_offsets and has_critical_value and has_standard_error:
            return True
    return False


def _is_explicit_confidence_interval_array(
    node: ast.AST,
    assignments: AssignmentVersions,
    use_before_index: int,
) -> bool:
    if _has_unsupported_bounded_formula_prelude(assignments, use_before_index):
        return False
    root, root_before_index = _resolve_assigned_name(
        node,
        assignments,
        use_before_index,
    )
    if not _is_exact_numpy_array_call(
        root,
        assignments,
        root_before_index,
    ):
        return False
    assert isinstance(root, ast.Call)
    values, values_before_index = _resolve_assigned_name(
        root.args[0],
        assignments,
        root_before_index,
    )
    if not isinstance(values, (ast.List, ast.Tuple)) or len(values.elts) != 2:
        return False
    lower, upper = values.elts
    return _is_confidence_interval_endpoint(
        lower,
        assignments,
        values_before_index,
        lower=True,
    ) and _is_confidence_interval_endpoint(
        upper,
        assignments,
        values_before_index,
        lower=False,
    )


def _is_confidence_interval_endpoint(
    node: ast.AST,
    assignments: AssignmentVersions,
    use_before_index: int,
    *,
    lower: bool,
) -> bool:
    root, root_before_index = _resolve_assigned_name(
        node,
        assignments,
        use_before_index,
    )
    if not isinstance(root, ast.BinOp):
        return False
    if lower:
        return (
            isinstance(root.op, ast.Sub)
            and _resolved_name_is(
                root.left,
                "mean",
                assignments,
                root_before_index,
            )
            and _is_confidence_margin(
                root.right,
                assignments,
                root_before_index,
                scale_name="standard_error",
                require_critical_value=True,
            )
        )
    if not isinstance(root.op, ast.Add):
        return False
    for mean_side, margin_side in ((root.left, root.right), (root.right, root.left)):
        if _resolved_name_is(
            mean_side,
            "mean",
            assignments,
            root_before_index,
        ) and _is_confidence_margin(
            margin_side,
            assignments,
            root_before_index,
            scale_name="standard_error",
            require_critical_value=True,
        ):
            return True
    return False


def _is_confidence_margin(
    node: ast.AST,
    assignments: AssignmentVersions,
    use_before_index: int,
    *,
    scale_name: str,
    require_critical_value: bool,
) -> bool:
    if not require_critical_value:
        return _resolved_name_is(node, scale_name, assignments, use_before_index)
    factors = _flatten_mult_factors(node, assignments, use_before_index)
    if len(factors) != 2:
        return False
    return any(
        _is_confidence_critical_value(factor, assignments, before_index)
        for factor, before_index in factors
    ) and any(
        _resolved_name_is(factor, scale_name, assignments, before_index)
        for factor, before_index in factors
    )


def _is_confidence_critical_value(
    node: ast.AST,
    assignments: AssignmentVersions,
    use_before_index: int,
) -> bool:
    return _literal_token(node) == "1.96" or _resolved_name_is(
        node,
        "critical_value",
        assignments,
        use_before_index,
    )


def _is_confidence_interval_without_critical_value(
    node: ast.AST,
    assignments: AssignmentVersions,
    use_before_index: int,
) -> bool:
    if _has_unsupported_bounded_formula_prelude(assignments, use_before_index):
        return False
    root, root_before_index = _resolve_assigned_name(
        node,
        assignments,
        use_before_index,
    )
    if not isinstance(root, ast.BinOp) or not isinstance(root.op, ast.Add):
        return False
    for mean_side, margin_side in ((root.left, root.right), (root.right, root.left)):
        if not _resolved_name_is(
            mean_side,
            "mean",
            assignments,
            root_before_index,
        ):
            continue
        factors = _flatten_mult_factors(
            margin_side,
            assignments,
            root_before_index,
        )
        if len(factors) != 2:
            continue
        has_offsets = any(
            _is_offset_array(
                factor,
                assignments,
                before_index,
                offset_tokens=("-1", "1"),
            )
            for factor, before_index in factors
        )
        has_standard_error = any(
            _resolved_name_is(
                factor,
                "standard_error",
                assignments,
                before_index,
            )
            for factor, before_index in factors
        )
        if has_offsets and has_standard_error:
            return True
    return False


def _is_confidence_interval_with_standard_deviation(
    node: ast.AST,
    assignments: AssignmentVersions,
    use_before_index: int,
) -> bool:
    if _has_unsupported_bounded_formula_prelude(assignments, use_before_index):
        return False
    root, root_before_index = _resolve_assigned_name(
        node,
        assignments,
        use_before_index,
    )
    if not isinstance(root, ast.BinOp) or not isinstance(root.op, ast.Add):
        return False
    for mean_side, margin_side in ((root.left, root.right), (root.right, root.left)):
        if not _resolved_name_is(
            mean_side,
            "mean",
            assignments,
            root_before_index,
        ):
            continue
        factors = _flatten_mult_factors(
            margin_side,
            assignments,
            root_before_index,
        )
        if len(factors) != 3:
            continue
        has_offsets = any(
            _is_offset_array(
                factor,
                assignments,
                before_index,
                offset_tokens=("-1", "1"),
            )
            for factor, before_index in factors
        )
        has_critical_value = any(
            _is_confidence_critical_value(
                factor,
                assignments,
                before_index,
            )
            for factor, before_index in factors
        )
        has_standard_deviation = any(
            _resolved_name_is(
                factor,
                "standard_deviation",
                assignments,
                before_index,
            )
            for factor, before_index in factors
        )
        if has_offsets and has_critical_value and has_standard_deviation:
            return True
    return False


def _flatten_mult_factors(
    node: ast.AST,
    assignments: AssignmentVersions,
    use_before_index: int,
) -> list[ScopedExpression]:
    resolved, resolved_before_index = _resolve_assigned_name(
        node,
        assignments,
        use_before_index,
    )
    if isinstance(resolved, ast.BinOp) and isinstance(resolved.op, ast.Mult):
        return [
            *_flatten_mult_factors(
                resolved.left,
                assignments,
                resolved_before_index,
            ),
            *_flatten_mult_factors(
                resolved.right,
                assignments,
                resolved_before_index,
            ),
        ]
    return [(resolved, resolved_before_index)]


def _is_offset_array(
    node: ast.AST,
    assignments: AssignmentVersions,
    use_before_index: int,
    *,
    offset_tokens: tuple[str, str],
) -> bool:
    resolved, resolved_before_index = _resolve_assigned_name(
        node,
        assignments,
        use_before_index,
    )
    if not _is_exact_numpy_array_call(
        resolved,
        assignments,
        resolved_before_index,
    ):
        return False
    assert isinstance(resolved, ast.Call)
    values, _ = _resolve_assigned_name(
        resolved.args[0],
        assignments,
        resolved_before_index,
    )
    if not isinstance(values, (ast.List, ast.Tuple)):
        return False
    actual_values = tuple(_literal_finite_number(element) for element in values.elts)
    expected_values = tuple(float(token) for token in offset_tokens)
    return actual_values == expected_values


def _is_exact_numpy_array_call(
    node: ast.AST,
    assignments: AssignmentVersions,
    use_before_index: int,
) -> bool:
    return (
        isinstance(node, ast.Call)
        and _expression_path(node.func) == "np.array"
        and len(node.args) == 1
        and not node.keywords
        and _latest_assignment(assignments, "np", use_before_index) is None
    )


def _literal_finite_number(node: ast.AST) -> float | None:
    sign = 1.0
    value_node = node
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        sign = -1.0
        value_node = node.operand
    if (
        not isinstance(value_node, ast.Constant)
        or isinstance(value_node.value, bool)
        or not isinstance(value_node.value, (int, float))
    ):
        return None
    value = sign * float(value_node.value)
    return value if math.isfinite(value) else None


def _is_call_with_path(node: ast.AST, required_path: str) -> bool:
    if not isinstance(node, ast.Call):
        return False
    path = _expression_path(node.func)
    return path is not None and _paths_match(path, required_path)


def _call_has_keyword(
    node: ast.AST,
    keyword_name: str,
    expected_token: str,
) -> bool:
    if not isinstance(node, ast.Call):
        return False
    return any(
        keyword.arg == keyword_name and _literal_token(keyword.value) == expected_token
        for keyword in node.keywords
    )


def _call_first_argument_is_name(
    node: ast.AST,
    expected_name: str,
    assignments: AssignmentVersions,
    use_before_index: int,
) -> bool:
    if not isinstance(node, ast.Call) or not node.args:
        return False
    return _resolved_name_is(
        node.args[0],
        expected_name,
        assignments,
        use_before_index,
    )


def _resolved_name_is(
    node: ast.AST,
    expected_name: str,
    assignments: AssignmentVersions,
    use_before_index: int,
) -> bool:
    resolved, _ = _resolve_assigned_name(
        node,
        assignments,
        use_before_index,
    )
    return isinstance(resolved, ast.Name) and resolved.id == expected_name


def _expression_path(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _expression_path(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if isinstance(node, ast.Subscript):
        return _expression_path(node.value)
    return None


def _literal_token(node: ast.AST) -> str | None:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        operand = _literal_token(node.operand)
        return f"-{operand}" if operand is not None else None
    if not isinstance(node, ast.Constant):
        return None
    value = node.value
    if isinstance(value, bool) or value is None:
        return str(value).casefold()
    if isinstance(value, (int, float, str)):
        return str(value).casefold()
    return None


def _path_in_set(path: str, candidates: frozenset[str]) -> bool:
    return any(_paths_match(path, candidate) for candidate in candidates)


def _paths_match(actual: str, required: str) -> bool:
    return actual == required or actual.endswith(f".{required}")


def _locate_phrase(
    submission: LearnerSubmission,
    phrases: Sequence[str],
    *,
    negation_guards: Sequence[str] = (),
) -> tuple[SubmissionField, str] | None:
    for source in SubmissionField:
        text = _submission_field_text(submission, source)
        normalized = _normalize_evidence_text(text)
        if any(
            _contains_unguarded_phrase(normalized, phrase, negation_guards)
            for phrase in phrases
        ):
            return source, text
    return None


def _contains_unguarded_phrase(
    normalized_text: str,
    phrase: str,
    negation_guards: Sequence[str],
) -> bool:
    normalized_text = _canonicalize_evidence_text(normalized_text)
    normalized_phrase = _canonicalize_evidence_text(phrase)
    search_from = 0
    while True:
        start = normalized_text.find(normalized_phrase, search_from)
        if start < 0:
            return False
        clause_start = max(
            (normalized_text.rfind(separator, 0, start) for separator in _CLAUSE_SEPARATORS),
            default=-1,
        ) + 1
        local_prefix = normalized_text[max(clause_start, start - 16) : start]
        guarded = False
        for guard in negation_guards:
            normalized_guard = _canonicalize_evidence_text(guard)
            guard_start = local_prefix.rfind(normalized_guard)
            if guard_start < 0:
                continue
            distance = len(local_prefix) - guard_start - len(normalized_guard)
            allowed_distance = (
                0 if len(normalized_guard) == 1 else MAX_NEGATION_DISTANCE
            )
            if distance <= allowed_distance:
                guarded = True
                break
        if not guarded:
            phrase_end = start + len(normalized_phrase)
            following_separators = (
                normalized_text.find(separator, phrase_end)
                for separator in _CLAUSE_SEPARATORS
            )
            clause_end = min(
                (position for position in following_separators if position >= 0),
                default=len(normalized_text),
            )
            local_suffix = normalized_text[
                phrase_end : min(clause_end, phrase_end + 16)
            ]
            for guard in negation_guards:
                normalized_guard = _canonicalize_evidence_text(guard)
                guard_start = local_suffix.find(normalized_guard)
                allowed_distance = (
                    0 if len(normalized_guard) == 1 else MAX_NEGATION_DISTANCE
                )
                if 0 <= guard_start <= allowed_distance:
                    guarded = True
                    break
        if not guarded:
            return True
        search_from = start + len(normalized_phrase)


def _reasoning_after_latest_retraction(normalized_text: str) -> str:
    """Ignore claims the learner explicitly withdrew before their latest correction."""

    latest_retraction_end = max(
        (
            position + len(normalized_phrase)
            for phrase in _REASONING_RETRACTION_PHRASES
            if (normalized_phrase := _normalize_evidence_text(phrase))
            and (position := normalized_text.rfind(normalized_phrase)) >= 0
        ),
        default=0,
    )
    return normalized_text[latest_retraction_end:]


def _submission_field_text(
    submission: LearnerSubmission,
    source: SubmissionField,
) -> str:
    return {
        SubmissionField.ANSWER: submission.answer,
        SubmissionField.REASONING: submission.reasoning,
        SubmissionField.PYTHON_CODE: submission.python_code,
    }[source]


def _normalize_evidence_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def _canonicalize_evidence_text(value: str) -> str:
    """Normalize bounded statistical paraphrases for deterministic phrase rules."""

    canonical = _normalize_evidence_text(value)
    for source, target in _EVIDENCE_CANONICAL_REPLACEMENTS:
        canonical = canonical.replace(source, target)
    return re.sub(r"\s+", "", canonical)


def _primary_dimension(question: Question) -> CapabilityDimension:
    return max(
        CapabilityDimension,
        key=lambda dimension: getattr(question.dimension_weights, dimension.value),
    )


def _excerpt(text: str) -> str:
    stripped = text.strip()
    return stripped if len(stripped) <= 500 else f"{stripped[:499]}…"


def _unique_strings(values: Iterable[str | None]) -> list[str]:
    unique: list[str] = []
    for value in values:
        if isinstance(value, str) and value and value not in unique:
            unique.append(value)
    return unique


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
