"""Run offline diagnostic evaluations and print each metric separately."""

import argparse
import asyncio
import hashlib
import json
import platform
import statistics
import tempfile
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from probstat_tutor.config import Settings
from probstat_tutor.curriculum import load_default_question_bank
from probstat_tutor.schemas import (
    CapabilityDimension,
    ConceptId,
    DiagnosticReport,
    Question,
)
from probstat_tutor.service import LearningService, LearningServiceError
from probstat_tutor.storage import LearningStateStore

REQUIRED_CATEGORIES = {
    "fully_correct",
    "concept_correct_code_wrong",
    "code_correct_interpretation_wrong",
    "calculation_correct_conclusion_wrong",
    "pandas_syntax_error",
    "sd_se_confusion",
    "ci_probability_misinterpretation",
    "correct_after_hint",
    "irrelevant",
    "insufficient_information",
}

BASELINE_NAME = "v0.1-mvp-offline-eval"
BASELINE_GIT_TAG = "v0.1-mvp"
EVALUATION_MODE = "offline"
EVAL_ROOT = Path(__file__).resolve().parent
NAMED_DATASET_PATHS = {
    "v0.1": EVAL_ROOT / "cases.jsonl",
    "development": EVAL_ROOT / "development" / "cases_v0.2_dev.jsonl",
    "blind": EVAL_ROOT / "blind" / "cases_v0.2_blind.jsonl",
}

STRICT_BASELINE_METRICS = (
    "deterministic_grading_accuracy",
    "misconception_tag_accuracy",
    "recommended_action_match_rate",
    "level_one_hint_leak_rate",
    "api_failure_rate",
)

METRIC_SCOPES = {
    "deterministic_grading_accuracy": (
        "成功生成报告的案例；系统完全正确判断与 expected_correct 严格匹配"
    ),
    "misconception_tag_accuracy": (
        "成功生成报告的案例；系统误区标签集合与人工标签集合完全相等"
    ),
    "recommended_action_match_rate": (
        "成功生成报告的案例；系统建议分类与 expected_action 完全相等"
    ),
    "level_one_hint_leak_rate": (
        "成功生成报告且 hint_level=1 的案例；feedback 与 recommended_action "
        "命中 forbidden_hint_tokens"
    ),
    "api_failure_rate": (
        "全部离线案例中评测流程未生成 DiagnosticReport 的比例；"
        "不代表真实 OpenAI API 的在线可靠性"
    ),
    "average_latency_ms": (
        "全部离线案例从提交到成功或失败结束的墙钟时间平均值；"
        "受机器负载影响，不进行跨运行精确相等比较"
    ),
}

KNOWN_BASELINE_LIMITATIONS = [
    "当前评测强制使用离线模式，不包含真实 OpenAI API 的网络、限流或提供商失败。",
    "API 失败率实际表示离线评测流程未生成 DiagnosticReport 的比例。",
    "平均延迟依赖机器和当时负载，只检查字段、类型与非负性，不做精确回归比较。",
    "前三项准确率以成功生成报告的案例为分母；当前基线没有失败案例。",
    "误区标签准确率要求标签集合完全相等，不分别报告 precision、recall 或 F1。",
    "一级提示泄露率只检查报告 feedback 与 recommended_action，不检查 Streamlit UI 提示。",
    "一级提示泄露率只有 3 个案例，样本量很小。",
    "36 个案例与当前 12 道题共同开发，尚未建立独立盲测集，可能出现过拟合。",
    "主能力维度按题目 dimension_weights 的最大值分类，因此 calculation 主维度为 0。",
    "确定性判题主要依据 answer，推理或代码文本与答案矛盾时可能无法识别。",
    "尚未通过真实学习者实验验证诊断或推荐是否改善学习效果。",
]


class EvalCase(BaseModel):
    """One human-labelled learner response."""

    model_config = ConfigDict(extra="forbid")

    id: str
    categories: list[str] = Field(min_length=1)
    question_id: str
    answer: str
    reasoning: str = ""
    python_code: str = ""
    hint_level: int = Field(default=0, ge=0, le=3)
    expected_correct: bool
    expected_misconception_tags: list[str]
    expected_action: str
    forbidden_hint_tokens: list[str] = Field(default_factory=list)


class EvalObservation(BaseModel):
    """Measured result for one evaluation case."""

    case: EvalCase
    report: DiagnosticReport | None
    latency_ms: float = Field(ge=0.0)
    api_failed: bool = False


class Metric(BaseModel):
    """A separately reported metric with an explicit numerator and denominator."""

    value: float = Field(ge=0.0)
    numerator: int = Field(ge=0)
    denominator: int = Field(ge=0)


class EvalSummary(BaseModel):
    """Six independent metrics; intentionally no aggregate score."""

    model_config = ConfigDict(extra="forbid")

    deterministic_grading_accuracy: Metric
    misconception_tag_accuracy: Metric
    recommended_action_match_rate: Metric
    level_one_hint_leak_rate: Metric
    average_latency_ms: float = Field(ge=0.0)
    api_failure_rate: Metric

    @model_validator(mode="after")
    def metrics_are_ratios(self) -> Self:
        for name in (
            "deterministic_grading_accuracy",
            "misconception_tag_accuracy",
            "recommended_action_match_rate",
            "level_one_hint_leak_rate",
            "api_failure_rate",
        ):
            metric = getattr(self, name)
            if not 0.0 <= metric.value <= 1.0:
                raise ValueError(f"{name} 必须处于 0 到 1")
        return self


class CaseDistribution(BaseModel):
    """Frozen counts used to notice accidental evaluation-set drift."""

    model_config = ConfigDict(extra="forbid")

    by_concept_id: dict[str, int]
    by_primary_dimension: dict[str, int]
    by_observed_dimension: dict[str, int]
    by_case_category: dict[str, int]
    notes: list[str]


class BaselineRatioMetric(BaseModel):
    """One ratio metric with its exact count-based provenance."""

    model_config = ConfigDict(extra="forbid")

    numerator: int = Field(ge=0)
    denominator: int = Field(ge=0)
    value: float = Field(ge=0.0, le=1.0)
    metric_scope: str = Field(min_length=1)
    exact_regression: Literal[True] = True

    @model_validator(mode="after")
    def value_matches_counts(self) -> Self:
        expected = self.numerator / self.denominator if self.denominator else 0.0
        if abs(self.value - expected) > 1e-12:
            raise ValueError("比例指标的 value 必须等于 numerator / denominator")
        return self


class BaselineLatencyMetric(BaseModel):
    """Environment-sensitive latency recorded without exact regression matching."""

    model_config = ConfigDict(extra="forbid")

    numerator: float = Field(ge=0.0, description="全部案例延迟之和，单位为毫秒")
    denominator: int = Field(ge=0)
    value: float = Field(ge=0.0)
    unit: Literal["milliseconds"] = "milliseconds"
    metric_scope: str = Field(min_length=1)
    exact_regression: Literal[False] = False

    @model_validator(mode="after")
    def value_matches_average(self) -> Self:
        expected = self.numerator / self.denominator if self.denominator else 0.0
        if abs(self.value - expected) > 1e-9:
            raise ValueError("平均延迟的 value 必须等于 numerator / denominator")
        return self


class BaselineMetrics(BaseModel):
    """Six independent baseline metrics; intentionally no aggregate score."""

    model_config = ConfigDict(extra="forbid")

    deterministic_grading_accuracy: BaselineRatioMetric
    misconception_tag_accuracy: BaselineRatioMetric
    recommended_action_match_rate: BaselineRatioMetric
    level_one_hint_leak_rate: BaselineRatioMetric
    api_failure_rate: BaselineRatioMetric
    average_latency_ms: BaselineLatencyMetric


class EvaluationBaseline(BaseModel):
    """Machine-readable v0.1 evaluation baseline."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    baseline_name: str = Field(min_length=1)
    git_tag: str = Field(min_length=1)
    evaluation_mode: Literal["offline"]
    case_count: int = Field(ge=1)
    normalized_case_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_distribution: CaseDistribution
    metrics: BaselineMetrics
    python_version: str = Field(min_length=1)
    operating_system: str = Field(min_length=1)
    generated_at: datetime
    known_limitations: list[str] = Field(min_length=1)


class BaselineMismatchError(ValueError):
    """Specific drift details returned by baseline checks."""


def load_cases(path: str | Path) -> list[EvalCase]:
    """Load JSONL cases and validate count, IDs, and required category coverage."""

    case_path = Path(path)
    cases: list[EvalCase] = []
    for line_number, line in enumerate(
        case_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            cases.append(EvalCase.model_validate_json(line))
        except Exception as error:
            raise ValueError(f"评测案例第 {line_number} 行无效：{error}") from error

    if len(cases) < 30:
        raise ValueError(f"评测案例至少需要 30 个，当前只有 {len(cases)} 个")
    case_ids = [case.id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("评测案例 ID 不能重复")
    covered = {category for case in cases for category in case.categories}
    missing = REQUIRED_CATEGORIES - covered
    if missing:
        raise ValueError(f"评测案例缺少类别：{', '.join(sorted(missing))}")
    return cases


def load_named_dataset(
    dataset: Literal["v0.1", "development", "blind"],
    *,
    custom_cases_path: str | Path | None = None,
) -> list[EvalCase]:
    """Load one explicitly selected set while keeping v0.1 as the default."""

    if dataset == "v0.1":
        return load_cases(custom_cases_path or NAMED_DATASET_PATHS["v0.1"])
    if custom_cases_path is not None:
        raise ValueError("--cases 只能与 v0.1 数据集一起使用")

    if __package__:
        from evals.dataset import load_v2_cases
    else:
        from dataset import load_v2_cases

    v2_cases = load_v2_cases(
        NAMED_DATASET_PATHS[dataset],
        expected_split=dataset,
    )
    return [
        EvalCase.model_validate(case.to_legacy_payload())
        for case in v2_cases
    ]


def normalized_case_sha256(cases: list[EvalCase]) -> str:
    """Hash parsed, canonically encoded cases independently of JSONL line endings."""

    canonical_cases = [
        case.model_dump(mode="json")
        for case in sorted(cases, key=lambda item: item.id)
    ]
    canonical_json = json.dumps(
        canonical_cases,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def analyze_case_distribution(
    cases: list[EvalCase],
    questions: list[Question] | None = None,
) -> CaseDistribution:
    """Count cases by concept, primary weighted dimension, and multi-label category."""

    question_list = questions or load_default_question_bank().questions
    questions_by_id = {question.id: question for question in question_list}
    concept_counts: Counter[str] = Counter()
    primary_dimension_counts: Counter[str] = Counter()
    observed_dimension_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()

    for case in cases:
        try:
            question = questions_by_id[case.question_id]
        except KeyError as error:
            raise ValueError(
                f"评测案例 {case.id} 引用了不存在的题目：{case.question_id}"
            ) from error

        concept_counts[question.concept_id.value] += 1
        primary_dimension = max(
            CapabilityDimension,
            key=lambda dimension: getattr(
                question.dimension_weights,
                dimension.value,
            ),
        )
        primary_dimension_counts[primary_dimension.value] += 1
        for dimension in CapabilityDimension:
            if getattr(question.dimension_weights, dimension.value) > 0:
                observed_dimension_counts[dimension.value] += 1
        category_counts.update(case.categories)

    return CaseDistribution(
        by_concept_id={
            concept.value: concept_counts[concept.value]
            for concept in ConceptId
        },
        by_primary_dimension={
            dimension.value: primary_dimension_counts[dimension.value]
            for dimension in CapabilityDimension
        },
        by_observed_dimension={
            dimension.value: observed_dimension_counts[dimension.value]
            for dimension in CapabilityDimension
        },
        by_case_category=dict(sorted(category_counts.items())),
        notes=[
            "知识点和主能力维度每个案例只计一次，因此各自合计等于 case_count。",
            "主能力维度取题目 dimension_weights 中的最大权重；并列时按能力枚举顺序。",
            "观察维度包含题目中权重大于 0 的所有维度，允许多维，因此合计可大于 case_count。",
            "案例 category 允许多标签，因此 by_case_category 合计可能大于 case_count。",
        ],
    )


async def run_evaluations(cases: list[EvalCase], workdir: Path) -> EvalSummary:
    """Run cases sequentially in isolated offline learner states."""

    settings = Settings(
        openai_api_key=None,
        openai_model=None,
        session_db_path=workdir / "eval_sessions.sqlite3",
        learning_state_db_path=workdir / "eval_learning.sqlite3",
    )
    service = LearningService(
        settings=settings,
        store=LearningStateStore(settings.learning_state_db_path),
    )
    observations: list[EvalObservation] = []

    for case in cases:
        started = time.perf_counter()
        try:
            report = await service.submit(
                learner_id=f"eval-{case.id}",
                session_id=f"eval-session-{case.id}",
                question_id=case.question_id,
                answer=case.answer,
                reasoning=case.reasoning,
                python_code=case.python_code,
                hint_level=case.hint_level,
            )
        except LearningServiceError:
            observations.append(
                EvalObservation(
                    case=case,
                    report=None,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    api_failed=True,
                )
            )
        else:
            observations.append(
                EvalObservation(
                    case=case,
                    report=report,
                    latency_ms=(time.perf_counter() - started) * 1000,
                )
            )

    return summarize(observations)


def summarize(observations: list[EvalObservation]) -> EvalSummary:
    """Calculate independent metrics without blending them into a total score."""

    successful = [observation for observation in observations if observation.report is not None]

    grading_matches = sum(
        (observation.report.overall_correctness == 1.0)
        is observation.case.expected_correct
        for observation in successful
        if observation.report is not None
    )
    tag_matches = sum(
        set(observation.report.misconception_tags)
        == set(observation.case.expected_misconception_tags)
        for observation in successful
        if observation.report is not None
    )
    action_matches = sum(
        _classify_action(observation.report) == observation.case.expected_action
        for observation in successful
        if observation.report is not None
    )

    level_one = [
        observation
        for observation in successful
        if observation.case.hint_level == 1 and observation.report is not None
    ]
    leaks = sum(_hint_leaks(observation) for observation in level_one)
    failures = sum(observation.api_failed for observation in observations)
    latencies = [observation.latency_ms for observation in observations]

    return EvalSummary(
        deterministic_grading_accuracy=_ratio(grading_matches, len(successful)),
        misconception_tag_accuracy=_ratio(tag_matches, len(successful)),
        recommended_action_match_rate=_ratio(action_matches, len(successful)),
        level_one_hint_leak_rate=_ratio(leaks, len(level_one)),
        average_latency_ms=statistics.fmean(latencies) if latencies else 0.0,
        api_failure_rate=_ratio(failures, len(observations)),
    )


def build_baseline(
    cases: list[EvalCase],
    summary: EvalSummary,
    *,
    baseline_name: str = BASELINE_NAME,
    git_tag: str = BASELINE_GIT_TAG,
    generated_at: datetime | None = None,
) -> EvaluationBaseline:
    """Combine frozen case metadata and measured metrics without changing results."""

    case_count = len(cases)
    return EvaluationBaseline(
        baseline_name=baseline_name,
        git_tag=git_tag,
        evaluation_mode=EVALUATION_MODE,
        case_count=case_count,
        normalized_case_sha256=normalized_case_sha256(cases),
        case_distribution=analyze_case_distribution(cases),
        metrics=BaselineMetrics(
            deterministic_grading_accuracy=_baseline_ratio(
                summary.deterministic_grading_accuracy,
                "deterministic_grading_accuracy",
            ),
            misconception_tag_accuracy=_baseline_ratio(
                summary.misconception_tag_accuracy,
                "misconception_tag_accuracy",
            ),
            recommended_action_match_rate=_baseline_ratio(
                summary.recommended_action_match_rate,
                "recommended_action_match_rate",
            ),
            level_one_hint_leak_rate=_baseline_ratio(
                summary.level_one_hint_leak_rate,
                "level_one_hint_leak_rate",
            ),
            api_failure_rate=_baseline_ratio(
                summary.api_failure_rate,
                "api_failure_rate",
            ),
            average_latency_ms=BaselineLatencyMetric(
                numerator=summary.average_latency_ms * case_count,
                denominator=case_count,
                value=summary.average_latency_ms,
                metric_scope=METRIC_SCOPES["average_latency_ms"],
            ),
        ),
        python_version=platform.python_version(),
        operating_system=platform.platform(),
        generated_at=generated_at or datetime.now(UTC),
        known_limitations=KNOWN_BASELINE_LIMITATIONS,
    )


def write_baseline(path: str | Path, baseline: EvaluationBaseline) -> None:
    """Write one UTF-8 machine-readable baseline."""

    baseline_path = Path(path)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(
        f"{baseline.model_dump_json(indent=2)}\n",
        encoding="utf-8",
    )


def load_baseline(path: str | Path) -> EvaluationBaseline:
    """Load and validate one machine-readable baseline."""

    baseline_path = Path(path)
    try:
        content = baseline_path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise ValueError(f"基线文件不存在：{baseline_path}") from error
    return EvaluationBaseline.model_validate_json(content)


def baseline_mismatches(
    expected: EvaluationBaseline,
    current: EvaluationBaseline,
) -> list[str]:
    """Return precise drift messages; latency value is deliberately not compared."""

    mismatches: list[str] = []
    if expected.normalized_case_sha256 != current.normalized_case_sha256:
        mismatches.append(
            "案例指纹漂移："
            f"基线为 {expected.normalized_case_sha256}，"
            f"当前为 {current.normalized_case_sha256}"
        )
    if expected.case_count != current.case_count:
        mismatches.append(
            f"案例数量漂移：基线为 {expected.case_count}，当前为 {current.case_count}"
        )
    if expected.case_distribution != current.case_distribution:
        mismatches.append(
            "案例分布漂移："
            f"基线为 {expected.case_distribution.model_dump()}，"
            f"当前为 {current.case_distribution.model_dump()}"
        )

    for metric_name in STRICT_BASELINE_METRICS:
        expected_metric = getattr(expected.metrics, metric_name)
        current_metric = getattr(current.metrics, metric_name)
        if expected_metric != current_metric:
            mismatches.append(
                f"确定性指标漂移（{metric_name}）："
                f"基线为 {expected_metric.numerator}/{expected_metric.denominator}"
                f"={expected_metric.value}，"
                f"当前为 {current_metric.numerator}/{current_metric.denominator}"
                f"={current_metric.value}"
            )
    return mismatches


def check_baseline(
    expected: EvaluationBaseline,
    current: EvaluationBaseline,
) -> None:
    """Raise one readable error containing every detected deterministic drift."""

    mismatches = baseline_mismatches(expected, current)
    if mismatches:
        details = "\n".join(f"- {message}" for message in mismatches)
        raise BaselineMismatchError(f"评测基线检查失败：\n{details}")


def print_summary(summary: EvalSummary) -> None:
    """Print every metric on its own line; do not calculate a total score."""

    _print_ratio("确定性判题准确率", summary.deterministic_grading_accuracy)
    _print_ratio("误区标签准确率", summary.misconception_tag_accuracy)
    _print_ratio("下一步建议匹配率", summary.recommended_action_match_rate)
    _print_ratio("一级提示泄露答案比例", summary.level_one_hint_leak_rate)
    print(f"平均延迟: {summary.average_latency_ms:.2f} ms")
    _print_ratio("API 失败比例", summary.api_failure_rate)


def _classify_action(report: DiagnosticReport) -> str:
    if report.overall_correctness < 1.0:
        return "retry_with_guidance"
    if report.next_question_id is not None:
        return "next_question"
    return "complete"


def _hint_leaks(observation: EvalObservation) -> bool:
    report = observation.report
    if report is None:
        return False
    learner_facing_hint = f"{report.feedback}\n{report.recommended_action}".casefold()
    return any(
        token.strip().casefold() in learner_facing_hint
        for token in observation.case.forbidden_hint_tokens
        if token.strip()
    )


def _ratio(numerator: int, denominator: int) -> Metric:
    return Metric(
        value=numerator / denominator if denominator else 0.0,
        numerator=numerator,
        denominator=denominator,
    )


def _baseline_ratio(metric: Metric, metric_name: str) -> BaselineRatioMetric:
    return BaselineRatioMetric(
        numerator=metric.numerator,
        denominator=metric.denominator,
        value=metric.value,
        metric_scope=METRIC_SCOPES[metric_name],
    )


def _print_ratio(label: str, metric: Metric) -> None:
    print(
        f"{label}: {metric.value:.2%} "
        f"({metric.numerator}/{metric.denominator})"
    )


def main() -> int:
    """Run one explicitly selected evaluation set."""
    parser = argparse.ArgumentParser(description="运行离线诊断评测")
    parser.add_argument(
        "--dataset",
        choices=("v0.1", "development", "blind"),
        default="v0.1",
        help="选择评测数据集；默认只加载冻结的 v0.1 案例",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=None,
        help="JSONL 评测案例路径",
    )
    parser.add_argument("--limit", type=int, default=None, help="仅运行前 N 个案例")
    baseline_group = parser.add_mutually_exclusive_group()
    baseline_group.add_argument(
        "--write-baseline",
        type=Path,
        help="把本次全量离线结果写成机器可读基线 JSON",
    )
    baseline_group.add_argument(
        "--check-baseline",
        type=Path,
        help="运行全量评测，并检查案例、分布和确定性指标是否漂移",
    )
    args = parser.parse_args()

    if args.dataset != "v0.1" and args.cases is not None:
        parser.error("--cases 只能与 --dataset v0.1 一起使用")
    if args.dataset == "blind" and args.limit is not None:
        parser.error("盲测集禁止使用 --limit，避免通过小样本聚合结果反推标签")
    if args.dataset != "v0.1" and (
        args.write_baseline is not None or args.check_baseline is not None
    ):
        parser.error("v0.1 基线写入和检查只能使用冻结的 v0.1 数据集")

    cases = load_named_dataset(
        args.dataset,
        custom_cases_path=args.cases,
    )
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit 必须大于 0")
        if args.write_baseline is not None or args.check_baseline is not None:
            parser.error("--limit 不能与基线写入或检查同时使用；基线必须使用完整案例集")
        cases = cases[: args.limit]

    with tempfile.TemporaryDirectory(prefix="probstat-evals-") as temporary_directory:
        summary = asyncio.run(run_evaluations(cases, Path(temporary_directory)))
    print(f"数据集: {args.dataset}（{len(cases)} 个案例）")
    print_summary(summary)
    current_baseline = build_baseline(cases, summary)

    if args.write_baseline is not None:
        write_baseline(args.write_baseline, current_baseline)
        print(f"已写入评测基线：{args.write_baseline}")
    elif args.check_baseline is not None:
        expected_baseline = load_baseline(args.check_baseline)
        try:
            check_baseline(expected_baseline, current_baseline)
        except BaselineMismatchError as error:
            print(str(error))
            return 1
        print(
            "评测基线检查通过：案例指纹、案例数量、案例分布和五项比例指标均未漂移；"
            "平均延迟仅检查合法性。"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
