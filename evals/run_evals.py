"""Run offline diagnostic evaluations and print each metric separately."""

import argparse
import asyncio
import statistics
import tempfile
import time
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from probstat_tutor.config import Settings
from probstat_tutor.schemas import DiagnosticReport
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


def _print_ratio(label: str, metric: Metric) -> None:
    print(
        f"{label}: {metric.value:.2%} "
        f"({metric.numerator}/{metric.denominator})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="运行离线诊断评测")
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path(__file__).with_name("cases.jsonl"),
        help="JSONL 评测案例路径",
    )
    parser.add_argument("--limit", type=int, default=None, help="仅运行前 N 个案例")
    args = parser.parse_args()

    cases = load_cases(args.cases)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit 必须大于 0")
        cases = cases[: args.limit]

    with tempfile.TemporaryDirectory(prefix="probstat-evals-") as temporary_directory:
        summary = asyncio.run(run_evaluations(cases, Path(temporary_directory)))
    print_summary(summary)


if __name__ == "__main__":
    main()
