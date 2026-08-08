"""Run five isolated local journeys plus one injected model-network fallback."""

import argparse
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from probstat_tutor.analytics import TeacherDashboardStatus
from probstat_tutor.config import Settings
from probstat_tutor.schemas import ConceptId, DeliveryMode
from probstat_tutor.service import LearningService


class JourneyResult(BaseModel):
    """Non-sensitive verification result for one isolated profile."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_number: int = Field(ge=1, le=5)
    hint_levels_verified: tuple[int, ...]
    wrong_answer_detected: bool
    correction_accepted: bool
    recommendation_present: bool
    next_question_available: bool
    history_count: int


class ProductDemoSummary(BaseModel):
    """Auditable G3.3 demo result without learner content or identifiers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    run_count: int
    serious_fault_count: int
    journeys: tuple[JourneyResult, ...]
    teacher_status: TeacherDashboardStatus
    teacher_profile_count: int
    teacher_attempt_count: int
    teacher_raw_answer_fields_present: bool
    offline_core_available: bool
    model_failure_fallback_verified: bool
    fault_event_count: int
    passed: bool


async def run_demo(output_path: Path | None = None) -> ProductDemoSummary:
    """Execute the required product path deterministically in temporary storage."""

    with tempfile.TemporaryDirectory(prefix="probstat-g3-product-") as directory:
        root = Path(directory)
        settings = Settings(
            openai_api_key=None,
            openai_model=None,
            learning_state_db_path=root / "learning.sqlite3",
            session_db_path=root / "sessions.sqlite3",
            fault_log_path=root / "logs" / "faults.jsonl",
        )
        service = LearningService(settings=settings)
        journeys: list[JourneyResult] = []

        for run_number in range(1, 6):
            learner_id = service.create_anonymous_learner_id()
            question = service.choose_question(learner_id, ConceptId.DATA_QUALITY)
            hints = tuple(service.get_hint(question.id, level) for level in range(1, 5))
            wrong = await service.submit(
                learner_id=learner_id,
                session_id=f"demo-session-{run_number}",
                question_id=question.id,
                answer="3",
                reasoning="0 也算缺失。",
                hint_level=0,
            )
            corrected = await service.submit(
                learner_id=learner_id,
                session_id=f"demo-session-{run_number}",
                question_id=question.id,
                answer="2",
                reasoning="0 是合法分数，两个空白才是缺失。",
                hint_level=1,
            )
            next_available = corrected.next_question_id is not None
            if corrected.next_question_id is not None:
                service.get_question(corrected.next_question_id)
            history = service.get_dashboard(learner_id, ConceptId.DATA_QUALITY).state.history
            journeys.append(
                JourneyResult(
                    run_number=run_number,
                    hint_levels_verified=tuple(range(1, len(hints) + 1)),
                    wrong_answer_detected=wrong.overall_correctness < 1.0,
                    correction_accepted=corrected.overall_correctness == 1.0,
                    recommendation_present=bool(corrected.recommended_action),
                    next_question_available=next_available,
                    history_count=len(history),
                )
            )

        teacher = service.get_teacher_dashboard()
        teacher_dump = json.dumps(teacher.model_dump(mode="json"), ensure_ascii=False)

        fallback_settings = Settings(
            openai_api_key=SecretStr("demo-key-not-sent"),
            openai_model="demo-model-not-called",
            learning_state_db_path=root / "fallback-learning.sqlite3",
            session_db_path=root / "fallback-sessions.sqlite3",
            fault_log_path=root / "fallback-logs" / "faults.jsonl",
        )
        fallback_service = LearningService(settings=fallback_settings)

        async def fail_model(*args: object, **kwargs: object) -> None:
            raise ConnectionError("injected network outage")

        with patch("probstat_tutor.tutor_agent.Runner.run", new=fail_model):
            fallback_report = await fallback_service.submit(
                learner_id=fallback_service.create_anonymous_learner_id(),
                session_id="fallback-session",
                question_id="data_quality_concept_01",
                answer="2",
                reasoning="0 是合法分数，两个空白才是缺失。",
                hint_level=1,
            )

        fault_event_count = (
            len(fallback_settings.fault_log_path.read_text(encoding="utf-8").splitlines())
            if fallback_settings.fault_log_path.exists()
            else 0
        )
        raw_fields_present = any(
            token in teacher_dump
            for token in ('"learner_id"', '"answer"', '"reasoning"', '"python_code"')
        )
        journeys_tuple = tuple(journeys)
        passed = (
            len(journeys_tuple) == 5
            and all(
                journey.wrong_answer_detected
                and journey.correction_accepted
                and journey.recommendation_present
                and journey.next_question_available
                and journey.history_count == 2
                and journey.hint_levels_verified == (1, 2, 3, 4)
                for journey in journeys_tuple
            )
            and teacher.status == TeacherDashboardStatus.READY
            and teacher.profile_count == 5
            and teacher.total_attempt_count == 10
            and not raw_fields_present
            and service.offline_mode
            and fallback_report.delivery_mode == DeliveryMode.MODEL_FALLBACK
            and fault_event_count == 1
        )
        summary = ProductDemoSummary(
            run_count=len(journeys_tuple),
            serious_fault_count=0 if passed else 1,
            journeys=journeys_tuple,
            teacher_status=teacher.status,
            teacher_profile_count=teacher.profile_count,
            teacher_attempt_count=teacher.total_attempt_count,
            teacher_raw_answer_fields_present=raw_fields_present,
            offline_core_available=service.offline_mode,
            model_failure_fallback_verified=(
                fallback_report.delivery_mode == DeliveryMode.MODEL_FALLBACK
            ),
            fault_event_count=fault_event_count,
            passed=passed,
        )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            f"{summary.model_dump_json(indent=2)}\n", encoding="utf-8"
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = asyncio.run(run_demo(args.output))
    print(summary.model_dump_json(indent=2))
    return 0 if summary.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
