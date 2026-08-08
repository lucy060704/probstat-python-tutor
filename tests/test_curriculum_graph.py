"""Tests for the eight-unit graph and directory-level chapter mapping."""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from probstat_tutor.curriculum import load_default_question_bank
from probstat_tutor.curriculum_graph import (
    CoverageLevel,
    CurriculumGraphLoadError,
    TextbookId,
    UnitContentStatus,
    load_curriculum_catalog,
    load_default_curriculum_catalog,
    validate_catalog_question_readiness,
)
from probstat_tutor.schemas import DeepUnitId, QuestionBank

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "data" / "curriculum_catalog.yaml"


def _catalog_data() -> dict[str, object]:
    data = yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _write_catalog(tmp_path: Path, data: dict[str, object]) -> Path:
    path = tmp_path / "catalog.yaml"
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def test_default_catalog_has_eight_units_and_exact_8_plus_14_chapters() -> None:
    catalog = load_default_curriculum_catalog()

    assert catalog.coverage_level_semantics == "target_only_not_implementation_status"
    assert {unit.unit_id for unit in catalog.units} == set(DeepUnitId)
    assert [unit.order for unit in catalog.units] == list(range(1, 9))
    by_book = {
        textbook_id: [
            mapping.chapter_number
            for mapping in catalog.chapter_mappings
            if mapping.textbook_id == textbook_id
        ]
        for textbook_id in TextbookId
    }
    assert by_book[TextbookId.PROBABILITY_STATISTICS] == list(range(1, 9))
    assert by_book[TextbookId.PYTHON_DATA_ANALYSIS] == list(range(1, 15))
    assert any(
        mapping.coverage_level == CoverageLevel.DIRECTORY_ONLY
        for mapping in catalog.chapter_mappings
    )


def test_all_eight_units_are_currently_ready_for_teacher_review() -> None:
    catalog = load_default_curriculum_catalog()
    ready_units = {
        unit.unit_id
        for unit in catalog.units
        if unit.content_status == UnitContentStatus.PENDING_TEACHER_REVIEW
    }

    assert ready_units == {
        DeepUnitId.DATA_QUALITY,
        DeepUnitId.DESCRIPTIVE_STATISTICS,
        DeepUnitId.PROBABILITY_SIMULATION,
        DeepUnitId.COMMON_DISTRIBUTIONS,
        DeepUnitId.JOINT_CORRELATION,
        DeepUnitId.SAMPLING_INFERENCE,
        DeepUnitId.ESTIMATION_CONFIDENCE_INTERVAL,
        DeepUnitId.HYPOTHESIS_TESTING,
    }
    assert not any(unit.content_status == UnitContentStatus.APPROVED for unit in catalog.units)


def test_catalog_rejects_missing_chapter(tmp_path: Path) -> None:
    data = _catalog_data()
    mappings = data["chapter_mappings"]
    assert isinstance(mappings, list)
    data["chapter_mappings"] = mappings[:-1]

    with pytest.raises(CurriculumGraphLoadError, match="chapter_mappings"):
        load_curriculum_catalog(_write_catalog(tmp_path, data))


def test_catalog_rejects_unit_prerequisite_cycle(tmp_path: Path) -> None:
    data = _catalog_data()
    units = data["units"]
    assert isinstance(units, list)
    first_unit = units[0]
    assert isinstance(first_unit, dict)
    first_unit["prerequisite_unit_ids"] = ["data_quality"]

    with pytest.raises(CurriculumGraphLoadError, match="不能把自身设为前置"):
        load_curriculum_catalog(_write_catalog(tmp_path, data))


def test_ready_unit_rejects_question_without_hints() -> None:
    catalog = load_default_curriculum_catalog()
    bank = load_default_question_bank()
    questions = list(bank.questions)
    target_index = next(
        index
        for index, question in enumerate(questions)
        if question.id == "data_quality_concept_01"
    )
    questions[target_index] = questions[target_index].model_copy(update={"hints": None})
    incomplete_bank = QuestionBank(
        schema_version=bank.schema_version,
        questions=deepcopy(questions),
    )

    with pytest.raises(CurriculumGraphLoadError, match="缺少四级提示"):
        validate_catalog_question_readiness(catalog, incomplete_bank)
