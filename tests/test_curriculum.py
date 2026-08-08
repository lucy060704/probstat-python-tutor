"""Tests for curriculum loading and v0.1 question bank integrity."""

from collections import Counter
from pathlib import Path

import pytest

from probstat_tutor.curriculum import CurriculumLoadError, load_default_question_bank
from probstat_tutor.curriculum import load_question_bank as load_bank
from probstat_tutor.schemas import ConceptId, DeepUnitId, QuestionBank, QuestionType


@pytest.fixture
def question_bank() -> QuestionBank:
    return load_default_question_bank()


def test_question_ids_are_unique(question_bank: QuestionBank) -> None:
    question_ids = [question.id for question in question_bank.questions]

    assert len(question_ids) == 33
    assert len(question_ids) == len(set(question_ids))


def test_difficulty_is_between_zero_and_one(question_bank: QuestionBank) -> None:
    assert all(0.0 <= question.difficulty <= 1.0 for question in question_bank.questions)


def test_dimension_weights_sum_to_one(question_bank: QuestionBank) -> None:
    for question in question_bank.questions:
        weights = question.dimension_weights
        total = weights.concept + weights.calculation + weights.python + weights.interpretation
        assert total == pytest.approx(1.0)


def test_each_published_concept_has_at_least_three_question_types(
    question_bank: QuestionBank,
) -> None:
    concept_counts = Counter(question.concept_id for question in question_bank.questions)

    assert set(concept_counts) == set(ConceptId)
    assert all(count >= 3 for count in concept_counts.values())
    for concept in ConceptId:
        question_types = {
            question.question_type
            for question in question_bank.questions
            if question.concept_id == concept
        }
        assert question_types == set(QuestionType)


def test_data_quality_unit_has_three_structured_four_level_hints(
    question_bank: QuestionBank,
) -> None:
    questions = [
        question
        for question in question_bank.questions
        if question.unit_id == DeepUnitId.DATA_QUALITY
    ]

    assert len(questions) == 3
    assert {question.question_type for question in questions} == set(QuestionType)
    assert all(question.hints is not None for question in questions)
    assert all(question.knowledge_node_ids for question in questions)
    assert all(
        question.hints and question.hints.for_level(4).startswith("完整解释：")
        for question in questions
    )


def test_all_prerequisite_concepts_exist(question_bank: QuestionBank) -> None:
    available_concepts = {question.concept_id for question in question_bank.questions}

    for question in question_bank.questions:
        assert set(question.prerequisites) <= available_concepts


def test_missing_file_has_clear_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(CurriculumLoadError, match="文件不存在"):
        load_bank(missing_path)


def test_invalid_yaml_has_clear_error(tmp_path: Path) -> None:
    invalid_path = tmp_path / "invalid.yaml"
    invalid_path.write_text("questions: [", encoding="utf-8")

    with pytest.raises(CurriculumLoadError, match="YAML 格式错误"):
        load_bank(invalid_path)
