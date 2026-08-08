"""Content, UX, leakage, and AST-only checks for G2.2 Unit 2."""

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
UNIT_ID = DeepUnitId.DESCRIPTIVE_STATISTICS


def _questions() -> tuple[Question, ...]:
    return tuple(
        question
        for question in load_default_question_bank().questions
        if question.unit_id == UNIT_ID
    )


def _question(question_id: str) -> Question:
    return next(question for question in _questions() if question.id == question_id)


def _grade_numeric_question(question: Question, answer: str, code: str):
    submission = LearnerSubmission(answer=answer, python_code=code)
    return combine_submission_evidence(
        question,
        submission,
        grade_numeric(
            answer,
            question.expected_answer,
            absolute_tolerance=question.numeric_tolerance or 0.0,
        ),
    )


def test_unit_has_six_ready_questions_with_four_level_hints() -> None:
    questions = _questions()

    assert len(questions) == 6
    assert {question.question_type for question in questions} == set(QuestionType)
    assert all(question.hints is not None for question in questions)
    assert all(question.knowledge_node_ids for question in questions)
    assert all(
        question.review_status == ContentReviewStatus.PENDING_TEACHER_REVIEW
        for question in questions
    )
    python_questions = {
        question.id for question in questions if question.python_code_required
    }
    assert python_questions == {
        "mean_median_python_01",
        "variance_std_python_01",
    }


def test_chinese_learner_answers_use_controlled_aliases() -> None:
    cases = (
        ("mean_median_concept_01", "中位数"),
        ("mean_median_interpretation_01", "6000元"),
        ("variance_std_concept_01", "B组"),
        ("variance_std_interpretation_01", "A班"),
    )

    for question_id, answer in cases:
        question = _question(question_id)
        result = grade_multiple_choice(
            answer,
            question.expected_answer,
            accepted_answers=question.accepted_answers,
        )

        assert result.is_correct is True


def test_first_two_hints_do_not_reveal_descriptive_answers() -> None:
    forbidden_by_question = {
        "mean_median_concept_01": ("中位数", "median", "5500", "14300"),
        "mean_median_python_01": ("8.0", "(6+10)/2", "结果为 8"),
        "mean_median_interpretation_01": ("中位数更", "应选 6000"),
        "variance_std_concept_01": ("B 组", "B组", "group_b", "8/3"),
        "variance_std_python_01": ("2.0", "√4", "结果为 2"),
        "variance_std_interpretation_01": ("A 班更", "A班更", "group_a"),
    }

    for question_id, forbidden_phrases in forbidden_by_question.items():
        hints = _question(question_id).hints
        assert hints is not None
        early_text = f"{hints.for_level(1)}\n{hints.for_level(2)}"
        assert not any(phrase.casefold() in early_text.casefold() for phrase in forbidden_phrases)
        complete = hints.complete_explanation
        assert all(
            getattr(complete, field).strip()
            for field in ("concept", "calculation", "python", "interpretation")
        )


def test_median_python_requires_exact_reviewed_column_call() -> None:
    question = _question("mean_median_python_01")

    for code in ('df["value"].median()', "df.value.median()"):
        assert _grade_numeric_question(question, "8", code).is_correct is True

    for code in (
        'df["other"].median()',
        'df["value"].median(axis=1)',
        "df.median()",
        'DF["value"].median()',
        "df.Value.median()",
        'df["value"].Median()',
    ):
        result = _grade_numeric_question(question, "8", code)
        assert result.is_correct is False
        assert result.misconception_candidates == ["python_code_conflicts_with_answer"]


def test_sample_std_python_accepts_only_reviewed_ddof_forms() -> None:
    question = _question("variance_std_python_01")

    for code in ("s.std()", "np.std(s, ddof=1)"):
        assert _grade_numeric_question(question, "2", code).is_correct is True

    for code in (
        "s.std(ddof=0)",
        "other.std()",
        "np.std(s, ddof=1, axis=0)",
        "np.std(other, ddof=1)",
        "S.std()",
        "NP.std(s, ddof=1)",
        "np.std(S, ddof=1)",
    ):
        result = _grade_numeric_question(question, "2", code)
        assert result.is_correct is False
        assert result.misconception_candidates == ["python_code_conflicts_with_answer"]


def test_correct_choice_with_reversed_statistical_reasoning_is_rejected() -> None:
    cases = (
        (
            "mean_median_concept_01",
            "中位数",
            "中位数最容易被极端值大幅拉动，所以选中位数。",
            "median_affected_by_outlier",
        ),
        (
            "variance_std_interpretation_01",
            "A班",
            "A班标准差更小，但标准差越小代表数据越分散，所以更稳定。",
            "smaller_std_more_dispersed",
        ),
    )

    for question_id, answer, reasoning, expected_tag in cases:
        question = _question(question_id)
        submission = LearnerSubmission(answer=answer, reasoning=reasoning)
        result = combine_submission_evidence(
            question,
            submission,
            grade_multiple_choice(
                answer,
                question.expected_answer,
                accepted_answers=question.accepted_answers,
            ),
        )

        assert result.is_correct is False
        assert result.misconception_candidates == [expected_tag]


def test_catalog_and_original_sources_are_pending_teacher_review() -> None:
    catalog = load_default_curriculum_catalog()
    unit = next(unit for unit in catalog.units if unit.unit_id == UNIT_ID)

    assert unit.content_status == UnitContentStatus.PENDING_TEACHER_REVIEW
    assert {
        edge.target_node_id
        for edge in catalog.edges
        if edge.source_node_id in {"ds_center", "ds_spread", "ds_outlier_robustness"}
    } >= {"ds_outlier_robustness", "ds_context_interpretation"}

    manifest = load_rag_manifest(ROOT / "data" / "rag" / "manifest.yaml")
    entries = {
        entry.source_id: entry
        for entry in manifest.sources
        if entry.source_id in {"mean_median_core", "variance_std_core"}
    }
    assert set(entries) == {"mean_median_core", "variance_std_core"}
    for entry in entries.values():
        loaded = load_rag_source(entry, ROOT)
        assert loaded.eligibility.eligible_for_chunking is True
        assert entry.metadata["content_status"] == "pending_teacher_review"
        assert entry.metadata["unit_id"] == UNIT_ID.value
        assert entry.metadata["reviewer_role"] == "course_teacher"
