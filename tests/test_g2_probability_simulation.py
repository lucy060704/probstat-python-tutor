"""Content, AST, leakage, and source checks for G2.3 Unit 3."""

from pathlib import Path

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
    EvidenceVerdict,
    LearnerSubmission,
    Question,
    QuestionType,
)

ROOT = Path(__file__).resolve().parents[1]
UNIT_ID = DeepUnitId.PROBABILITY_SIMULATION


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
            question.expected_answer,
            accepted_answers=question.accepted_answers,
        ),
    )


def test_unit_has_three_ready_questions_and_two_sources() -> None:
    questions = _questions()
    catalog = load_default_curriculum_catalog()
    unit = next(unit for unit in catalog.units if unit.unit_id == UNIT_ID)

    assert len(questions) == 3
    assert {question.question_type for question in questions} == set(QuestionType)
    assert all(question.hints is not None for question in questions)
    assert all(question.knowledge_node_ids for question in questions)
    assert all(question.evidence_policy.reasoning_support_groups for question in questions)
    assert all(
        question.review_status == ContentReviewStatus.PENDING_TEACHER_REVIEW
        for question in questions
    )
    assert {
        question.id for question in questions if question.python_code_required
    } == {"probability_simulation_python_01"}
    assert unit.content_status == UnitContentStatus.PENDING_TEACHER_REVIEW
    assert set(unit.source_ids) == {
        "probability_rules_independence_core",
        "numpy_random_simulation_core",
    }


def test_conditional_probability_requires_correct_denominator_and_relationship() -> None:
    question = _question("probability_simulation_concept_01")
    correct = LearnerSubmission(
        answer="选项A",
        reasoning=(
            "分母是参加 A 的 40 人，10/40=0.25。交集不为 0，"
            "且 0.10 不等于 0.12，所以既不互斥也不独立。"
        ),
    )

    result = _grade_choice(question, correct)

    assert result.is_correct is True
    assert any(finding.verdict == EvidenceVerdict.SUPPORTS for finding in result.findings)

    wrong_cases = (
        ("条件概率是 10/100，分母仍然是全部 100 人。", "conditional_uses_total_denominator"),
        (
            "条件概率是 10/30，分母是参加 B 的 30 人。",
            "conditional_uses_wrong_condition_denominator",
        ),
        ("两个社团名称不同所以互斥。", "mutually_exclusive_despite_overlap"),
        ("有人同时参加所以独立。", "nonexclusive_assumed_independent"),
    )
    for reasoning, expected_tag in wrong_cases:
        wrong = _grade_choice(question, LearnerSubmission(answer="A", reasoning=reasoning))
        assert wrong.is_correct is False
        assert wrong.misconception_candidates == [expected_tag]


def test_probability_reasoning_requires_every_group_and_rejects_reversed_logic() -> None:
    question = _question("probability_simulation_concept_01")
    cases = (
        ("既不互斥也不独立。", "independence_not_checked"),
        (
            "不能说分母是参加 A 的 40 人；交集不为 0，所以不互斥；"
            "0.10 不等于 0.12，所以不独立。",
            "independence_not_checked",
        ),
        (
            "分母是参加 A 的 40 人，所以交集为0，两事件互相排斥并且相互独立。",
            "mutually_exclusive_despite_overlap",
        ),
        (
            "10/40=0.25；交集不为0，所以互斥；0.10不等于0.12所以独立。",
            "mutually_exclusive_despite_overlap",
        ),
        (
            "10/40=0.25；交集不为0，但这证明两个事件相互独立；最后我写不独立。",
            "nonexclusive_assumed_independent",
        ),
        (
            "分母是参加 A 的 40 人这一说法不对；交集不为0，所以不互斥；"
            "0.10不等于0.12，所以不独立。",
            "independence_not_checked",
        ),
        (
            "10/40=0.25；交集不为0所以不互斥；0.10不等于0.12，"
            "但这意味着彼此独立；最后写不独立。",
            "nonexclusive_assumed_independent",
        ),
    )

    for reasoning, expected_tag in cases:
        result = _grade_choice(
            question,
            LearnerSubmission(answer="A", reasoning=reasoning),
        )
        assert result.is_correct is False
        assert expected_tag in result.misconception_candidates


def test_seeded_binomial_python_accepts_two_connected_proportion_forms() -> None:
    question = _question("probability_simulation_python_01")
    codes = (
        (
            "rng = np.random.default_rng(2026)\n"
            "outcomes = rng.binomial(1, 0.5, size=1000)\n"
            "outcomes.mean()"
        ),
        (
            "generator = np.random.default_rng(seed=2026)\n"
            "draws = generator.binomial(n=1, p=0.5, size=1000)\n"
            "count = draws.sum()\n"
            "count / 1000"
        ),
        "np.random.default_rng(2026).binomial(1, 0.5, size=1000).mean()",
        (
            "rng = np.random.default_rng(2026)\n"
            "draws = rng.binomial(1, 0.5, size=1000)\n"
            "np.mean(draws)"
        ),
        (
            "rng = np.random.default_rng(2026)\n"
            "draws = rng.binomial(1, 0.5, size=1000)\n"
            "draws.sum() / len(draws)"
        ),
    )

    for code in codes:
        result = _grade_choice(
            question,
            LearnerSubmission(
                answer="A",
                reasoning="种子用于复现，有限模拟比例存在随机误差。",
                python_code=code,
            ),
        )
        assert result.is_correct is True
        assert result.misconception_candidates == []


def test_binomial_python_rejects_unseeded_and_disconnected_structures() -> None:
    question = _question("probability_simulation_python_01")
    cases = (
        (
            "rng = np.random.default_rng()\n"
            "draws = rng.binomial(1, 0.5, size=1000)\n"
            "draws.mean()",
            "seed_missing_for_reproducibility",
        ),
        (
            "rng = np.random.default_rng(2027)\n"
            "draws = rng.binomial(1, 0.5, size=1000)\n"
            "draws.mean()",
            "probability_simulation_code_conflict",
        ),
        (
            "rng = np.random.default_rng(2026)\n"
            "draws = rng.binomial(1, 0.5, size=1)\n"
            "draws.mean()",
            "probability_simulation_code_conflict",
        ),
        (
            "rng = np.random.default_rng(2026)\n"
            "draws = rng.binomial(1000, 0.5, size=1)\n"
            "draws.mean()",
            "probability_simulation_code_conflict",
        ),
        (
            "rng = np.random.default_rng(2026)\n"
            "draws = rng.binomial(1, 0.5, size=1000)\n"
            "draws.sum()",
            "probability_simulation_code_conflict",
        ),
        (
            "rng = np.random.default_rng(2026)\n"
            "draws = rng.binomial(1, 0.5, size=1000)\n"
            "draws.mean()\n0.5",
            "probability_simulation_code_conflict",
        ),
        (
            "RNG = np.random.default_rng(2026)\n"
            "rng = np.random.default_rng(2027)\n"
            "draws = RNG.binomial(1, 0.5, size=1000)\n"
            "draws.mean()",
            None,
        ),
        (
            "rng = NP.random.default_rng(2026)\n"
            "draws = rng.binomial(1, 0.5, size=1000)\n"
            "draws.mean()",
            "probability_simulation_code_conflict",
        ),
        (
            "rng = np.random.default_rng(2026)\n"
            "draws = rng.binomial(1, 0.5, size=1000)\n"
            "other = [1, 0]\n"
            "draws.sum() / len(other)",
            "probability_simulation_code_conflict",
        ),
        (
            "rng = np.random.default_rng(2026)\n"
            "draws = rng.binomial(1, 0.5, size=1000)\n"
            "other = [1, 0]\n"
            "np.mean(other)",
            "probability_simulation_code_conflict",
        ),
        (
            "rng = np.random.default_rng(2026)\n"
            "draws = rng.binomial(1, 0.5, size=1000)\n"
            "count = draws.sum()\n"
            "draws = [1, 0]\n"
            "count / len(draws)",
            "probability_simulation_code_conflict",
        ),
    )

    for code, expected_tag in cases:
        result = _grade_choice(
            question,
            LearnerSubmission(
                answer="A",
                reasoning="种子用于复现，有限模拟比例存在随机误差。",
                python_code=code,
            ),
        )
        if expected_tag is None:
            assert result.is_correct is True
        else:
            assert result.is_correct is False
            assert result.misconception_candidates == [expected_tag]


def test_python_reasoning_requires_seed_and_finite_simulation_meaning() -> None:
    question = _question("probability_simulation_python_01")
    code = (
        "rng = np.random.default_rng(2026)\n"
        "draws = rng.binomial(1, 0.5, size=1000)\n"
        "draws.mean()"
    )
    cases = (
        ("初始化一次。", "seed_role_unexplained"),
        (
            "种子用于复现，所以模拟比例是没有随机误差的精确值。",
            "seed_guarantees_accuracy",
        ),
        (
            "种子用于复现；有限模拟比例是确定且完全无误差的。",
            "seed_guarantees_accuracy",
        ),
        (
            "模拟比例是估计，但每轮都应重设种子。",
            "seed_reset_each_trial",
        ),
        (
            "种子用于复现这一说法不对；有限模拟比例存在随机误差。",
            "seed_role_unexplained",
        ),
        (
            "种子用于复现；有限模拟比例是固定值，完全不会波动。",
            "simulation_uncertainty_denied",
        ),
        (
            "种子用于复现；实验室的测量仪器存在随机误差。",
            "seed_role_unexplained",
        ),
        (
            "种子用于复现；实验室的测量仪器有随机误差。",
            "seed_role_unexplained",
        ),
        (
            "种子用于复现；这是有限模拟比例。",
            "seed_role_unexplained",
        ),
        (
            "这张图可复现；模拟比例是估计。",
            "seed_role_unexplained",
        ),
    )

    for reasoning, expected_tag in cases:
        result = _grade_choice(
            question,
            LearnerSubmission(answer="A", reasoning=reasoning, python_code=code),
        )
        assert result.is_correct is False
        assert expected_tag in result.misconception_candidates

    correct_variants = (
        "种子用于复现；有限模拟比例不是精确值，而是存在随机误差的估计。",
        "种子用于复现；有限模拟比例不是精确值而是有随机误差的估计。",
    )
    for reasoning in correct_variants:
        result = _grade_choice(
            question,
            LearnerSubmission(answer="A", reasoning=reasoning, python_code=code),
        )
        assert result.is_correct is True


def test_simulation_interpretation_rejects_deterministic_overclaims() -> None:
    question = _question("probability_simulation_interpretation_01")
    correct = LearnerSubmission(
        answer="C",
        reasoning=(
            "两者都是有限模拟比例，理论概率仍为 0.5。"
            "增加次数通常更稳定，但不保证每次更接近。"
        ),
    )
    assert _grade_choice(question, correct).is_correct is True

    wrong_cases = (
        ("58 次正面证明理论概率是 0.58。", "simulation_changes_theoretical_probability"),
        ("0.5041 是精确理论概率。", "simulated_proportion_equals_exact_probability"),
        ("模拟次数增加后每次都一定更接近。", "larger_n_guarantees_monotonic_closeness"),
        ("固定种子后模拟比例必然等于 0.5。", "seed_guarantees_accuracy"),
        ("5041 就是模拟比例。", "count_confused_with_proportion"),
    )
    for reasoning, expected_tag in wrong_cases:
        result = _grade_choice(
            question,
            LearnerSubmission(answer="C", reasoning=reasoning),
        )
        assert result.is_correct is False
        assert result.misconception_candidates == [expected_tag]


def test_interpretation_requires_theory_simulation_and_trend_groups() -> None:
    question = _question("probability_simulation_interpretation_01")
    cases = (
        ("理论概率仍为 0.5。", "simulation_uncertainty_unexplained"),
        (
            "不能说理论概率仍为 0.5；它是有限模拟比例；通常更稳定。",
            "simulation_uncertainty_unexplained",
        ),
        (
            "理论概率仍为 0.5，但有限模拟比例就是精确真概率，并且次数越多保证每次都更接近。",
            "simulated_proportion_equals_exact_probability",
        ),
        (
            "理论概率仍为 0.5；有限模拟比例没有任何误差；次数增加通常更稳定。",
            "simulated_proportion_equals_exact_probability",
        ),
        (
            "理论概率不变；存在随机误差；通常更稳定，但保证每次更接近。",
            "larger_n_guarantees_monotonic_closeness",
        ),
        (
            "存在随机误差，但第一次已经证明公平硬币概率改成0.58。",
            "simulation_changes_theoretical_probability",
        ),
        (
            "存在随机误差，但第一次已经证明公平硬币概率改成了 0.58。",
            "simulation_changes_theoretical_probability",
        ),
        (
            "理论概率仍为 0.5 这一结论不对；有限模拟比例存在随机误差；"
            "通常更稳定。",
            "simulation_uncertainty_unexplained",
        ),
        (
            "理论概率不变；有限模拟比例存在随机误差；通常更稳定，"
            "不过样本越大每次误差都会下降。",
            "larger_n_guarantees_monotonic_closeness",
        ),
        (
            "理论概率不变；实验室的测量仪器存在随机误差；"
            "通常更稳定但不保证每次更接近。",
            "simulation_uncertainty_unexplained",
        ),
        (
            "理论概率不变；实验室的测量仪器有随机误差；"
            "通常更稳定但不保证每次更接近。",
            "simulation_uncertainty_unexplained",
        ),
        (
            "理论概率不变；它不是有限模拟比例；通常更稳定但不保证每次更接近。",
            "simulation_uncertainty_unexplained",
        ),
        (
            "理论概率不变；不知道是不是有限模拟比例；"
            "通常更稳定但不保证每次更接近。",
            "simulation_uncertainty_unexplained",
        ),
    )

    for reasoning, expected_tag in cases:
        result = _grade_choice(
            question,
            LearnerSubmission(answer="C", reasoning=reasoning),
        )
        assert result.is_correct is False
        assert expected_tag in result.misconception_candidates

    natural_spacing = _grade_choice(
        question,
        LearnerSubmission(
            answer="C",
            reasoning=(
                "理论概率仍为0.5；有限模拟比例不是精确理论概率，存在随机误差；"
                "次数增加通常更稳定但不保证每次更接近。"
            ),
        ),
    )
    assert natural_spacing.is_correct is True


def test_first_two_hints_do_not_reveal_probability_answers() -> None:
    forbidden_by_question = {
        "probability_simulation_concept_01": ("0.25", "既不互斥也不独立", "应选 A"),
        "probability_simulation_python_01": ("应选 A", "default_rng(2026)", "size=1000"),
        "probability_simulation_interpretation_01": ("应选 C", "0.58", "0.5041", "0.0041"),
    }
    for question_id, forbidden_phrases in forbidden_by_question.items():
        hints = _question(question_id).hints
        assert hints is not None
        early_text = f"{hints.for_level(1)}\n{hints.for_level(2)}"
        assert not any(phrase in early_text for phrase in forbidden_phrases)


def test_original_probability_sources_are_registered_without_question_leakage() -> None:
    manifest = load_rag_manifest(ROOT / "data" / "rag" / "manifest.yaml")
    entries = {
        entry.source_id: entry
        for entry in manifest.sources
        if entry.metadata.get("unit_id") == UNIT_ID.value
    }

    assert set(entries) == {
        "probability_rules_independence_core",
        "numpy_random_simulation_core",
    }
    for entry in entries.values():
        loaded = load_rag_source(entry, ROOT)
        source_text = (ROOT / entry.file_path).read_text(encoding="utf-8")
        assert loaded.eligibility.eligible_for_chunking is True
        assert entry.metadata["content_status"] == "pending_teacher_review"
        assert entry.metadata["reviewer_role"] == "course_teacher"
        assert "100 名学生" not in source_text
        assert "5041" not in source_text
        assert "2026" not in source_text
