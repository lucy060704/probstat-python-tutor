"""G3.2 retrieval-evaluation contract, safety, and metric regressions."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from evals.rag_eval import (
    DEVELOPMENT_DATASET_PATH,
    MINIMUM_HOLDOUT_CASES,
    MINIMUM_HOLDOUT_MATCHED_CASES_PER_UNIT,
    RagEvalCase,
    RagEvalTarget,
    RatioMetric,
    _build_gate_checks,
    _case_has_disclosure_violation,
    _citation_is_integral,
    dataset_fingerprint,
    evaluate_rag_cases,
    load_rag_eval_cases,
    load_rag_eval_freeze_manifest,
    run_dataset,
    validate_frozen_dataset,
    validate_split_independence,
)
from probstat_tutor.config import PROJECT_ROOT
from probstat_tutor.rag import (
    LocalRagIndex,
    RagQuery,
    RagSearchResult,
    RagSection,
    RetrievalPurpose,
    build_local_rag_index,
    infer_minimum_disclosure_level,
)
from probstat_tutor.schemas import ConceptId, DeepUnitId, KnowledgeContextStatus

FROZEN_DEVELOPMENT_FINGERPRINT = (
    "sha256:b84486116e74d50e4e9152ab65af8893c51f56d3bdb53013ffe2c0285f69490a"
)


def _metric(numerator: int, denominator: int) -> RatioMetric:
    return RatioMetric(
        numerator=numerator,
        denominator=denominator,
        value=numerator / denominator,
    )


def _passing_holdout_gate_inputs() -> dict[str, object]:
    perfect = _metric(1, 1)
    return {
        "split": "holdout",
        "case_count": 48,
        "minimum_cases": 48,
        "no_match_count": 8,
        "unit_counts": Counter({unit.value: 5 for unit in DeepUnitId}),
        "minimum_matched_per_unit": 5,
        "covered_concepts": set(ConceptId),
        "source_case_counts": Counter({"source_a": 2, "source_b": 2}),
        "source_ids": ("source_a", "source_b"),
        "query_only_matched": 8,
        "purpose_counts": Counter({purpose.value: 8 for purpose in RetrievalPurpose}),
        "protected_level_one_count": 8,
        "protected_level_one_by_unit": Counter(
            {unit.value: 1 for unit in DeepUnitId}
        ),
        "target_recall_at_1": perfect,
        "target_recall_at_3": perfect,
        "per_unit_target_recall": {unit.value: perfect for unit in DeepUnitId},
        "query_only_target_recall": perfect,
        "mean_reciprocal_rank": 1.0,
        "citation_correctness": perfect,
        "citation_integrity": perfect,
        "no_result_accuracy": perfect,
        "disclosure_rate": _metric(0, 1),
        "level_one_rate": _metric(0, 1),
        "replay_accuracy": perfect,
        "failure_rate": _metric(0, 1),
        "split_independence_checked": True,
    }


@pytest.fixture(scope="module")
def rag_index() -> LocalRagIndex:
    return build_local_rag_index(PROJECT_ROOT)


@pytest.fixture(scope="module")
def development_cases(rag_index: LocalRagIndex) -> tuple[RagEvalCase, ...]:
    return load_rag_eval_cases(
        DEVELOPMENT_DATASET_PATH,
        expected_split="development",
        index=rag_index,
    )


def test_development_dataset_is_frozen_and_balanced(
    development_cases: tuple[RagEvalCase, ...],
    rag_index: LocalRagIndex,
) -> None:
    matched = [case for case in development_cases if case.expected_status == "matched"]
    no_match = [case for case in development_cases if case.expected_status == "no_match"]
    unit_counts = Counter(case.unit_id for case in matched)
    source_counts = Counter(
        target.source_id for case in matched for target in case.required_targets
    )
    concepts = {
        concept for case in matched for concept in case.effective_gold_concept_ids
    }

    assert len(development_cases) == 64
    assert len(matched) == 56
    assert len(no_match) == 8
    assert all(unit_counts[unit] == 7 for unit in DeepUnitId)
    assert concepts == set(ConceptId)
    assert all(source_counts[source_id] >= 2 for source_id in rag_index.source_ids)
    assert sum(case.effective_metadata_mode == "query_only" for case in matched) == 8
    assert sum(case.query_style == "summary_request" for case in matched) == 8
    assert len({case.query_family for case in development_cases}) == 64
    assert Counter(case.purpose for case in development_cases) == {
        RetrievalPurpose.DIAGNOSTIC: 37,
        RetrievalPurpose.HINT: 11,
        RetrievalPurpose.KNOWLEDGE_SEARCH: 16,
    }
    protected_level_one = [
        case
        for case in matched
        if case.disclosure_level == 1
        and (case.forbidden_sections or case.forbidden_output_fragments)
    ]
    assert len(protected_level_one) == 8
    assert {case.unit_id for case in protected_level_one} == set(DeepUnitId)
    assert {case.disclosure_level for case in development_cases} == {1, 2, 3, 4}
    assert dataset_fingerprint(development_cases) == FROZEN_DEVELOPMENT_FINGERPRINT


def test_development_quality_gates_pass() -> None:
    summary = run_dataset()

    assert summary.all_gates_pass
    assert summary.target_recall_at_3.value == 1.0
    assert summary.query_only_target_recall_at_3.value == 1.0
    assert all(
        metric.value == 1.0
        for metric in summary.per_unit_target_recall_at_3.values()
    )
    assert summary.citation_correctness.value >= 0.95
    assert summary.citation_integrity_accuracy.value == 1.0
    assert summary.no_result_accuracy.value == 1.0
    assert summary.disclosure_violation_rate.value == 0.0
    assert summary.level_one_leakage_rate.value == 0.0
    assert summary.replay_stability_accuracy.value == 1.0
    assert summary.evaluation_failure_rate.value == 0.0


def test_freeze_manifest_is_enforced_before_metrics(
    development_cases: tuple[RagEvalCase, ...],
    rag_index: LocalRagIndex,
) -> None:
    manifest = load_rag_eval_freeze_manifest(
        PROJECT_ROOT / "evals" / "rag" / "development_manifest.json"
    )
    validate_frozen_dataset(development_cases, index=rag_index, manifest=manifest)

    stale_updates: tuple[tuple[dict[str, object], str], ...] = (
        ({"cases_sha256": f"sha256:{'0' * 64}"}, "dataset_fingerprint"),
        ({"index_fingerprint": f"sha256:{'0' * 64}"}, "index_fingerprint"),
        (
            {"retriever_fingerprint": f"sha256:{'0' * 64}"},
            "retriever_fingerprint",
        ),
        (
            {"evaluator_fingerprint": f"sha256:{'0' * 64}"},
            "evaluator_fingerprint",
        ),
        (
            {
                "source_snapshot": (
                    manifest.source_snapshot[0].model_copy(
                        update={"checksum": f"sha256:{'0' * 64}"}
                    ),
                    *manifest.source_snapshot[1:],
                )
            },
            "source_snapshot",
        ),
    )
    for update, failed_check in stale_updates:
        stale_manifest = manifest.model_copy(update=update)
        with pytest.raises(ValueError, match=failed_check):
            validate_frozen_dataset(
                development_cases,
                index=rag_index,
                manifest=stale_manifest,
            )


def test_target_pairs_do_not_create_a_cartesian_product() -> None:
    case = RagEvalCase(
        id="rag_dev_target_pairs",
        split="development",
        origin="team_authored",
        query_family="target_pairs",
        query_style="summary_request",
        metadata_mode="query_only",
        query_text="请综合复习中心位置和离散程度。",
        unit_id=DeepUnitId.DESCRIPTIVE_STATISTICS,
        gold_concept_ids=(ConceptId.MEAN_MEDIAN, ConceptId.VARIANCE_STD),
        disclosure_level=4,
        expected_status="matched",
        required_targets=(
            RagEvalTarget(
                source_id="mean_median_core",
                sections=(RagSection.SUMMARY,),
            ),
            RagEvalTarget(
                source_id="variance_std_core",
                sections=(RagSection.CONCEPT_EXPLANATION,),
            ),
        ),
        acceptable_targets=(
            RagEvalTarget(
                source_id="mean_median_core",
                sections=(RagSection.SUMMARY,),
            ),
            RagEvalTarget(
                source_id="variance_std_core",
                sections=(RagSection.CONCEPT_EXPLANATION,),
            ),
        ),
        rationale_zh="验证两个来源的章节标签不会被错误交叉组合。",
    )

    assert case.required_target_pairs == {
        ("mean_median_core", RagSection.SUMMARY),
        ("variance_std_core", RagSection.CONCEPT_EXPLANATION),
    }
    assert ("mean_median_core", RagSection.CONCEPT_EXPLANATION) not in (
        case.required_target_pairs
    )


def test_correct_source_with_wrong_section_fails_formal_recall(
    development_cases: tuple[RagEvalCase, ...],
    rag_index: LocalRagIndex,
) -> None:
    original = next(case for case in development_cases if case.id == "rag_dev_dq_field_rules")
    wrong_section_case = original.model_copy(
        update={
            "required_targets": (
                RagEvalTarget(
                        source_id="data_quality_core",
                        sections=(RagSection.FORMULA_MEANING,),
                ),
            ),
            "acceptable_targets": (
                RagEvalTarget(
                        source_id="data_quality_core",
                        sections=(RagSection.FORMULA_MEANING,),
                ),
            ),
        }
    )

    _observations, summary = evaluate_rag_cases(
        (wrong_section_case,),
        index=rag_index,
    )

    assert summary.source_recall_at_3.value == 1.0
    assert summary.target_recall_at_3.value == 0.0
    assert summary.citation_integrity_accuracy.value == 1.0
    assert summary.citation_correctness.value == 0.0


def test_schema_rejects_unreviewed_or_ambiguous_cases() -> None:
    with pytest.raises(ValidationError, match="required_targets"):
        RagEvalCase(
            id="rag_dev_missing_targets",
            split="development",
            origin="team_authored",
            query_family="missing_targets",
            query_text="为什么要检查缺失数据？",
            unit_id=DeepUnitId.DATA_QUALITY,
            concept_id=ConceptId.DATA_QUALITY,
            knowledge_node_ids=("dq_missing_values",),
            disclosure_level=2,
            expected_status="matched",
            rationale_zh="这是一个缺失严格目标的无效案例。",
        )

    with pytest.raises(ValidationError, match="out_of_domain"):
        RagEvalCase(
            id="rag_dev_bad_no_match_style",
            split="development",
            origin="team_authored",
            query_family="bad_no_match_style",
            query_text="如何配置路由器的动态路由？",
            disclosure_level=1,
            expected_status="no_match",
            rationale_zh="这是课程边界外且类型标注错误的案例。",
        )

    assert MINIMUM_HOLDOUT_CASES == 48
    assert MINIMUM_HOLDOUT_MATCHED_CASES_PER_UNIT == 5


def test_all_holdout_gate_boundaries_are_enforced() -> None:
    passing = _passing_holdout_gate_inputs()
    assert all(_build_gate_checks(**passing).values())  # type: ignore[arg-type]

    scenarios: tuple[tuple[str, object, str], ...] = (
        ("case_count", 47, "minimum_case_count"),
        ("no_match_count", 7, "minimum_no_match_cases"),
        ("query_only_matched", 7, "minimum_query_only_matched_cases"),
        ("protected_level_one_count", 7, "protected_level_one_coverage"),
        ("split_independence_checked", False, "split_independence_checked"),
    )
    for field_name, value, expected_failed_gate in scenarios:
        changed = {**passing, field_name: value}
        checks = _build_gate_checks(**changed)  # type: ignore[arg-type]
        assert not checks[expected_failed_gate]

    low_unit_counts = Counter(passing["unit_counts"])
    low_unit_counts[DeepUnitId.DATA_QUALITY.value] = 4
    checks = _build_gate_checks(  # type: ignore[arg-type]
        **{**passing, "unit_counts": low_unit_counts}
    )
    assert not checks["unit_coverage"]

    incomplete_concepts = set(ConceptId) - {ConceptId.DATA_QUALITY}
    checks = _build_gate_checks(  # type: ignore[arg-type]
        **{**passing, "covered_concepts": incomplete_concepts}
    )
    assert not checks["concept_coverage"]

    low_source_counts = Counter(passing["source_case_counts"])
    low_source_counts["source_a"] = 1
    checks = _build_gate_checks(  # type: ignore[arg-type]
        **{**passing, "source_case_counts": low_source_counts}
    )
    assert not checks["source_positive_coverage"]

    low_purpose_counts = Counter(passing["purpose_counts"])
    low_purpose_counts[RetrievalPurpose.HINT.value] = 7
    checks = _build_gate_checks(  # type: ignore[arg-type]
        **{**passing, "purpose_counts": low_purpose_counts}
    )
    assert not checks["purpose_coverage"]

    per_unit_recall = dict(passing["per_unit_target_recall"])
    per_unit_recall[DeepUnitId.DATA_QUALITY.value] = _metric(3, 5)
    checks = _build_gate_checks(  # type: ignore[arg-type]
        **{
            **passing,
            "target_recall_at_3": _metric(46, 50),
            "per_unit_target_recall": per_unit_recall,
        }
    )
    assert checks["target_recall_at_3_ge_0_90"]
    assert not checks["per_unit_target_recall_at_3_ge_0_80"]


@pytest.mark.parametrize("leak_kind", ["source_id", "near_duplicate"])
def test_loader_rejects_query_leakage(
    tmp_path: Path,
    rag_index: LocalRagIndex,
    development_cases: tuple[RagEvalCase, ...],
    leak_kind: str,
) -> None:
    payload = development_cases[0].model_dump(mode="json")
    payload["id"] = f"rag_dev_leak_{leak_kind}"
    if leak_kind == "source_id":
        payload["query_text"] = "请直接查找 data_quality_core 的内容。"
    else:
        payload["query_text"] = f"{rag_index.chunks[0].text} 请再解释一次。"
    path = tmp_path / f"{leak_kind}.jsonl"
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="source_id|过度相似"):
        load_rag_eval_cases(path, expected_split="development", index=rag_index)


def test_holdout_requires_independent_origin(
    tmp_path: Path,
    rag_index: LocalRagIndex,
    development_cases: tuple[RagEvalCase, ...],
) -> None:
    payload = development_cases[0].model_dump(mode="json")
    payload.update(
        {
            "id": "rag_holdout_wrong_origin",
            "split": "holdout",
            "origin": "team_authored",
        }
    )
    path = tmp_path / "holdout.jsonl"
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="independent_judge"):
        load_rag_eval_cases(path, expected_split="holdout", index=rag_index)


@pytest.mark.parametrize(
    ("changed_fields", "message"),
    [
        ({"query_text": "这是一条完全不同的独立评测查询。"}, "query_family"),
        ({"query_family": "independent_family"}, "重复规范化查询"),
        (
            {
                "query_family": "independent_family",
                "query_text": "表格里的空值、合法零值和异常记录应当怎样区分？请说明。",
            },
            "高度近重复",
        ),
    ],
)
def test_split_independence_rejects_family_exact_and_near_duplicates(
    development_cases: tuple[RagEvalCase, ...],
    changed_fields: dict[str, str],
    message: str,
) -> None:
    source_case = development_cases[0]
    holdout_case = source_case.model_copy(
        update={
            "id": "rag_holdout_independence",
            "split": "holdout",
            "origin": "independent_judge",
            **changed_fields,
        }
    )

    with pytest.raises(ValueError, match=message):
        validate_split_independence(development_cases, (holdout_case,))


def test_quote_and_forbidden_fragment_checks_are_strict(
    development_cases: tuple[RagEvalCase, ...],
    rag_index: LocalRagIndex,
) -> None:
    case = next(case for case in development_cases if case.id == "rag_dev_dq_python_audit")
    result = rag_index.search(
        RagQuery(
            text=case.query_text,
            concept_id=case.concept_id,
            knowledge_node_ids=case.knowledge_node_ids,
            disclosure_level=case.disclosure_level,
        )
    )
    hit = result.hits[0]
    chunks_by_id = {chunk.chunk_id: chunk for chunk in rag_index.chunks}
    assert _citation_is_integral(hit, chunks_by_id)

    tampered_citation = hit.citation.model_copy(update={"quote": "…"})
    tampered_hit = hit.model_copy(update={"citation": tampered_citation})
    assert not _citation_is_integral(tampered_hit, chunks_by_id)

    forbidden_case = case.model_copy(
        update={"forbidden_output_fragments": (hit.content[:12],)}
    )
    assert _case_has_disclosure_violation(forbidden_case, result)


def test_level_one_plain_conclusion_and_forbidden_section_are_detected(
    development_cases: tuple[RagEvalCase, ...],
    rag_index: LocalRagIndex,
) -> None:
    case = next(
        case for case in development_cases if case.id == "rag_dev_ds_outlier_center"
    )
    result = rag_index.search(
        RagQuery(
            text=case.query_text,
            concept_id=case.concept_id,
            knowledge_node_ids=case.knowledge_node_ids,
            purpose=case.purpose,
            disclosure_level=1,
        )
    )
    hit = result.hits[0]
    assert infer_minimum_disclosure_level(hit.content) == 1

    plain_fragment_case = case.model_copy(
        update={"forbidden_output_fragments": (hit.content[:12],)}
    )
    assert _case_has_disclosure_violation(plain_fragment_case, result)

    forbidden_section_case = case.model_copy(
        update={"forbidden_sections": (RagSection(hit.citation.section),)}
    )
    assert _case_has_disclosure_violation(forbidden_section_case, result)


def test_runtime_failure_is_counted_and_not_silenced(
    development_cases: tuple[RagEvalCase, ...],
    rag_index: LocalRagIndex,
) -> None:
    class BrokenIndex:
        chunks = rag_index.chunks
        source_ids = rag_index.source_ids
        index_fingerprint = rag_index.index_fingerprint

        def search(self, _query: RagQuery) -> None:
            raise RuntimeError("deliberate test failure")

    observations, summary = evaluate_rag_cases(
        (development_cases[0],),
        index=cast(LocalRagIndex, BrokenIndex()),
    )

    assert observations[0].failure_zh is not None
    assert not observations[0].replay_stable
    assert summary.evaluation_failure_rate.value == 1.0
    assert summary.replay_stability_accuracy.value == 0.0
    assert not summary.all_gates_pass


def test_different_replay_results_fail_stability_gate(
    development_cases: tuple[RagEvalCase, ...],
    rag_index: LocalRagIndex,
) -> None:
    class AlternatingIndex:
        chunks = rag_index.chunks
        source_ids = rag_index.source_ids
        index_fingerprint = rag_index.index_fingerprint
        calls = 0

        def search(self, query: RagQuery) -> RagSearchResult:
            self.calls += 1
            result = rag_index.search(query)
            if self.calls % 2:
                return result
            return result.model_copy(
                update={
                    "status": KnowledgeContextStatus.NO_MATCH,
                    "hits": (),
                    "message_zh": "测试中的不稳定无结果。",
                }
            )

    observations, summary = evaluate_rag_cases(
        (development_cases[0],),
        index=cast(LocalRagIndex, AlternatingIndex()),
    )

    assert not observations[0].replay_stable
    assert summary.replay_stability_accuracy.value == 0.0
    assert not summary.gate_checks["replay_stability_eq_1"]
