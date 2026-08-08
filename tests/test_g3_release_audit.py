"""G3.6 release audit regression tests."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from scripts.g3_release_audit import (
    EXPECTED_RAG_EVAL_SHA256,
    EXPECTED_RETRIEVAL_SHA256,
    run_release_audit,
)
from scripts.g3_verify_rebuilt_release import audit_snapshots_match
from scripts.run_release_tests import discover_public_test_paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_release_audit_proves_local_engineering_scope(tmp_path: Path) -> None:
    output = tmp_path / "release-audit.json"

    result = run_release_audit(PROJECT_ROOT, output_path=output)

    assert result.engineering_gate_passed is True
    assert result.competition_submission_ready is False
    assert all(check.status == "pass" for check in result.checks)
    assert result.metrics.deep_unit_count == 8
    assert result.metrics.question_count == 33
    assert result.metrics.knowledge_node_count == 33
    assert result.metrics.textbook_chapter_mapping_count == 22
    assert result.metrics.rag_source_count == 15
    assert result.metrics.rag_chunk_count == 478
    assert len(result.release_manifest_sha256) == 64
    assert json.loads(output.read_text(encoding="utf-8")) == result.model_dump(mode="json")


def test_release_manifest_excludes_private_runtime_and_invalid_eval_data() -> None:
    result = run_release_audit(PROJECT_ROOT)
    paths = {artifact.path for artifact in result.release_artifacts}

    assert "data/questions.yaml" in paths
    assert "data/curriculum_catalog.yaml" in paths
    assert "docs/api/openapi.json" in paths
    assert "evals/rag/development.jsonl" in paths
    assert "data/learning_state.sqlite3" not in paths
    assert "data/sessions.sqlite3" not in paths
    assert "data/logs/faults.jsonl" not in paths
    assert "docs/competition/adp_platform_spike.md" not in paths
    assert "docs/competition_first_prize_execution_plan.md" not in paths
    assert "tests/test_eval_datasets.py" not in paths
    assert "docs/competition/g3_rebuilt_release_result.json" not in paths
    assert "docs/competition/g3_judge_verdict.md" not in paths
    assert "scripts/run_release_tests.py" in paths
    assert "scripts/g3_verify_rebuilt_release.py" in paths
    assert not any(path.startswith("docs/competition/evidence/") for path in paths)
    assert not any("/blind/" in f"/{path}/" for path in paths)
    assert not any("holdout" in path.lower() for path in paths)
    assert not any(path.lower().endswith(".pdf") for path in paths)
    assert not any("__pycache__" in path for path in paths)


def test_release_audit_keeps_frozen_g3_2_files_unchanged() -> None:
    import hashlib

    retrieval_hash = hashlib.sha256(
        (PROJECT_ROOT / "src/probstat_tutor/rag/retrieval.py").read_bytes()
    ).hexdigest()
    evaluator_hash = hashlib.sha256(
        (PROJECT_ROOT / "evals/rag_eval.py").read_bytes()
    ).hexdigest()

    assert retrieval_hash == EXPECTED_RETRIEVAL_SHA256
    assert evaluator_hash == EXPECTED_RAG_EVAL_SHA256


def test_pending_human_items_cannot_be_misreported_as_submission_ready() -> None:
    result = run_release_audit(PROJECT_ROOT)
    pending_ids = {item.action_id for item in result.pending_human_actions}

    assert {
        "teacher_signed_review",
        "anonymous_student_pilot",
        "windows11_physical_replay",
        "registration_form",
        "design_specification",
        "five_minute_video",
        "adp_adaptation",
        "final_publish_and_submit",
    } <= pending_ids
    assert all(item.status == "pending_human_action" for item in result.pending_human_actions)
    assert all(item.blocks_competition_submission for item in result.pending_human_actions)
    assert result.competition_submission_ready is False


def test_public_release_test_suite_has_no_invalid_blind_dependency() -> None:
    paths = discover_public_test_paths(PROJECT_ROOT)

    assert "tests/test_eval_datasets.py" not in paths
    assert len(paths) >= 30


def test_allowlist_only_copy_can_collect_every_public_test(tmp_path: Path) -> None:
    result = run_release_audit(PROJECT_ROOT)
    bundle = tmp_path / "release-bundle"
    for artifact in result.release_artifacts:
        source = PROJECT_ROOT / artifact.path
        target = bundle / artifact.path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(bundle / "src"), str(bundle)))
    import_probe = subprocess.run(
        [sys.executable, "-c", "import probstat_tutor; print(probstat_tutor.__file__)"],
        cwd=bundle,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    collection = subprocess.run(
        [sys.executable, "scripts/run_release_tests.py", "--collect-only"],
        cwd=bundle,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert import_probe.returncode == 0, import_probe.stderr
    assert str(bundle / "src") in import_probe.stdout
    assert collection.returncode == 0, collection.stdout + collection.stderr
    assert "collected" in collection.stdout


def test_rebuilt_audit_requires_exact_snapshot_match() -> None:
    result = run_release_audit(PROJECT_ROOT)
    changed = result.model_copy(update={"competition_submission_ready": True})

    assert audit_snapshots_match(result, result.model_copy(deep=True))
    assert not audit_snapshots_match(result, changed)
