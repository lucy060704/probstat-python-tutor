"""Build the deterministic G3.6 local-engineering release audit.

The audit is deliberately offline. It does not start ADP, contact a model, run
learner code, or treat pending human evidence as if it already existed.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from probstat_tutor.config import PROJECT_ROOT
from probstat_tutor.curriculum import load_question_bank
from probstat_tutor.curriculum_graph import (
    TextbookId,
    UnitContentStatus,
    load_curriculum_catalog,
)
from probstat_tutor.rag import build_local_rag_index
from probstat_tutor.schemas import ContentReviewStatus, QuestionType

EXPECTED_RETRIEVAL_SHA256 = (
    "5e0d2f01c151204cdfe7d3e31c3754f201ae6f9fae4d550c8431e8dca9805371"
)
EXPECTED_RAG_EVAL_SHA256 = (
    "9a59205a1a1adf7ae2fbf29e1790ab8acc46ea24d7e0f20113fbea07f0d5cd7a"
)
EXPECTED_RAG_INDEX_FINGERPRINT = (
    "sha256:5b301d0c26f277080db8ca92a8036cf0e011eead0020a5860cd42779e3c3343e"
)

CORE_RELEASE_FILES = (
    ".env.example",
    ".gitignore",
    "AGENTS.md",
    "README.md",
    "app.py",
    "pyproject.toml",
)
RELEASE_GLOBS = (
    "src/**/*.py",
    "scripts/*.py",
    "scripts/*.command",
    "scripts/*.cmd",
    "data/questions.yaml",
    "data/curriculum_catalog.yaml",
    "data/rag/manifest.yaml",
    "data/rag/manifest.example.yaml",
    "data/rag/sources/*.yaml",
    "data/synthetic/*",
    "docs/product_spec.md",
    "docs/rag_manifest_spec.md",
    "docs/rag_eval_spec.md",
    "docs/api_contract.md",
    "docs/reliability.md",
    "docs/learner_code_sandbox.md",
    "docs/api/*.json",
    "docs/competition/g*.md",
    "docs/competition/g*.json",
    "docs/competition/assets/g3_3/*",
    "docs/competition/adp_upload/code/*.py",
    "evals/*.py",
    "evals/cases.jsonl",
    "evals/development/*.jsonl",
    "evals/rag/development.jsonl",
    "evals/rag/development_manifest.json",
    "evals/baselines/*.json",
    "tests/**/*.py",
)
AUDIT_RESULT_PATH = "docs/competition/g3_release_audit_result.json"
REBUILT_RESULT_PATH = "docs/competition/g3_rebuilt_release_result.json"
JUDGE_VERDICT_PATH = "docs/competition/g3_judge_verdict.md"
EXCLUDED_RELEASE_FILES = {
    AUDIT_RESULT_PATH,
    REBUILT_RESULT_PATH,
    JUDGE_VERDICT_PATH,
    # This legacy repository-only test reads the invalidated blind labels. The
    # public bundle keeps the isolation implementation but never ships/loads it.
    "tests/test_eval_datasets.py",
}
FORBIDDEN_RELEASE_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".pytest_run_cache",
    ".pytest_run_tmp",
    ".ruff_cache",
    "blind",
    "holdout",
    "private_sources",
    "normalized_private",
}
FORBIDDEN_RELEASE_SUFFIXES = {
    ".db",
    ".log",
    ".pdf",
    ".pyc",
    ".sqlite",
    ".sqlite3",
}
TEXT_SUFFIXES = {
    ".cmd",
    ".command",
    ".csv",
    ".example",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
SECRET_PATTERNS = {
    "private_key": re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
    ),
    "openai_style_key": re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    "tencent_secret_id": re.compile(r"AKID[A-Za-z0-9]{16,}"),
}
ADP_RUNTIME_MARKERS = (
    "101.42.184.216",
    "adp.cloud",
    "gaoxiaobang",
    "tengxun.gaoxiaobang",
)

REQUIRED_PHASE_EVIDENCE = (
    "docs/competition/g1_exit_report.md",
    *(f"docs/competition/g2_{number}_exit_report.md" for number in range(1, 9)),
    *(f"docs/competition/g2_{number}_teacher_review_form.md" for number in range(1, 9)),
    "docs/competition/g3_1_exit_report.md",
    "docs/competition/g3_2_exit_report.md",
    "docs/competition/g3_3_exit_report.md",
    "docs/competition/g3_3_demo_result.json",
    "docs/competition/g3_4_exit_report.md",
    "docs/competition/g3_4_api_contract_result.json",
    "docs/competition/g3_5_exit_report.md",
    "docs/competition/g3_5_reliability_result.json",
    "docs/api/openapi.json",
)


class AuditCheck(BaseModel):
    """One broad claim and the concrete evidence used to decide it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    check_id: str
    status: Literal["pass", "fail"]
    evidence: str


class CurriculumMetrics(BaseModel):
    """Counts loaded through the same validated domain loaders as the app."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    deep_unit_count: int = Field(ge=0)
    question_count: int = Field(ge=0)
    knowledge_node_count: int = Field(ge=0)
    knowledge_edge_count: int = Field(ge=0)
    textbook_chapter_mapping_count: int = Field(ge=0)
    rag_source_count: int = Field(ge=0)
    rag_chunk_count: int = Field(ge=0)
    rag_index_fingerprint: str


class ReleaseArtifact(BaseModel):
    """One allowlisted release file with a reproducible integrity digest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class PendingHumanAction(BaseModel):
    """Evidence that Codex cannot honestly manufacture."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action_id: str
    title_zh: str
    status: Literal["pending_human_action"] = "pending_human_action"
    blocks_competition_submission: bool = True


class G36ReleaseAudit(BaseModel):
    """Machine-readable G1–G3 local-engineering completion boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "1.0.0"
    scope: Literal["g1_g3_local_engineering"] = "g1_g3_local_engineering"
    metrics: CurriculumMetrics
    checks: tuple[AuditCheck, ...]
    release_artifacts: tuple[ReleaseArtifact, ...]
    release_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    verification_commands: tuple[str, ...]
    pending_human_actions: tuple[PendingHumanAction, ...]
    engineering_gate_passed: bool
    competition_submission_ready: bool


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check(check_id: str, passed: bool, evidence: str) -> AuditCheck:
    return AuditCheck(
        check_id=check_id,
        status="pass" if passed else "fail",
        evidence=evidence,
    )


def _collect_release_paths(project_root: Path) -> tuple[Path, ...]:
    candidates = {project_root / relative for relative in CORE_RELEASE_FILES}
    for pattern in RELEASE_GLOBS:
        candidates.update(path for path in project_root.glob(pattern) if path.is_file())
    candidates.difference_update(project_root / relative for relative in EXCLUDED_RELEASE_FILES)
    return tuple(sorted(candidates, key=lambda path: path.relative_to(project_root).as_posix()))


def _release_artifacts(project_root: Path) -> tuple[ReleaseArtifact, ...]:
    artifacts: list[ReleaseArtifact] = []
    for path in _collect_release_paths(project_root):
        relative = path.relative_to(project_root).as_posix()
        artifacts.append(
            ReleaseArtifact(
                path=relative,
                size_bytes=path.stat().st_size,
                sha256=_sha256(path),
            )
        )
    return tuple(artifacts)


def _manifest_sha256(artifacts: tuple[ReleaseArtifact, ...]) -> str:
    payload = json.dumps(
        [artifact.model_dump(mode="json") for artifact in artifacts],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _unsafe_runtime_calls(project_root: Path) -> tuple[str, ...]:
    findings: list[str] = []
    runtime_paths = (project_root / "app.py", *sorted((project_root / "src").rglob("*.py")))
    for path in runtime_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec", "compile"}:
                findings.append(f"{path.relative_to(project_root)}:{node.lineno}:{node.func.id}")
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "system"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "os"
            ):
                findings.append(f"{path.relative_to(project_root)}:{node.lineno}:os.system")
    return tuple(findings)


def _adp_runtime_references(project_root: Path) -> tuple[str, ...]:
    findings: list[str] = []
    paths = [project_root / "app.py", project_root / "pyproject.toml"]
    paths.extend(sorted((project_root / "src").rglob("*.py")))
    paths.extend(sorted((project_root / "scripts").glob("*")))
    for path in paths:
        if path.name == "g3_release_audit.py":
            continue
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        content = path.read_text(encoding="utf-8").lower()
        for marker in ADP_RUNTIME_MARKERS:
            if marker in content:
                findings.append(f"{path.relative_to(project_root)}:{marker}")
    return tuple(findings)


def _release_policy_violations(
    project_root: Path,
    artifacts: tuple[ReleaseArtifact, ...],
) -> tuple[str, ...]:
    findings: list[str] = []
    for artifact in artifacts:
        path = Path(artifact.path)
        if set(path.parts) & FORBIDDEN_RELEASE_PARTS:
            findings.append(f"forbidden_path:{artifact.path}")
        if path.suffix.lower() in FORBIDDEN_RELEASE_SUFFIXES:
            findings.append(f"forbidden_suffix:{artifact.path}")
        if path.name == ".env":
            findings.append(f"secret_env_file:{artifact.path}")
        absolute = project_root / path
        if absolute.is_symlink():
            findings.append(f"symlink:{artifact.path}")
    return tuple(findings)


def _secret_findings(
    project_root: Path,
    artifacts: tuple[ReleaseArtifact, ...],
) -> tuple[str, ...]:
    findings: list[str] = []
    for artifact in artifacts:
        path = project_root / artifact.path
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        content = path.read_text(encoding="utf-8")
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(content):
                findings.append(f"{name}:{artifact.path}")
    return tuple(findings)


def _demo_artifacts_pass(project_root: Path) -> tuple[bool, str]:
    paths = (
        "docs/competition/g3_3_demo_result.json",
        "docs/competition/g3_4_api_contract_result.json",
        "docs/competition/g3_5_reliability_result.json",
    )
    statuses: list[str] = []
    all_passed = True
    for relative in paths:
        path = project_root / relative
        try:
            passed = json.loads(path.read_text(encoding="utf-8")).get("passed") is True
        except (OSError, json.JSONDecodeError):
            passed = False
        statuses.append(f"{relative}={'pass' if passed else 'fail'}")
        all_passed = all_passed and passed
    return all_passed, "；".join(statuses)


def run_release_audit(
    project_root: Path = PROJECT_ROOT,
    *,
    output_path: Path | None = None,
) -> G36ReleaseAudit:
    """Audit the current tree and optionally save the stable JSON result."""

    root = project_root.resolve()
    question_bank = load_question_bank(root / "data/questions.yaml")
    catalog = load_curriculum_catalog(
        root / "data/curriculum_catalog.yaml",
        question_bank=question_bank,
    )
    rag_index = build_local_rag_index(root)
    artifacts = _release_artifacts(root)

    per_unit_counts = Counter(question.unit_id for question in question_bank.questions)
    per_unit_types: dict[object, set[QuestionType]] = {
        unit.unit_id: {
            question.question_type
            for question in question_bank.questions
            if question.unit_id == unit.unit_id
        }
        for unit in catalog.units
    }
    chapter_counts = Counter(mapping.textbook_id for mapping in catalog.chapter_mappings)
    all_pending_review = all(
        unit.content_status == UnitContentStatus.PENDING_TEACHER_REVIEW
        for unit in catalog.units
    ) and all(
        question.review_status == ContentReviewStatus.PENDING_TEACHER_REVIEW
        for question in question_bank.questions
        if question.unit_id is not None
    )

    retrieval_hash = _sha256(root / "src/probstat_tutor/rag/retrieval.py")
    evaluator_hash = _sha256(root / "evals/rag_eval.py")
    missing_evidence = tuple(
        relative for relative in REQUIRED_PHASE_EVIDENCE if not (root / relative).is_file()
    )
    unsafe_calls = _unsafe_runtime_calls(root)
    adp_references = _adp_runtime_references(root)
    policy_violations = _release_policy_violations(root, artifacts)
    secret_findings = _secret_findings(root, artifacts)
    demos_passed, demo_evidence = _demo_artifacts_pass(root)
    screenshot_count = len(tuple((root / "docs/competition/assets/g3_3").glob("*.jpg")))

    checks = (
        _check(
            "g2_eight_deep_units",
            len(catalog.units) == 8,
            f"Pydantic 课程目录加载成功，深度单元={len(catalog.units)}",
        ),
        _check(
            "g2_verifiable_question_bank",
            len(question_bank.questions) >= 24
            and all(per_unit_counts[unit.unit_id] >= 3 for unit in catalog.units)
            and all(types == set(QuestionType) for types in per_unit_types.values()),
            "题目总数="
            f"{len(question_bank.questions)}；每单元题数="
            + ",".join(
                f"{unit.unit_id.value}:{per_unit_counts[unit.unit_id]}"
                for unit in catalog.units
            )
            + "；每单元均覆盖 concept/python/interpretation",
        ),
        _check(
            "g2_knowledge_graph_and_textbook_directory",
            len(catalog.knowledge_nodes) >= 8
            and len(catalog.edges) >= 1
            and chapter_counts[TextbookId.PROBABILITY_STATISTICS] == 8
            and chapter_counts[TextbookId.PYTHON_DATA_ANALYSIS] == 14,
            f"知识节点={len(catalog.knowledge_nodes)}，边={len(catalog.edges)}，"
            "教材目录映射=8+14 章（目录目标，不宣称全部深度实现）",
        ),
        _check(
            "teacher_review_status_not_overclaimed",
            all_pending_review,
            "8 个单元及其正式题目保持 pending_teacher_review，未伪造教师批准",
        ),
        _check(
            "g3_local_rag_rebuild",
            len(rag_index.source_ids) == 15
            and len(rag_index.chunks) == 478
            and rag_index.index_fingerprint == EXPECTED_RAG_INDEX_FINGERPRINT,
            f"原创来源={len(rag_index.source_ids)}，切片={len(rag_index.chunks)}，"
            f"fingerprint={rag_index.index_fingerprint}",
        ),
        _check(
            "g3_2_frozen_retriever_and_evaluator",
            retrieval_hash == EXPECTED_RETRIEVAL_SHA256
            and evaluator_hash == EXPECTED_RAG_EVAL_SHA256,
            f"retrieval.py={retrieval_hash}；rag_eval.py={evaluator_hash}",
        ),
        _check(
            "g3_public_demo_artifacts",
            demos_passed,
            demo_evidence,
        ),
        _check(
            "g3_student_teacher_visual_evidence",
            screenshot_count >= 7,
            f"G3.3 学生/教师旅程截图={screenshot_count}",
        ),
        _check(
            "g1_g3_phase_evidence_complete",
            not missing_evidence,
            "阶段证据齐全" if not missing_evidence else f"缺失：{','.join(missing_evidence)}",
        ),
        _check(
            "learner_code_never_executed",
            not unsafe_calls,
            "运行时代码未发现 eval/exec/compile/os.system 调用"
            if not unsafe_calls
            else f"发现：{','.join(unsafe_calls)}",
        ),
        _check(
            "platform_independent_runtime",
            not adp_references,
            "app/src/scripts/pyproject 未绑定 ADP 或赛事域名"
            if not adp_references
            else f"发现：{','.join(adp_references)}",
        ),
        _check(
            "release_allowlist_excludes_private_state_and_blind_data",
            not policy_violations,
            "交付清单排除 PDF、SQLite、日志、缓存、private、blind 与 holdout"
            if not policy_violations
            else f"发现：{','.join(policy_violations)}",
        ),
        _check(
            "release_text_has_no_key_material",
            not secret_findings,
            "交付文本未匹配私钥、OpenAI 风格密钥或腾讯 SecretId"
            if not secret_findings
            else f"发现：{','.join(secret_findings)}",
        ),
    )
    pending = (
        PendingHumanAction(action_id="teacher_signed_review", title_zh="教师逐单元签字审核"),
        PendingHumanAction(
            action_id="teacher_permission_archive",
            title_zh="归档教师教材使用许可证明",
        ),
        PendingHumanAction(
            action_id="anonymous_student_pilot",
            title_zh="10–15 名学生知情同意匿名试点",
        ),
        PendingHumanAction(
            action_id="windows11_physical_replay",
            title_zh="Windows 11 实机双击启动与完整旅程复验",
        ),
        PendingHumanAction(action_id="registration_form", title_zh="填写并复核参赛申报表"),
        PendingHumanAction(action_id="design_specification", title_zh="完成比赛版设计说明书"),
        PendingHumanAction(action_id="five_minute_video", title_zh="录制并复核不超过 5 分钟视频"),
        PendingHumanAction(
            action_id="adp_adaptation",
            title_zh="本地工程冻结后再做赛事平台适配",
        ),
        PendingHumanAction(
            action_id="final_publish_and_submit",
            title_zh="用户确认正式发布与最终提交",
        ),
    )
    engineering_gate_passed = all(check.status == "pass" for check in checks)
    result = G36ReleaseAudit(
        metrics=CurriculumMetrics(
            deep_unit_count=len(catalog.units),
            question_count=len(question_bank.questions),
            knowledge_node_count=len(catalog.knowledge_nodes),
            knowledge_edge_count=len(catalog.edges),
            textbook_chapter_mapping_count=len(catalog.chapter_mappings),
            rag_source_count=len(rag_index.source_ids),
            rag_chunk_count=len(rag_index.chunks),
            rag_index_fingerprint=rag_index.index_fingerprint,
        ),
        checks=checks,
        release_artifacts=artifacts,
        release_manifest_sha256=_manifest_sha256(artifacts),
        verification_commands=(
            ".venv/bin/python scripts/g3_local_rag_demo.py "
            "'均值 中位数 异常值' --concept mean_median",
            ".venv/bin/python -m evals.rag_eval --split development",
            ".venv/bin/python scripts/g3_product_demo.py "
            "--output docs/competition/g3_3_demo_result.json",
            ".venv/bin/python scripts/g3_api_contract_demo.py "
            "--output docs/competition/g3_4_api_contract_result.json",
            ".venv/bin/python scripts/g3_reliability_demo.py "
            "--output docs/competition/g3_5_reliability_result.json",
            ".venv/bin/python scripts/run_release_tests.py",
            "scripts/start_macos.command --check",
            ".venv/bin/python -m ruff check .",
            ".venv/bin/python scripts/g3_verify_rebuilt_release.py "
            "--output docs/competition/g3_rebuilt_release_result.json",
        ),
        pending_human_actions=pending,
        engineering_gate_passed=engineering_gate_passed,
        competition_submission_ready=False,
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(f"{result.model_dump_json(indent=2)}\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / AUDIT_RESULT_PATH,
    )
    args = parser.parse_args()
    result = run_release_audit(output_path=args.output)
    print(result.model_dump_json(indent=2))
    return 0 if result.engineering_gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
