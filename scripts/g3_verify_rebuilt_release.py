"""Rebuild the allowlisted release in a temporary directory and verify it offline."""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from probstat_tutor.config import PROJECT_ROOT

if __package__:
    from scripts.g3_release_audit import AUDIT_RESULT_PATH, G36ReleaseAudit
else:
    from g3_release_audit import AUDIT_RESULT_PATH, G36ReleaseAudit

DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "docs/competition/g3_rebuilt_release_result.json"


class RebuiltCommandResult(BaseModel):
    """Sanitized result of one command inside the rebuilt allowlist tree."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    return_code: int
    duration_seconds: float = Field(ge=0.0)
    output_tail: tuple[str, ...]
    passed: bool


class RebuiltReleaseSummary(BaseModel):
    """Machine evidence that the release list is executable without private data."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "1.0.0"
    transport: str = "temporary_allowlist_copy_verified_environment_no_external_network"
    python_version: str
    operating_system: str
    source_manifest_sha256: str
    copied_artifact_count: int = Field(ge=1)
    blind_or_holdout_artifact_count: int = Field(ge=0)
    commands: tuple[RebuiltCommandResult, ...]
    all_passed: bool


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tail(text: str, *, bundle_root: Path, source_root: Path) -> tuple[str, ...]:
    sanitized = text
    replacements = (
        (str(bundle_root.resolve()), "<release_bundle>"),
        (str(bundle_root), "<release_bundle>"),
        (str(source_root.resolve()), "<source_project>"),
        (str(source_root), "<source_project>"),
        (str(Path.home()), "<user_home>"),
    )
    for original, replacement in replacements:
        sanitized = sanitized.replace(original, replacement)
    return tuple(sanitized.splitlines()[-12:])


def _run(
    name: str,
    command: list[str],
    *,
    bundle_root: Path,
    source_root: Path,
    environment: dict[str, str],
    timeout_seconds: int = 360,
) -> RebuiltCommandResult:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=bundle_root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        output = f"{completed.stdout}\n{completed.stderr}".strip()
        return_code = completed.returncode
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout.decode() if isinstance(error.stdout, bytes) else error.stdout or ""
        stderr = error.stderr.decode() if isinstance(error.stderr, bytes) else error.stderr or ""
        output = f"{stdout}\n{stderr}\n命令超时".strip()
        return_code = 124
    duration = round(time.perf_counter() - started, 3)
    return RebuiltCommandResult(
        name=name,
        return_code=return_code,
        duration_seconds=duration,
        output_tail=_tail(output, bundle_root=bundle_root, source_root=source_root),
        passed=return_code == 0,
    )


def _copy_allowlist(
    source_root: Path,
    bundle_root: Path,
    audit: G36ReleaseAudit,
) -> None:
    for artifact in audit.release_artifacts:
        source = source_root / artifact.path
        if _sha256(source) != artifact.sha256:
            raise RuntimeError(f"源文件校验失败：{artifact.path}")
        target = bundle_root / artifact.path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if _sha256(target) != artifact.sha256:
            raise RuntimeError(f"复制后校验失败：{artifact.path}")
    metadata_target = bundle_root / AUDIT_RESULT_PATH
    metadata_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_root / AUDIT_RESULT_PATH, metadata_target)


def audit_snapshots_match(expected: G36ReleaseAudit, actual: G36ReleaseAudit) -> bool:
    """Require the rebuilt tree to reproduce the complete source audit exactly."""

    return expected == actual


def verify_rebuilt_release(
    source_root: Path = PROJECT_ROOT,
    *,
    output_path: Path | None = DEFAULT_OUTPUT_PATH,
) -> RebuiltReleaseSummary:
    """Execute the declared public checks inside an integrity-checked copy."""

    root = source_root.resolve()
    audit = G36ReleaseAudit.model_validate_json(
        (root / AUDIT_RESULT_PATH).read_text(encoding="utf-8")
    )
    private_count = sum(
        "blind" in Path(item.path).parts or "holdout" in Path(item.path).parts
        for item in audit.release_artifacts
    )
    results: list[RebuiltCommandResult] = []
    with tempfile.TemporaryDirectory(prefix="probstat-g3-release-") as directory:
        bundle = Path(directory) / "probstat-python-tutor"
        bundle.mkdir()
        _copy_allowlist(root, bundle, audit)

        environment = os.environ.copy()
        environment.pop("OPENAI_API_KEY", None)
        environment.pop("OPENAI_MODEL", None)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        local_python_path = os.pathsep.join((str(bundle / "src"), str(bundle)))
        existing_python_path = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            f"{local_python_path}{os.pathsep}{existing_python_path}"
            if existing_python_path
            else local_python_path
        )
        # Do not resolve the interpreter symlink: resolving it would jump from
        # the active project venv to the base Python installation.
        source_venv = Path(sys.executable).absolute().parents[1]
        if not (source_venv / "pyvenv.cfg").is_file():
            raise RuntimeError("重建验证必须从项目虚拟环境运行")
        (bundle / ".venv").symlink_to(source_venv, target_is_directory=True)
        venv_python = bundle / ".venv/bin/python"
        probe = _run(
            "allowlist_import_probe",
            [
                str(venv_python),
                "-c",
                "import pathlib, probstat_tutor; "
                "path=pathlib.Path(probstat_tutor.__file__).resolve(); "
                "root=pathlib.Path.cwd().resolve(); "
                "assert path.is_relative_to(root / 'src'), path; print(path)",
            ],
            bundle_root=bundle,
            source_root=root,
            environment=environment,
        )
        results.append(probe)
        if probe.passed:
            commands = (
                ("g1_demo", [str(venv_python), "scripts/g1_offline_demo.py"]),
                (
                    "rag_demo",
                    [
                        str(venv_python),
                        "scripts/g3_local_rag_demo.py",
                        "均值 中位数 异常值",
                        "--concept",
                        "mean_median",
                    ],
                ),
                (
                    "rag_public_development",
                    [str(venv_python), "-m", "evals.rag_eval", "--split", "development"],
                ),
                (
                    "product_demo",
                    [
                        str(venv_python),
                        "scripts/g3_product_demo.py",
                        "--output",
                        ".verification/g3_3_demo_result.json",
                    ],
                ),
                (
                    "api_demo",
                    [
                        str(venv_python),
                        "scripts/g3_api_contract_demo.py",
                        "--output",
                        ".verification/g3_4_api_contract_result.json",
                    ],
                ),
                (
                    "reliability_demo",
                    [
                        str(venv_python),
                        "scripts/g3_reliability_demo.py",
                        "--output",
                        ".verification/g3_5_reliability_result.json",
                    ],
                ),
                ("macos_preflight", ["scripts/start_macos.command", "--check"]),
                (
                    "openapi_contract",
                    [str(venv_python), "scripts/export_api_contract.py", "--check"],
                ),
                ("public_pytest", [str(venv_python), "scripts/run_release_tests.py"]),
                ("ruff", [str(venv_python), "-m", "ruff", "check", "."]),
                ("pip_check", [str(venv_python), "-m", "pip", "check"]),
                (
                    "release_audit",
                    [
                        str(venv_python),
                        "scripts/g3_release_audit.py",
                        "--output",
                        AUDIT_RESULT_PATH,
                    ],
                ),
            )
            for name, command in commands:
                result = _run(
                    name,
                    command,
                    bundle_root=bundle,
                    source_root=root,
                    environment=environment,
                )
                results.append(result)
                if not result.passed:
                    break

            if all(result.passed for result in results):
                rebuilt_audit = G36ReleaseAudit.model_validate_json(
                    (bundle / AUDIT_RESULT_PATH).read_text(encoding="utf-8")
                )
                matched = audit_snapshots_match(audit, rebuilt_audit)
                results.append(
                    RebuiltCommandResult(
                        name="rebuilt_audit_exact_match",
                        return_code=0 if matched else 1,
                        duration_seconds=0.0,
                        output_tail=(
                            "重建审计与源审计逐字段完全一致"
                            if matched
                            else "重建审计与源审计不一致",
                        ),
                        passed=matched,
                    )
                )

    summary = RebuiltReleaseSummary(
        python_version=platform.python_version(),
        operating_system=platform.system(),
        source_manifest_sha256=audit.release_manifest_sha256,
        copied_artifact_count=len(audit.release_artifacts),
        blind_or_holdout_artifact_count=private_count,
        commands=tuple(results),
        all_passed=(
            private_count == 0
            and len(results) == 14
            and all(result.passed for result in results)
        ),
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(f"{summary.model_dump_json(indent=2)}\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    summary = verify_rebuilt_release(output_path=args.output)
    print(summary.model_dump_json(indent=2))
    return 0 if summary.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
