"""Run the public, self-contained pytest suite without blind/holdout data."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_REPOSITORY_ONLY_TESTS = {"tests/test_eval_datasets.py"}
FORBIDDEN_PUBLIC_DATA_MARKERS = (
    "evals/blind/",
    "evals\\blind\\",
    "cases_v0.2_blind.jsonl",
)


def discover_public_test_paths(project_root: Path = PROJECT_ROOT) -> tuple[str, ...]:
    """Return release-safe tests and reject accidental private-data dependencies."""

    root = project_root.resolve()
    paths = tuple(
        path.relative_to(root).as_posix()
        for path in sorted((root / "tests").glob("test_*.py"))
        if path.relative_to(root).as_posix() not in EXCLUDED_REPOSITORY_ONLY_TESTS
    )
    if not paths:
        raise RuntimeError("公开交付包中没有发现 pytest 测试")
    unsafe: list[str] = []
    for relative in paths:
        content = (root / relative).read_text(encoding="utf-8")
        if any(marker in content for marker in FORBIDDEN_PUBLIC_DATA_MARKERS):
            unsafe.append(relative)
    if unsafe:
        raise RuntimeError(f"公开测试仍引用 blind 数据：{','.join(unsafe)}")
    return paths


def run_public_tests(
    project_root: Path = PROJECT_ROOT,
    *,
    collect_only: bool = False,
) -> int:
    """Execute only allowlisted public tests with imports pinned to this tree."""

    root = project_root.resolve()
    paths = discover_public_test_paths(root)
    command = [sys.executable, "-m", "pytest", "-q"]
    if collect_only:
        command.append("--collect-only")
    command.extend(paths)
    environment = os.environ.copy()
    local_python_path = os.pathsep.join((str(root / "src"), str(root)))
    existing_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{local_python_path}{os.pathsep}{existing_python_path}"
        if existing_python_path
        else local_python_path
    )
    completed = subprocess.run(
        command,
        cwd=root,
        env=environment,
        check=False,
    )
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="只验证公开测试可收集，不执行测试正文",
    )
    args = parser.parse_args()
    return run_public_tests(collect_only=args.collect_only)


if __name__ == "__main__":
    raise SystemExit(main())
