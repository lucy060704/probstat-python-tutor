"""Portable launcher checks without starting a long-running web server."""

import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_macos_launcher_preflight() -> None:
    result = subprocess.run(
        [str(PROJECT_ROOT / "scripts" / "start_macos.command"), "--check"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "启动检查通过" in result.stdout


def test_windows_launcher_has_quoted_paths_and_preflight() -> None:
    content = (PROJECT_ROOT / "scripts" / "start_windows.cmd").read_text(
        encoding="utf-8"
    )

    assert 'set "PROJECT_DIR=%~dp0.."' in content
    assert 'if /I "%~1"=="--check"' in content
    assert '"%PYTHON_BIN%" -m streamlit run app.py' in content


def test_windows_preflight_propagates_import_failure_without_early_expansion() -> None:
    content = (PROJECT_ROOT / "scripts" / "start_windows.cmd").read_text(
        encoding="utf-8"
    )
    preflight_block = content.split('if /I "%~1"=="--check" (', maxsplit=1)[1].split(
        "\n)\n\n", maxsplit=1
    )[0]

    assert "%ERRORLEVEL%" not in preflight_block
    assert "if errorlevel 1 exit /b 1" in preflight_block
    assert "exit /b 0" in preflight_block
    assert preflight_block.index("if errorlevel 1 exit /b 1") < preflight_block.index(
        "exit /b 0"
    )
