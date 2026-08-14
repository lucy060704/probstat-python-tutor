from pathlib import Path

from probstat_tutor.config import PROJECT_ROOT
from probstat_tutor.rag import load_rag_manifest, load_rag_source
from scripts.export_adp_knowledge import (
    export_adp_knowledge_bundle,
    render_source_markdown,
)


def test_rendered_markdown_is_beginner_facing_and_traceable() -> None:
    manifest = load_rag_manifest(PROJECT_ROOT / "data" / "rag" / "manifest.yaml")
    source = load_rag_source(manifest.sources[0], PROJECT_ROOT)

    content = render_source_markdown(source)

    assert content.startswith(f"# {source.document.title}\n")
    assert f"资料标识：{source.document.source_id}" in content
    assert "## 学习目标" in content
    assert "## 公式与方法" in content
    assert "## Python联系" in content
    assert "## 常见误区" in content
    assert "correct_answer" not in content
    assert "expected_answer" not in content


def test_export_creates_one_markdown_file_per_manifest_source(tmp_path: Path) -> None:
    manifest = load_rag_manifest(PROJECT_ROOT / "data" / "rag" / "manifest.yaml")

    exported = export_adp_knowledge_bundle(
        project_root=PROJECT_ROOT,
        output_directory=tmp_path,
    )

    assert len(exported) == len(manifest.sources) == 15
    assert {path.name for path in exported} == {
        f"{entry.source_id}.md" for entry in manifest.sources
    }
    assert all(path.stat().st_size > 500 for path in exported)


def test_export_refuses_to_hide_stale_markdown_files(tmp_path: Path) -> None:
    stale_file = tmp_path / "stale.md"
    stale_file.write_text("过期资料", encoding="utf-8")

    try:
        export_adp_knowledge_bundle(
            project_root=PROJECT_ROOT,
            output_directory=tmp_path,
        )
    except RuntimeError as error:
        assert "stale.md" in str(error)
    else:
        raise AssertionError("存在过期Markdown文件时必须要求人工核对")
