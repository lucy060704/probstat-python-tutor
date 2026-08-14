"""Export reviewed local RAG sources as Tencent ADP-friendly Markdown files."""

from __future__ import annotations

import argparse
from pathlib import Path

from probstat_tutor.config import PROJECT_ROOT
from probstat_tutor.rag import LoadedRagSource, load_rag_manifest, load_rag_source

DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT / "docs" / "competition" / "adp_upload" / "knowledge"
)


def _bullet_list(items: list[str]) -> list[str]:
    return [f"- {item.strip()}" for item in items]


def render_source_markdown(source: LoadedRagSource) -> str:
    """Render one validated, retrieval-eligible source as readable Markdown."""

    if not source.eligibility.eligible_for_chunking:
        reasons = "；".join(source.eligibility.rejection_reasons)
        raise ValueError(f"资料 {source.document.source_id} 不允许导出：{reasons}")

    document = source.document
    lines = [
        f"# {document.title}",
        "",
        (
            f"> 团队原创课程资料｜资料标识：{document.source_id}｜"
            f"版本：{document.version}｜知识点：{document.concept_id.value}"
        ),
        "",
        "## 学习目标",
        "",
        *_bullet_list(document.learning_objectives),
        "",
        "## 前置知识",
        "",
        *_bullet_list(document.prerequisite_knowledge),
        "",
        "## 概念解释",
        "",
    ]
    for paragraph in document.concept_explanation:
        lines.extend((paragraph.strip(), ""))

    lines.extend(("## 公式与方法", ""))
    for formula in document.formula_explanation:
        lines.extend(
            (
                f"### {formula.name}",
                "",
                f"- 表达式：`{formula.expression}`",
                f"- 含义：{formula.meaning}",
                "- 符号说明：",
            )
        )
        lines.extend(f"  - `{symbol}`：{meaning}" for symbol, meaning in formula.symbols.items())
        lines.append("- 使用前提：")
        lines.extend(f"  - {item}" for item in formula.assumptions)
        lines.append("- 注意事项：")
        lines.extend(f"  - {item}" for item in formula.cautions)
        lines.append("")

    lines.extend(("## Python联系", ""))
    for connection in document.python_connection:
        lines.extend(
            (
                f"### {connection.library}：{connection.api}",
                "",
                f"- 用途：{connection.purpose}",
                f"- 输入要求：{connection.input_expectation}",
                f"- 解释提醒：{connection.interpretation_caution}",
                "",
            )
        )

    lines.extend(
        (
            "## 数据解释原则",
            "",
            *_bullet_list(document.data_interpretation_guidance),
            "",
            "## 常见误区",
            "",
        )
    )
    for index, misconception in enumerate(document.common_misconceptions, start=1):
        lines.extend(
            (
                f"### 误区{index}：{misconception.misconception}",
                "",
                f"- 为什么不正确：{misconception.why_incorrect}",
                f"- 更好的追问：{misconception.better_question}",
                "",
            )
        )

    lines.extend(
        (
            "## 反思问题",
            "",
            *_bullet_list(document.reflective_questions),
            "",
            "## 小结",
            "",
            *_bullet_list(document.summary),
            "",
        )
    )
    return "\n".join(lines)


def export_adp_knowledge_bundle(
    project_root: Path = PROJECT_ROOT,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
) -> tuple[Path, ...]:
    """Validate all formal sources and export one Markdown file per source."""

    manifest = load_rag_manifest(project_root / "data" / "rag" / "manifest.yaml")
    output_directory.mkdir(parents=True, exist_ok=True)

    exported: list[Path] = []
    expected_names: set[str] = set()
    for entry in manifest.sources:
        loaded = load_rag_source(entry, project_root)
        output_path = output_directory / f"{loaded.document.source_id}.md"
        expected_names.add(output_path.name)
        output_path.write_text(render_source_markdown(loaded), encoding="utf-8")
        exported.append(output_path)

    stale_files = [
        path for path in output_directory.glob("*.md") if path.name not in expected_names
    ]
    if stale_files:
        names = "、".join(path.name for path in stale_files)
        raise RuntimeError(f"导出目录包含过期文件，请人工核对后移除：{names}")

    return tuple(exported)


def main() -> None:
    parser = argparse.ArgumentParser(description="导出腾讯ADP知识库Markdown上传包")
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Markdown输出目录",
    )
    arguments = parser.parse_args()
    paths = export_adp_knowledge_bundle(output_directory=arguments.output_directory)
    print(f"已导出 {len(paths)} 份原创知识卡到 {arguments.output_directory}")


if __name__ == "__main__":
    main()
