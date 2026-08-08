"""Safe, integrity-checking loader for trusted local RAG course sources."""

import hashlib
import re
import unicodedata
from collections.abc import Iterable, Mapping
from enum import StrEnum
from pathlib import Path, PurePosixPath, PureWindowsPath

import yaml
from pydantic import ValidationError

from probstat_tutor.curriculum import load_default_question_bank
from probstat_tutor.rag.schemas import KnowledgeSource
from probstat_tutor.rag.source_schemas import (
    FORBIDDEN_SOURCE_KEYS,
    LoadedRagSource,
    RagSourceDocument,
    assess_chunking_eligibility,
)
from probstat_tutor.schemas import Question

RAG_SOURCE_RELATIVE_DIRECTORY = PurePosixPath("data/rag/sources")
_SOURCE_INJECTION_PATTERNS = (
    (
        "ignore_policy",
        re.compile(
            r"\b(?:ignore|disregard|bypass|override|forget)\b.{0,80}"
            r"\b(?:instruction|instructions|rule|rules|policy|policies)\b",
        ),
    ),
    (
        "reveal_answer",
        re.compile(
            r"\b(?:reveal|print|output|show|disclose|return)\b.{0,80}"
            r"\b(?:answer|answers|solution|solutions)\b",
        ),
    ),
    (
        "override_score",
        re.compile(
            r"\b(?:override|change|modify|set|replace)\b.{0,80}"
            r"\b(?:score|scores|grade|grades|mark|marks)\b",
        ),
    ),
    (
        "ignore_policy",
        re.compile(
            r"(?:无视|忽略|绕过|覆盖|抛开).{0,40}"
            r"(?:指令|规则|政策|策略|要求|系统要求)"
        ),
    ),
    (
        "reveal_answer",
        re.compile(
            r"(?:输出|泄露|显示|打印|透露|给出|回答).{0,40}"
            r"(?:答案|解答|标准答案|隐藏答案|标准解)"
        ),
    ),
    (
        "override_score",
        re.compile(r"(?:覆盖|修改|更改|改成|设为|设置).{0,40}(?:分数|成绩|评分|满分)"),
    ),
)
_LOCAL_NEGATION_PATTERN = re.compile(
    r"(?:不要|不得|不能|禁止|避免|切勿|do\s+not|does\s+not|did\s+not|"
    r"don't|never|must\s+not|should\s+not)"
)
_CLAUSE_BOUNDARY_PATTERN = re.compile(r"[,.;:，。；：！？!?]\s*")


class RagSourceLoadErrorCode(StrEnum):
    """Stable categories for source loading failures."""

    INVALID_MANIFEST_ENTRY = "invalid_manifest_entry"
    PROJECT_ROOT_INVALID = "project_root_invalid"
    SOURCE_DIRECTORY_MISSING = "source_directory_missing"
    PATH_ESCAPE = "path_escape"
    SYMLINK_ESCAPE = "symlink_escape"
    FILE_NOT_FOUND = "file_not_found"
    NOT_REGULAR_FILE = "not_regular_file"
    CHECKSUM_MISMATCH = "checksum_mismatch"
    INVALID_UTF8 = "invalid_utf8"
    INVALID_YAML = "invalid_yaml"
    FORBIDDEN_CONTENT = "forbidden_content"
    SCHEMA_INVALID = "schema_invalid"
    SOURCE_ID_MISMATCH = "source_id_mismatch"
    VERSION_MISMATCH = "version_mismatch"
    CONCEPT_ID_MISMATCH = "concept_id_mismatch"
    LANGUAGE_MISMATCH = "language_mismatch"


class RagSourceLoadError(ValueError):
    """Beginner-readable source error with a stable machine-readable code."""

    def __init__(self, code: RagSourceLoadErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


def load_rag_source(
    manifest_entry: KnowledgeSource | Mapping[str, object],
    project_root: Path,
) -> LoadedRagSource:
    """Load one source only after its real path has passed containment checks."""

    source = _validate_manifest_entry(manifest_entry)
    title_attacks = find_source_instruction_attacks(_normalize_text(source.title))
    if title_attacks:
        raise RagSourceLoadError(
            RagSourceLoadErrorCode.FORBIDDEN_CONTENT,
            "RAG manifest 标题包含试图改变系统行为或泄露答案的指令："
            f"{', '.join(title_attacks)}",
        )
    relative_path = _validate_relative_path(source.file_path)
    root = _resolve_project_root(project_root)
    allowed_root = _resolve_allowed_source_root(root)
    candidate_path = root.joinpath(*relative_path.parts)
    symlink_detected = _contains_symlink_component(candidate_path, allowed_root)
    resolved_candidate = candidate_path.resolve(strict=False)
    validate_resolved_source_path(
        allowed_root=allowed_root,
        resolved_candidate=resolved_candidate,
        symlink_detected=symlink_detected,
        display_path=source.file_path,
    )

    if not resolved_candidate.exists():
        raise RagSourceLoadError(
            RagSourceLoadErrorCode.FILE_NOT_FOUND,
            f"找不到 RAG 课程资料文件：{source.file_path}",
        )
    if not resolved_candidate.is_file():
        raise RagSourceLoadError(
            RagSourceLoadErrorCode.NOT_REGULAR_FILE,
            f"RAG 资料路径不是普通文件：{source.file_path}",
        )

    content_bytes = resolved_candidate.read_bytes()
    actual_checksum = f"sha256:{hashlib.sha256(content_bytes).hexdigest()}"
    if actual_checksum != source.checksum:
        raise RagSourceLoadError(
            RagSourceLoadErrorCode.CHECKSUM_MISMATCH,
            f"RAG 资料 checksum 不一致：{source.file_path}。"
            "文件可能已修改，请重新审核并更新 manifest。",
        )

    try:
        text = content_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RagSourceLoadError(
            RagSourceLoadErrorCode.INVALID_UTF8,
            f"RAG 资料不是有效的 UTF-8 文本：{source.file_path}",
        ) from error

    try:
        raw_document = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise RagSourceLoadError(
            RagSourceLoadErrorCode.INVALID_YAML,
            f"RAG 资料不是有效的 YAML：{source.file_path}",
        ) from error
    if not isinstance(raw_document, dict):
        raise RagSourceLoadError(
            RagSourceLoadErrorCode.SCHEMA_INVALID,
            f"RAG 资料顶层必须是 YAML 对象：{source.file_path}",
        )

    _validate_content_boundaries(raw_document)
    try:
        document = RagSourceDocument.model_validate(raw_document)
    except ValidationError as error:
        details = "；".join(_format_validation_issue(issue) for issue in error.errors())
        raise RagSourceLoadError(
            RagSourceLoadErrorCode.SCHEMA_INVALID,
            f"RAG 资料 schema 校验失败：{details}",
        ) from error

    _validate_document_identity(source, document)
    return LoadedRagSource(
        manifest_entry=source,
        document=document,
        relative_path=source.file_path,
        content_checksum=actual_checksum,
        eligibility=assess_chunking_eligibility(source),
    )


def validate_resolved_source_path(
    *,
    allowed_root: Path,
    resolved_candidate: Path,
    symlink_detected: bool,
    display_path: str,
) -> None:
    """Reject resolved paths outside the dedicated RAG source directory."""

    try:
        resolved_candidate.relative_to(allowed_root)
    except ValueError as error:
        if symlink_detected:
            raise RagSourceLoadError(
                RagSourceLoadErrorCode.SYMLINK_ESCAPE,
                f"RAG 资料路径通过符号链接逃出了 data/rag/sources/：{display_path}",
            ) from error
        raise RagSourceLoadError(
            RagSourceLoadErrorCode.PATH_ESCAPE,
            f"RAG 资料解析后的路径逃出了 data/rag/sources/：{display_path}",
        ) from error


def _validate_manifest_entry(
    entry: KnowledgeSource | Mapping[str, object],
) -> KnowledgeSource:
    raw_entry = entry.model_dump(mode="python") if isinstance(entry, KnowledgeSource) else entry
    raw_file_path = raw_entry.get("file_path")
    if isinstance(raw_file_path, str):
        # 先给路径做安全分类，避免 Pydantic 把路径穿越笼统归为 manifest 无效。
        _validate_relative_path(raw_file_path)
    try:
        return KnowledgeSource.model_validate(raw_entry)
    except ValidationError as error:
        details = "；".join(_format_validation_issue(issue) for issue in error.errors())
        raise RagSourceLoadError(
            RagSourceLoadErrorCode.INVALID_MANIFEST_ENTRY,
            f"RAG manifest entry 无效：{details}",
        ) from error


def _validate_relative_path(file_path: str) -> PurePosixPath:
    if "\\" in file_path or ":" in file_path:
        raise RagSourceLoadError(
            RagSourceLoadErrorCode.PATH_ESCAPE,
            "RAG 资料必须使用项目内相对路径，不能使用盘符、URL 或反斜杠",
        )
    path = PurePosixPath(file_path)
    if path.is_absolute() or PureWindowsPath(file_path).is_absolute():
        raise RagSourceLoadError(
            RagSourceLoadErrorCode.PATH_ESCAPE,
            "RAG 资料不能使用绝对路径",
        )
    if any(part in {".", ".."} for part in path.parts):
        raise RagSourceLoadError(
            RagSourceLoadErrorCode.PATH_ESCAPE,
            "RAG 资料路径不能包含 . 或 ..",
        )
    expected_prefix = RAG_SOURCE_RELATIVE_DIRECTORY.parts
    if path.parts[: len(expected_prefix)] != expected_prefix:
        raise RagSourceLoadError(
            RagSourceLoadErrorCode.PATH_ESCAPE,
            "RAG 资料只能位于 data/rag/sources/ 目录",
        )
    return path


def _resolve_project_root(project_root: Path) -> Path:
    try:
        root = Path(project_root).resolve(strict=True)
    except OSError as error:
        raise RagSourceLoadError(
            RagSourceLoadErrorCode.PROJECT_ROOT_INVALID,
            "项目根目录不存在或无法访问",
        ) from error
    if not root.is_dir():
        raise RagSourceLoadError(
            RagSourceLoadErrorCode.PROJECT_ROOT_INVALID,
            "项目根路径不是目录",
        )
    return root


def _resolve_allowed_source_root(project_root: Path) -> Path:
    source_root = project_root.joinpath(*RAG_SOURCE_RELATIVE_DIRECTORY.parts)
    if _contains_symlink_component(source_root, project_root):
        raise RagSourceLoadError(
            RagSourceLoadErrorCode.SYMLINK_ESCAPE,
            "受限 RAG 资料目录 data/rag/sources/ 不能由符号链接替代",
        )
    try:
        resolved = source_root.resolve(strict=True)
    except OSError as error:
        raise RagSourceLoadError(
            RagSourceLoadErrorCode.SOURCE_DIRECTORY_MISSING,
            "找不到受限 RAG 资料目录 data/rag/sources/",
        ) from error
    if not resolved.is_dir():
        raise RagSourceLoadError(
            RagSourceLoadErrorCode.SOURCE_DIRECTORY_MISSING,
            "RAG 资料根路径 data/rag/sources/ 不是目录",
        )
    return resolved


def _contains_symlink_component(candidate: Path, allowed_root: Path) -> bool:
    try:
        relative_parts = candidate.relative_to(allowed_root).parts
    except ValueError:
        return False
    current = allowed_root
    for part in relative_parts:
        current /= part
        is_junction = getattr(current, "is_junction", lambda: False)
        if current.is_symlink() or is_junction():
            return True
    return False


def _validate_content_boundaries(raw_document: dict[str, object]) -> None:
    forbidden_keys = sorted(_find_forbidden_keys(raw_document))
    if forbidden_keys:
        raise RagSourceLoadError(
            RagSourceLoadErrorCode.FORBIDDEN_CONTENT,
            "RAG 课程资料包含禁止的题库或评测内部字段："
            f"{', '.join(forbidden_keys)}",
        )

    all_text = _normalize_text("\n".join(_iter_text_values(raw_document)))
    forbidden_instructions = find_source_instruction_attacks(all_text)
    if forbidden_instructions:
        raise RagSourceLoadError(
            RagSourceLoadErrorCode.FORBIDDEN_CONTENT,
            "RAG 课程资料包含试图改变系统行为或泄露答案的指令："
            f"{', '.join(forbidden_instructions)}",
        )

    normalized_texts = {
        _normalize_text(text)
        for text in _iter_text_values(raw_document)
        if text.strip()
    }
    questions = load_default_question_bank().questions
    copied_question_ids = sorted(
        question.id
        for question in questions
        if _normalize_text(question.prompt) in normalized_texts
    )
    if copied_question_ids:
        raise RagSourceLoadError(
            RagSourceLoadErrorCode.FORBIDDEN_CONTENT,
            "RAG 课程资料复制了完整题干，涉及题目："
            f"{', '.join(copied_question_ids)}",
        )

    embedded_internal_ids = sorted(
        question.id
        for question in questions
        if _normalize_text(question.id) in all_text
    )
    if embedded_internal_ids:
        raise RagSourceLoadError(
            RagSourceLoadErrorCode.FORBIDDEN_CONTENT,
            "RAG 课程资料包含正式题内部 ID，涉及题目："
            f"{', '.join(embedded_internal_ids)}",
        )

    copied_long_answer_ids = sorted(
        question.id
        for question in questions
        if normalized_texts & _protected_long_answer_texts(question)
    )
    if copied_long_answer_ids:
        raise RagSourceLoadError(
            RagSourceLoadErrorCode.FORBIDDEN_CONTENT,
            "RAG 课程资料复制了正式题的长答案或完整解释，涉及题目："
            f"{', '.join(copied_long_answer_ids)}",
        )


def _protected_long_answer_texts(question: Question) -> set[str]:
    protected: list[str] = []
    expected_answer = question.expected_answer
    if isinstance(expected_answer, str):
        protected.append(expected_answer)
    hints = question.hints
    if hints is not None:
        explanation = hints.complete_explanation
        protected.extend(
            (
                explanation.concept,
                explanation.calculation,
                explanation.python,
                explanation.interpretation,
                explanation.render_zh(),
            )
        )
    return {
        _normalize_text(text)
        for text in protected
        if len(_normalize_text(text)) >= 80
    }


def _find_forbidden_keys(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            if key_text in FORBIDDEN_SOURCE_KEYS or key_text.startswith("expected_"):
                found.add(key_text)
            found.update(_find_forbidden_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_find_forbidden_keys(child))
    return found


def find_source_instruction_attacks(normalized_text: str) -> tuple[str, ...]:
    found: set[str] = set()
    for category, pattern in _SOURCE_INJECTION_PATTERNS:
        for match in pattern.finditer(normalized_text):
            prefix = normalized_text[max(0, match.start() - 100) : match.start()]
            local_clause = _CLAUSE_BOUNDARY_PATTERN.split(prefix)[-1]
            if _LOCAL_NEGATION_PATTERN.search(local_clause):
                continue
            found.add(category)
            break
    return tuple(sorted(found))


def _iter_text_values(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _iter_text_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_text_values(child)


def _normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def _validate_document_identity(
    source: KnowledgeSource,
    document: RagSourceDocument,
) -> None:
    if document.source_id != source.source_id:
        raise RagSourceLoadError(
            RagSourceLoadErrorCode.SOURCE_ID_MISMATCH,
            f"正文 source_id 与 manifest 不一致：{source.file_path}",
        )
    if document.version != source.version:
        raise RagSourceLoadError(
            RagSourceLoadErrorCode.VERSION_MISMATCH,
            f"正文 version 与 manifest 不一致：{source.file_path}",
        )
    if set(source.concept_ids) != {document.concept_id}:
        raise RagSourceLoadError(
            RagSourceLoadErrorCode.CONCEPT_ID_MISMATCH,
            f"正文 concept_id 与 manifest concept_ids 不一致：{source.file_path}",
        )
    if document.language != source.language:
        raise RagSourceLoadError(
            RagSourceLoadErrorCode.LANGUAGE_MISMATCH,
            f"正文 language 与 manifest 不一致：{source.file_path}",
        )


def _format_validation_issue(issue: dict[str, object]) -> str:
    location = ".".join(str(part) for part in issue.get("loc", ())) or "document"
    issue_type = str(issue.get("type", ""))
    if issue_type == "missing":
        message = "缺少必填字段"
    elif issue_type == "enum":
        message = "值不在允许的枚举范围内"
    elif issue_type == "extra_forbidden":
        message = "包含未定义字段"
    elif issue_type == "string_pattern_mismatch":
        message = "格式不正确"
    else:
        message = str(issue.get("msg", "值无效")).removeprefix("Value error, ")
    return f"{location}：{message}"
