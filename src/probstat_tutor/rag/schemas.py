"""Pydantic contracts for versioned, auditable RAG source manifests."""

from datetime import datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Self, TypeAlias

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from probstat_tutor.schemas import ConceptId

SUPPORTED_MANIFEST_VERSION = "1.0"
SOURCE_PATH_PREFIX = "data/rag/sources/"
SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"
MetadataValue: TypeAlias = str | int | float | bool | list[str]


class SourceType(StrEnum):
    """How the project obtained the source material."""

    PROJECT_AUTHORED = "project_authored"
    OPEN_LICENSED = "open_licensed"
    INSTITUTIONAL = "institutional"
    PUBLIC_DOMAIN = "public_domain"


class SourceLanguage(StrEnum):
    """Languages currently supported by the manifest contract."""

    SIMPLIFIED_CHINESE = "zh-CN"
    ENGLISH = "en"
    SIMPLIFIED_CHINESE_AND_ENGLISH = "zh-CN+en"


class SourceLicense(StrEnum):
    """Small, explicit license allow-list for reviewed course sources."""

    PROJECT_OWNED = "project-owned"
    CC_BY_4_0 = "CC-BY-4.0"
    CC_BY_SA_4_0 = "CC-BY-SA-4.0"
    CC0_1_0 = "CC0-1.0"
    PUBLIC_DOMAIN = "public-domain"
    PERMISSION_REQUIRED = "permission-required"


class AllowedUsage(StrEnum):
    """Operations explicitly permitted for one source."""

    RETRIEVAL = "retrieval"
    QUOTATION = "quotation"
    ADAPTATION = "adaptation"
    REDISTRIBUTION = "redistribution"
    EVALUATION = "evaluation"
    AUDIT_ONLY = "audit_only"


class AnswerLeakageRisk(StrEnum):
    """Risk that a source could reveal a hidden answer or worked solution."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    PROHIBITED = "prohibited"


class KnowledgeSource(BaseModel):
    """One versioned source entry; it contains metadata but not course content."""

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(
        min_length=3,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_]*$",
    )
    title: str = Field(min_length=1, max_length=200)
    source_type: SourceType
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    language: SourceLanguage
    concept_ids: list[ConceptId] = Field(min_length=1)
    file_path: str = Field(min_length=1)
    checksum: str = Field(pattern=SHA256_PATTERN)
    license: SourceLicense
    allowed_usage: list[AllowedUsage] = Field(min_length=1)
    answer_leakage_risk: AnswerLeakageRisk
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)
    updated_at: datetime

    @field_validator("concept_ids")
    @classmethod
    def concept_ids_are_unique(cls, value: list[ConceptId]) -> list[ConceptId]:
        if len(value) != len(set(value)):
            raise ValueError("concept_ids 不能重复")
        return value

    @field_validator("file_path")
    @classmethod
    def file_path_is_safe(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("file_path 不能为空")
        if "\\" in normalized:
            raise ValueError("file_path 必须使用正斜杠 /，不能使用 Windows 反斜杠")
        if ":" in normalized:
            raise ValueError("file_path 必须是项目内相对路径，不能包含盘符或 URL")

        posix_path = PurePosixPath(normalized)
        windows_path = PureWindowsPath(normalized)
        if posix_path.is_absolute() or windows_path.is_absolute():
            raise ValueError("file_path 必须是相对于项目根目录的路径，不能使用绝对路径")
        if any(part in {".", ".."} for part in posix_path.parts):
            raise ValueError("file_path 不能包含 . 或 ..，以免发生路径穿越")
        if not normalized.startswith(SOURCE_PATH_PREFIX):
            raise ValueError(
                f"file_path 必须位于 {SOURCE_PATH_PREFIX} 目录下"
            )
        if normalized.endswith("/"):
            raise ValueError("file_path 必须指向资料文件，不能只填写目录")
        return normalized

    @field_validator("allowed_usage")
    @classmethod
    def allowed_usage_is_unique(
        cls,
        value: list[AllowedUsage],
    ) -> list[AllowedUsage]:
        if len(value) != len(set(value)):
            raise ValueError("allowed_usage 不能重复")
        return value

    @field_validator("metadata")
    @classmethod
    def metadata_keys_are_stable(
        cls,
        value: dict[str, MetadataValue],
    ) -> dict[str, MetadataValue]:
        invalid_keys = sorted(
            key
            for key in value
            if not key
            or len(key) > 64
            or not key[0].islower()
            or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in key)
        )
        if invalid_keys:
            raise ValueError(
                "metadata 键必须使用小写字母、数字和下划线："
                f"{', '.join(invalid_keys)}"
            )
        return value

    @field_validator("updated_at")
    @classmethod
    def updated_at_has_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("updated_at 必须包含时区，例如 2026-07-26T09:00:00+08:00")
        return value

    @model_validator(mode="after")
    def prohibited_sources_are_audit_only(self) -> Self:
        if self.answer_leakage_risk == AnswerLeakageRisk.PROHIBITED and set(
            self.allowed_usage
        ) != {AllowedUsage.AUDIT_ONLY}:
            raise ValueError(
                "answer_leakage_risk=prohibited 的资料只能设置 allowed_usage=[audit_only]"
            )
        return self


class RagManifest(BaseModel):
    """The complete source manifest for one supported schema version."""

    model_config = ConfigDict(extra="forbid")

    manifest_version: str
    sources: list[KnowledgeSource] = Field(min_length=1)

    @field_validator("manifest_version")
    @classmethod
    def manifest_version_is_supported(cls, value: str) -> str:
        if value != SUPPORTED_MANIFEST_VERSION:
            raise ValueError(
                f"暂不支持 manifest_version={value!r}，"
                f"当前仅支持 {SUPPORTED_MANIFEST_VERSION}"
            )
        return value

    @model_validator(mode="after")
    def source_references_are_unique(self) -> Self:
        source_ids = [source.source_id for source in self.sources]
        duplicate_ids = _duplicates(source_ids)
        if duplicate_ids:
            raise ValueError(
                f"source_id 不能重复：{', '.join(duplicate_ids)}"
            )

        file_paths = [source.file_path for source in self.sources]
        duplicate_paths = _duplicates(file_paths)
        if duplicate_paths:
            raise ValueError(
                f"同一个 file_path 不能登记多次：{', '.join(duplicate_paths)}"
            )
        return self


class RagManifestError(ValueError):
    """Beginner-readable error raised while loading a RAG manifest."""


def load_rag_manifest(path: str | Path) -> RagManifest:
    """Read YAML safely and validate it as one supported RAG manifest."""

    manifest_path = Path(path)
    try:
        raw_content = manifest_path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise RagManifestError(f"找不到 RAG manifest 文件：{manifest_path}") from error
    except OSError as error:
        raise RagManifestError(
            f"无法读取 RAG manifest 文件 {manifest_path}：{error}"
        ) from error

    try:
        raw_data = yaml.safe_load(raw_content)
    except yaml.YAMLError as error:
        raise RagManifestError(
            f"RAG manifest 不是有效的 YAML：{manifest_path}"
        ) from error

    if not isinstance(raw_data, dict):
        raise RagManifestError("RAG manifest 顶层必须是一个 YAML 对象")

    try:
        return RagManifest.model_validate(raw_data)
    except ValidationError as error:
        details = "；".join(_format_validation_issue(issue) for issue in error.errors())
        raise RagManifestError(f"RAG manifest 校验失败：{details}") from error


def _duplicates(values: list[str]) -> list[str]:
    return sorted({value for value in values if values.count(value) > 1})


def _format_validation_issue(issue: dict[str, object]) -> str:
    location = ".".join(str(part) for part in issue.get("loc", ())) or "manifest"
    issue_type = str(issue.get("type", ""))
    if issue_type == "missing":
        message = "缺少必填字段"
    elif issue_type == "enum":
        message = "值不在允许的枚举范围内"
    elif issue_type == "string_pattern_mismatch":
        message = "格式不正确"
    else:
        message = str(issue.get("msg", "值无效")).removeprefix("Value error, ")
    return f"{location}：{message}"
