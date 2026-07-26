"""Tests for the versioned RAG source manifest data contract."""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from probstat_tutor.rag import (
    AnswerLeakageRisk,
    RagManifestError,
    load_rag_manifest,
)
from probstat_tutor.rag.schemas import SUPPORTED_MANIFEST_VERSION
from probstat_tutor.schemas import ConceptId

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_MANIFEST_PATH = ROOT / "data" / "rag" / "manifest.example.yaml"


def _valid_manifest_data() -> dict[str, object]:
    return {
        "manifest_version": SUPPORTED_MANIFEST_VERSION,
        "sources": [
            {
                "source_id": "example_source",
                "title": "示例资料",
                "source_type": "project_authored",
                "version": "0.1.0",
                "language": "zh-CN",
                "concept_ids": ["mean_median"],
                "file_path": "data/rag/sources/example.yaml",
                "checksum": f"sha256:{'a' * 64}",
                "license": "project-owned",
                "allowed_usage": ["retrieval", "quotation"],
                "answer_leakage_risk": "low",
                "metadata": {
                    "placeholder": True,
                    "content_status": "planned",
                },
                "updated_at": "2026-07-26T09:00:00+08:00",
            }
        ],
    }


def _write_manifest(
    tmp_path: Path,
    data: object,
    *,
    filename: str = "manifest.yaml",
) -> Path:
    path = tmp_path / filename
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def _source(data: dict[str, object]) -> dict[str, object]:
    sources = data["sources"]
    assert isinstance(sources, list)
    source = sources[0]
    assert isinstance(source, dict)
    return source


def test_valid_manifest_can_be_loaded(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path, _valid_manifest_data())

    manifest = load_rag_manifest(path)

    assert manifest.manifest_version == "1.0"
    assert manifest.sources[0].source_id == "example_source"
    assert manifest.sources[0].concept_ids == [ConceptId.MEAN_MEDIAN]
    assert manifest.sources[0].answer_leakage_risk == AnswerLeakageRisk.LOW


def test_duplicate_source_id_fails_with_clear_error(tmp_path: Path) -> None:
    data = _valid_manifest_data()
    duplicate = deepcopy(_source(data))
    duplicate["file_path"] = "data/rag/sources/second.yaml"
    sources = data["sources"]
    assert isinstance(sources, list)
    sources.append(duplicate)

    with pytest.raises(RagManifestError, match="source_id 不能重复"):
        load_rag_manifest(_write_manifest(tmp_path, data))


def test_invalid_concept_id_fails(tmp_path: Path) -> None:
    data = _valid_manifest_data()
    _source(data)["concept_ids"] = ["correlation"]

    with pytest.raises(RagManifestError, match="concept_ids"):
        load_rag_manifest(_write_manifest(tmp_path, data))


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("source_type", "copied_from_web"),
        ("language", "fr"),
        ("license", "unknown-license"),
        ("allowed_usage", ["model_training"]),
        ("answer_leakage_risk", "very_high"),
    ],
)
def test_invalid_enum_values_fail(
    tmp_path: Path,
    field_name: str,
    invalid_value: object,
) -> None:
    data = _valid_manifest_data()
    _source(data)[field_name] = invalid_value

    with pytest.raises(
        RagManifestError,
        match=rf"{field_name}(?:\.0)?：值不在允许的枚举范围内",
    ):
        load_rag_manifest(_write_manifest(tmp_path, data))


def test_missing_required_field_fails_with_beginner_message(tmp_path: Path) -> None:
    data = _valid_manifest_data()
    del _source(data)["title"]

    with pytest.raises(RagManifestError, match="title：缺少必填字段"):
        load_rag_manifest(_write_manifest(tmp_path, data))


def test_invalid_checksum_fails(tmp_path: Path) -> None:
    data = _valid_manifest_data()
    _source(data)["checksum"] = "md5:not-allowed"

    with pytest.raises(RagManifestError, match="checksum：格式不正确"):
        load_rag_manifest(_write_manifest(tmp_path, data))


@pytest.mark.parametrize(
    "file_path",
    [
        "/etc/private/source.yaml",
        "C:/Users/Learner/private/source.yaml",
    ],
)
def test_absolute_file_path_fails(tmp_path: Path, file_path: str) -> None:
    data = _valid_manifest_data()
    _source(data)["file_path"] = file_path

    with pytest.raises(RagManifestError, match="file_path 必须是"):
        load_rag_manifest(_write_manifest(tmp_path, data))


@pytest.mark.parametrize(
    "file_path",
    [
        "../private/source.yaml",
        "data/rag/sources/../../questions.yaml",
    ],
)
def test_path_traversal_fails(tmp_path: Path, file_path: str) -> None:
    data = _valid_manifest_data()
    _source(data)["file_path"] = file_path

    with pytest.raises(RagManifestError, match="路径穿越"):
        load_rag_manifest(_write_manifest(tmp_path, data))


def test_unsupported_manifest_version_fails(tmp_path: Path) -> None:
    data = _valid_manifest_data()
    data["manifest_version"] = "2.0"

    with pytest.raises(RagManifestError, match="暂不支持 manifest_version"):
        load_rag_manifest(_write_manifest(tmp_path, data))


def test_example_manifest_passes_validation_without_course_content() -> None:
    manifest = load_rag_manifest(EXAMPLE_MANIFEST_PATH)

    assert manifest.manifest_version == "1.0"
    assert len(manifest.sources) == 4
    assert {concept for source in manifest.sources for concept in source.concept_ids} == set(
        ConceptId
    )
    assert all(source.metadata["placeholder"] is True for source in manifest.sources)
    assert all(
        source.metadata["contains_course_content"] is False
        for source in manifest.sources
    )


def test_duplicate_file_path_fails(tmp_path: Path) -> None:
    data = _valid_manifest_data()
    duplicate = deepcopy(_source(data))
    duplicate["source_id"] = "second_source"
    sources = data["sources"]
    assert isinstance(sources, list)
    sources.append(duplicate)

    with pytest.raises(RagManifestError, match="同一个 file_path 不能登记多次"):
        load_rag_manifest(_write_manifest(tmp_path, data))


def test_prohibited_source_must_be_audit_only(tmp_path: Path) -> None:
    data = _valid_manifest_data()
    source = _source(data)
    source["answer_leakage_risk"] = "prohibited"
    source["allowed_usage"] = ["retrieval"]

    with pytest.raises(RagManifestError, match="只能设置 allowed_usage"):
        load_rag_manifest(_write_manifest(tmp_path, data))


def test_updated_at_requires_timezone(tmp_path: Path) -> None:
    data = _valid_manifest_data()
    _source(data)["updated_at"] = "2026-07-26T09:00:00"

    with pytest.raises(RagManifestError, match="updated_at 必须包含时区"):
        load_rag_manifest(_write_manifest(tmp_path, data))


def test_missing_manifest_file_has_clear_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(RagManifestError, match="找不到 RAG manifest 文件"):
        load_rag_manifest(missing_path)


def test_invalid_yaml_has_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("sources: [", encoding="utf-8")

    with pytest.raises(RagManifestError, match="不是有效的 YAML"):
        load_rag_manifest(path)


def test_manifest_top_level_must_be_object(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path, ["not", "an", "object"])

    with pytest.raises(RagManifestError, match="顶层必须是一个 YAML 对象"):
        load_rag_manifest(path)
