"""Tests for safe loading and integrity checks of RAG course sources."""

import hashlib
import json
from collections.abc import Callable, Iterator
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from probstat_tutor.curriculum import load_default_question_bank
from probstat_tutor.rag import (
    AllowedUsage,
    AnswerLeakageRisk,
    EligibilityRejectionCode,
    RagSourceDocument,
    RagSourceLoadError,
    RagSourceLoadErrorCode,
    assess_chunking_eligibility,
    load_rag_manifest,
    load_rag_source,
    validate_resolved_source_path,
)
from probstat_tutor.rag.source_schemas import FORBIDDEN_SOURCE_KEYS
from probstat_tutor.schemas import ConceptId

ROOT = Path(__file__).resolve().parents[1]
FORMAL_MANIFEST_PATH = ROOT / "data" / "rag" / "manifest.yaml"
SOURCE_DIRECTORY = ROOT / "data" / "rag" / "sources"
EVAL_DIRECTORIES = (
    ROOT / "evals" / "development",
    ROOT / "evals" / "blind",
)


def _formal_manifest():
    return load_rag_manifest(FORMAL_MANIFEST_PATH)


def _mean_source_entry_data() -> dict[str, object]:
    entry = next(
        source
        for source in _formal_manifest().sources
        if source.source_id == "mean_median_core"
    )
    return entry.model_dump(mode="json")


def _mean_document_data() -> dict[str, object]:
    path = SOURCE_DIRECTORY / "mean_median.yaml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def _write_temporary_source(
    tmp_path: Path,
    *,
    mutate: Callable[[dict[str, object]], None] | None = None,
    raw_bytes: bytes | None = None,
    entry_updates: dict[str, object] | None = None,
) -> tuple[Path, dict[str, object]]:
    project_root = tmp_path / "project"
    source_path = project_root / "data" / "rag" / "sources" / "source.yaml"
    source_path.parent.mkdir(parents=True)

    if raw_bytes is None:
        document = deepcopy(_mean_document_data())
        if mutate is not None:
            mutate(document)
        raw_bytes = yaml.safe_dump(
            document,
            allow_unicode=True,
            sort_keys=False,
        ).encode("utf-8")
    source_path.write_bytes(raw_bytes)

    entry = _mean_source_entry_data()
    entry.update(
        {
            "file_path": "data/rag/sources/source.yaml",
            "checksum": f"sha256:{hashlib.sha256(raw_bytes).hexdigest()}",
        }
    )
    if entry_updates:
        entry.update(entry_updates)
    return project_root, entry


def _walk_keys(value: object) -> Iterator[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def _walk_text(value: object) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _walk_text(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_text(child)


def _normalized(value: str) -> str:
    return " ".join(value.split()).casefold()


def _assert_error(
    error_info: pytest.ExceptionInfo[RagSourceLoadError],
    code: RagSourceLoadErrorCode,
    message_part: str,
) -> None:
    assert error_info.value.code == code
    assert message_part in str(error_info.value)


@pytest.mark.parametrize("entry", _formal_manifest().sources, ids=lambda item: item.source_id)
def test_four_course_sources_pass_document_schema(entry) -> None:
    source_path = ROOT.joinpath(*Path(entry.file_path).parts)
    raw_document = yaml.safe_load(source_path.read_text(encoding="utf-8"))

    document = RagSourceDocument.model_validate(raw_document)

    assert document.source_id == entry.source_id
    assert document.version == entry.version
    assert {document.concept_id} == set(entry.concept_ids)


@pytest.mark.parametrize("entry", _formal_manifest().sources, ids=lambda item: item.source_id)
def test_four_course_sources_load_safely_and_are_eligible(entry) -> None:
    loaded = load_rag_source(entry, ROOT)

    assert loaded.content_checksum == entry.checksum
    assert loaded.document.source_id == entry.source_id
    assert loaded.eligibility.eligible_for_chunking is True
    assert loaded.eligibility.rejection_reasons == []


def test_formal_manifest_registers_four_unique_sources_and_concepts() -> None:
    manifest = _formal_manifest()

    assert len(manifest.sources) == 4
    assert len({entry.source_id for entry in manifest.sources}) == 4
    assert {concept for entry in manifest.sources for concept in entry.concept_ids} == set(
        ConceptId
    )
    assert all(ROOT.joinpath(*Path(entry.file_path).parts).is_file() for entry in manifest.sources)


@pytest.mark.parametrize(
    ("field_name", "changed_value", "expected_code", "message_part"),
    [
        (
            "source_id",
            "different_source",
            RagSourceLoadErrorCode.SOURCE_ID_MISMATCH,
            "source_id",
        ),
        (
            "version",
            "9.9.9",
            RagSourceLoadErrorCode.VERSION_MISMATCH,
            "version",
        ),
        (
            "concept_id",
            "variance_std",
            RagSourceLoadErrorCode.CONCEPT_ID_MISMATCH,
            "concept_id",
        ),
    ],
)
def test_manifest_and_document_identity_mismatch_fails(
    tmp_path: Path,
    field_name: str,
    changed_value: str,
    expected_code: RagSourceLoadErrorCode,
    message_part: str,
) -> None:
    def mutate(document: dict[str, object]) -> None:
        document[field_name] = changed_value

    project_root, entry = _write_temporary_source(tmp_path, mutate=mutate)

    with pytest.raises(RagSourceLoadError) as error_info:
        load_rag_source(entry, project_root)

    _assert_error(error_info, expected_code, message_part)


def test_missing_file_fails_clearly(tmp_path: Path) -> None:
    source_root = tmp_path / "data" / "rag" / "sources"
    source_root.mkdir(parents=True)
    entry = _mean_source_entry_data()
    entry["file_path"] = "data/rag/sources/missing.yaml"

    with pytest.raises(RagSourceLoadError) as error_info:
        load_rag_source(entry, tmp_path)

    _assert_error(error_info, RagSourceLoadErrorCode.FILE_NOT_FOUND, "找不到")


def test_wrong_checksum_fails_clearly(tmp_path: Path) -> None:
    project_root, entry = _write_temporary_source(tmp_path)
    entry["checksum"] = f"sha256:{'0' * 64}"

    with pytest.raises(RagSourceLoadError) as error_info:
        load_rag_source(entry, project_root)

    _assert_error(error_info, RagSourceLoadErrorCode.CHECKSUM_MISMATCH, "checksum")


def test_modified_file_bytes_invalidate_old_checksum(tmp_path: Path) -> None:
    project_root, entry = _write_temporary_source(tmp_path)
    source_path = project_root / "data" / "rag" / "sources" / "source.yaml"
    source_path.write_bytes(source_path.read_bytes() + b"\n")

    with pytest.raises(RagSourceLoadError) as error_info:
        load_rag_source(entry, project_root)

    _assert_error(error_info, RagSourceLoadErrorCode.CHECKSUM_MISMATCH, "checksum")


@pytest.mark.parametrize(
    "file_path",
    [
        "C:/Users/Learner/private.yaml",
        "/private/source.yaml",
        "../private/source.yaml",
        "data/rag/sources/../../questions.yaml",
    ],
)
def test_absolute_and_traversal_paths_are_rejected_before_reading(
    tmp_path: Path,
    file_path: str,
) -> None:
    entry = _mean_source_entry_data()
    entry["file_path"] = file_path

    with pytest.raises(RagSourceLoadError) as error_info:
        load_rag_source(entry, tmp_path)

    _assert_error(error_info, RagSourceLoadErrorCode.PATH_ESCAPE, "路径")


def test_resolved_path_outside_dedicated_directory_is_rejected(tmp_path: Path) -> None:
    allowed_root = (tmp_path / "data" / "rag" / "sources").resolve()
    outside = (tmp_path / "outside.yaml").resolve()

    with pytest.raises(RagSourceLoadError) as error_info:
        validate_resolved_source_path(
            allowed_root=allowed_root,
            resolved_candidate=outside,
            symlink_detected=False,
            display_path="data/rag/sources/source.yaml",
        )

    _assert_error(error_info, RagSourceLoadErrorCode.PATH_ESCAPE, "解析后的路径")


def test_symlink_escape_classification_without_admin_privileges(tmp_path: Path) -> None:
    allowed_root = (tmp_path / "data" / "rag" / "sources").resolve()
    outside = (tmp_path / "outside.yaml").resolve()

    with pytest.raises(RagSourceLoadError) as error_info:
        validate_resolved_source_path(
            allowed_root=allowed_root,
            resolved_candidate=outside,
            symlink_detected=True,
            display_path="data/rag/sources/link.yaml",
        )

    _assert_error(error_info, RagSourceLoadErrorCode.SYMLINK_ESCAPE, "符号链接")


def test_directory_is_rejected_as_non_regular_file(tmp_path: Path) -> None:
    source_path = tmp_path / "data" / "rag" / "sources" / "directory"
    source_path.mkdir(parents=True)
    entry = _mean_source_entry_data()
    entry["file_path"] = "data/rag/sources/directory"

    with pytest.raises(RagSourceLoadError) as error_info:
        load_rag_source(entry, tmp_path)

    _assert_error(error_info, RagSourceLoadErrorCode.NOT_REGULAR_FILE, "不是普通文件")


def test_invalid_yaml_has_clear_source_error(tmp_path: Path) -> None:
    project_root, entry = _write_temporary_source(
        tmp_path,
        raw_bytes=b"source_id: [\n",
    )

    with pytest.raises(RagSourceLoadError) as error_info:
        load_rag_source(entry, project_root)

    _assert_error(error_info, RagSourceLoadErrorCode.INVALID_YAML, "不是有效的 YAML")


def test_invalid_document_schema_has_clear_source_error(tmp_path: Path) -> None:
    def mutate(document: dict[str, object]) -> None:
        del document["summary"]

    project_root, entry = _write_temporary_source(tmp_path, mutate=mutate)

    with pytest.raises(RagSourceLoadError) as error_info:
        load_rag_source(entry, project_root)

    _assert_error(error_info, RagSourceLoadErrorCode.SCHEMA_INVALID, "summary")


def test_forbidden_question_bank_field_is_rejected(tmp_path: Path) -> None:
    def mutate(document: dict[str, object]) -> None:
        document["expected_answer"] = 42

    project_root, entry = _write_temporary_source(tmp_path, mutate=mutate)

    with pytest.raises(RagSourceLoadError) as error_info:
        load_rag_source(entry, project_root)

    _assert_error(error_info, RagSourceLoadErrorCode.FORBIDDEN_CONTENT, "expected_answer")


def test_retrieval_not_allowed_returns_ineligible_result_after_loading() -> None:
    entry = next(
        source
        for source in _formal_manifest().sources
        if source.source_id == "mean_median_core"
    )
    restricted = entry.model_copy(update={"allowed_usage": [AllowedUsage.EVALUATION]})

    loaded = load_rag_source(restricted, ROOT)

    assert loaded.eligibility.eligible_for_chunking is False
    assert loaded.eligibility.rejection_codes == [
        EligibilityRejectionCode.RETRIEVAL_NOT_ALLOWED
    ]
    assert "allowed_usage" in loaded.eligibility.rejection_reasons[0]


@pytest.mark.parametrize(
    ("risk", "usage", "expected_code"),
    [
        (
            AnswerLeakageRisk.HIGH,
            [AllowedUsage.RETRIEVAL],
            EligibilityRejectionCode.ANSWER_LEAKAGE_RISK_HIGH,
        ),
        (
            AnswerLeakageRisk.PROHIBITED,
            [AllowedUsage.AUDIT_ONLY],
            EligibilityRejectionCode.ANSWER_LEAKAGE_RISK_PROHIBITED,
        ),
    ],
)
def test_excessive_leakage_risk_returns_ineligible_result_after_loading(
    risk: AnswerLeakageRisk,
    usage: list[AllowedUsage],
    expected_code: EligibilityRejectionCode,
) -> None:
    entry = next(
        source
        for source in _formal_manifest().sources
        if source.source_id == "mean_median_core"
    )
    restricted = entry.model_copy(
        update={
            "answer_leakage_risk": risk,
            "allowed_usage": usage,
        }
    )

    loaded = load_rag_source(restricted, ROOT)

    assert loaded.eligibility.eligible_for_chunking is False
    assert expected_code in loaded.eligibility.rejection_codes
    assert loaded.document.source_id == restricted.source_id


def test_permission_required_license_is_not_eligible_for_chunking() -> None:
    entry = next(
        source
        for source in _formal_manifest().sources
        if source.source_id == "mean_median_core"
    )
    restricted = entry.model_copy(update={"license": "permission-required"})

    decision = assess_chunking_eligibility(restricted)

    assert decision.eligible_for_chunking is False
    assert EligibilityRejectionCode.LICENSE_PERMISSION_REQUIRED in decision.rejection_codes


def test_course_sources_do_not_copy_internal_fields_questions_or_eval_case_ids() -> None:
    question_prompts = {
        _normalized(question.prompt)
        for question in load_default_question_bank().questions
    }
    eval_case_ids: set[str] = set()
    for directory in EVAL_DIRECTORIES:
        for path in directory.glob("*.jsonl"):
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    case = json.loads(line)
                    eval_case_ids.add(str(case["id"]))

    for source_path in SOURCE_DIRECTORY.glob("*.yaml"):
        raw_document = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        keys = set(_walk_keys(raw_document))
        texts = list(_walk_text(raw_document))
        normalized_texts = {_normalized(text) for text in texts}
        combined_text = "\n".join(texts)

        assert not (keys & FORBIDDEN_SOURCE_KEYS)
        assert not any(key.startswith("expected_") for key in keys)
        assert not (normalized_texts & question_prompts)
        assert not any(case_id in combined_text for case_id in eval_case_ids)
