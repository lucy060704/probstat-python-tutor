"""Atomic SQLite tests for learner state and idempotency receipts."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from probstat_tutor.schemas import DiagnosticReport, LearningState, MasteryScores
from probstat_tutor.storage import CommitSubmissionStatus, LearningStateStore


def _state_with_marker(marker: str) -> LearningState:
    return LearningState(completed_question_ids=frozenset({marker}))


def _report(marker: str) -> DiagnosticReport:
    return DiagnosticReport(
        question_id="mean_median_python_01",
        overall_correctness=1.0,
        dimension_scores=MasteryScores(),
        evidence=[marker],
        misconception_tags=[],
        feedback="确定性报告",
        hint_level=0,
        recommended_action="继续学习",
        next_question_id=None,
        uncertainty="无",
    )


def test_atomic_commit_saves_state_and_receipt_together(tmp_path: Path) -> None:
    store = LearningStateStore(tmp_path / "learning.sqlite3")
    base_state = LearningState()
    updated_state = _state_with_marker("created")
    report = _report("created")

    result = store.commit_submission(
        learner_id="learner",
        submission_key="key-created",
        request_fingerprint="fingerprint-created",
        expected_state=base_state,
        updated_state=updated_state,
        report=report,
    )

    assert result.status == CommitSubmissionStatus.CREATED
    assert result.report == report
    assert store.load("learner") == updated_state
    assert store.load_submission_report("learner", "key-created") == report


def test_duplicate_commit_returns_cached_winner_without_rewriting_state(
    tmp_path: Path,
) -> None:
    store = LearningStateStore(tmp_path / "learning.sqlite3")
    base_state = LearningState()
    winning_state = _state_with_marker("winner")
    winning_report = _report("winner")
    store.commit_submission(
        learner_id="learner",
        submission_key="same-key",
        request_fingerprint="fingerprint-same",
        expected_state=base_state,
        updated_state=winning_state,
        report=winning_report,
    )

    result = store.commit_submission(
        learner_id="learner",
        submission_key="same-key",
        request_fingerprint="fingerprint-same",
        expected_state=base_state,
        updated_state=_state_with_marker("loser"),
        report=_report("loser"),
    )

    assert result.status == CommitSubmissionStatus.CACHED
    assert result.report == winning_report
    assert store.load("learner") == winning_state


def test_stale_snapshot_conflicts_without_overwriting_current_state(tmp_path: Path) -> None:
    store = LearningStateStore(tmp_path / "learning.sqlite3")
    base_state = LearningState()
    current_state = _state_with_marker("first")
    store.commit_submission(
        learner_id="learner",
        submission_key="first-key",
        request_fingerprint="fingerprint-first",
        expected_state=base_state,
        updated_state=current_state,
        report=_report("first"),
    )

    result = store.commit_submission(
        learner_id="learner",
        submission_key="second-key",
        request_fingerprint="fingerprint-second",
        expected_state=base_state,
        updated_state=_state_with_marker("stale-second"),
        report=_report("stale-second"),
    )

    assert result.status == CommitSubmissionStatus.CONFLICT
    assert result.report is None
    assert store.load("learner") == current_state
    assert store.load_submission_report("learner", "second-key") is None


def test_threaded_identical_commits_create_once_and_return_cached_winner(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "learning.sqlite3"
    first_store = LearningStateStore(db_path)
    second_store = LearningStateStore(db_path)
    barrier = Barrier(2)

    def commit(store: LearningStateStore, marker: str) -> CommitSubmissionStatus:
        barrier.wait()
        return store.commit_submission(
            learner_id="learner",
            submission_key="same-key",
            request_fingerprint="fingerprint-same",
            expected_state=LearningState(),
            updated_state=_state_with_marker(marker),
            report=_report(marker),
        ).status

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(
            executor.map(
                commit,
                (first_store, second_store),
                ("first", "second"),
            )
        )

    assert results.count(CommitSubmissionStatus.CREATED) == 1
    assert results.count(CommitSubmissionStatus.CACHED) == 1
    receipt = first_store.load_submission_report("learner", "same-key")
    assert receipt is not None
    assert first_store.load("learner").completed_question_ids == frozenset(
        {receipt.evidence[0]}
    )


def test_threaded_distinct_commits_reject_one_stale_snapshot(tmp_path: Path) -> None:
    db_path = tmp_path / "learning.sqlite3"
    first_store = LearningStateStore(db_path)
    second_store = LearningStateStore(db_path)
    barrier = Barrier(2)

    def commit(store: LearningStateStore, marker: str) -> CommitSubmissionStatus:
        barrier.wait()
        return store.commit_submission(
            learner_id="learner",
            submission_key=f"key-{marker}",
            request_fingerprint=f"fingerprint-{marker}",
            expected_state=LearningState(),
            updated_state=_state_with_marker(marker),
            report=_report(marker),
        ).status

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(
            executor.map(
                commit,
                (first_store, second_store),
                ("first", "second"),
            )
        )

    assert results.count(CommitSubmissionStatus.CREATED) == 1
    assert results.count(CommitSubmissionStatus.CONFLICT) == 1
    final_state = first_store.load("learner")
    assert len(final_state.completed_question_ids) == 1
    winning_marker = next(iter(final_state.completed_question_ids))
    assert first_store.load_submission_report(
        "learner", f"key-{winning_marker}"
    ) is not None


def test_threaded_same_key_different_fingerprints_create_once_and_conflict_once(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "learning.sqlite3"
    first_store = LearningStateStore(db_path)
    second_store = LearningStateStore(db_path)
    barrier = Barrier(2)

    def commit(store: LearningStateStore, marker: str) -> CommitSubmissionStatus:
        barrier.wait()
        return store.commit_submission(
            learner_id="learner",
            submission_key="shared-external-key",
            request_fingerprint=f"fingerprint-{marker}",
            expected_state=LearningState(),
            updated_state=_state_with_marker(marker),
            report=_report(marker),
        ).status

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(
            executor.map(
                commit,
                (first_store, second_store),
                ("first", "second"),
            )
        )

    assert results.count(CommitSubmissionStatus.CREATED) == 1
    assert results.count(CommitSubmissionStatus.IDEMPOTENCY_CONFLICT) == 1
    receipt = first_store.load_submission_receipt("learner", "shared-external-key")
    assert receipt is not None
    assert receipt.request_fingerprint == f"fingerprint-{receipt.report.evidence[0]}"
    assert first_store.load("learner").completed_question_ids == frozenset(
        {receipt.report.evidence[0]}
    )


def test_receipt_insert_failure_rolls_back_state_write(tmp_path: Path) -> None:
    store = LearningStateStore(tmp_path / "learning.sqlite3")
    with sqlite3.connect(store.db_path) as connection:
        connection.execute(
            """
            CREATE TRIGGER fail_receipt_insert
            BEFORE INSERT ON submission_receipts
            BEGIN
                SELECT RAISE(ABORT, 'injected receipt failure');
            END
            """
        )

    with pytest.raises(sqlite3.IntegrityError, match="injected receipt failure"):
        store.commit_submission(
            learner_id="learner",
            submission_key="rollback-key",
            request_fingerprint="fingerprint-rollback",
            expected_state=LearningState(),
            updated_state=_state_with_marker("must-rollback"),
            report=_report("must-rollback"),
        )

    assert store.load("learner") == LearningState()
    assert store.load_submission_report("learner", "rollback-key") is None


def test_same_key_with_different_fingerprint_conflicts_without_state_write(
    tmp_path: Path,
) -> None:
    store = LearningStateStore(tmp_path / "learning.sqlite3")
    winning_state = _state_with_marker("winner")
    winning_report = _report("winner")
    store.commit_submission(
        learner_id="learner",
        submission_key="same-key",
        request_fingerprint="fingerprint-first",
        expected_state=LearningState(),
        updated_state=winning_state,
        report=winning_report,
    )

    result = store.commit_submission(
        learner_id="learner",
        submission_key="same-key",
        request_fingerprint="fingerprint-changed",
        expected_state=winning_state,
        updated_state=_state_with_marker("must-not-write"),
        report=_report("must-not-write"),
    )

    assert result.status == CommitSubmissionStatus.IDEMPOTENCY_CONFLICT
    assert result.report is None
    assert store.load("learner") == winning_state
    assert store.load_submission_report("learner", "same-key") == winning_report


def test_pre_g35_receipt_schema_is_migrated_without_guessing_external_requests(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "legacy.sqlite3"
    legacy_report = _report("legacy")
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE submission_receipts (
                learner_id TEXT NOT NULL,
                submission_key TEXT NOT NULL,
                report_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (learner_id, submission_key)
            )
            """
        )
        connection.execute(
            """
            INSERT INTO submission_receipts (learner_id, submission_key, report_json)
            VALUES (?, ?, ?)
            """,
            ("learner", "legacy-content-fingerprint", legacy_report.model_dump_json()),
        )

    store = LearningStateStore(db_path)
    receipt = store.load_submission_receipt("learner", "legacy-content-fingerprint")

    assert receipt is not None
    assert receipt.request_fingerprint is None
    assert receipt.matches(
        request_fingerprint="legacy-content-fingerprint",
        submission_key="legacy-content-fingerprint",
    )
    assert not receipt.matches(
        request_fingerprint="different-request-body",
        submission_key="legacy-content-fingerprint",
    )
