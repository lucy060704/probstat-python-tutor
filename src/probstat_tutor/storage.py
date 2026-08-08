"""Small SQLite repository for deterministic learner state."""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from probstat_tutor.schemas import DiagnosticReport, LearningState


class CommitSubmissionStatus(StrEnum):
    """Possible outcomes of an atomic state-and-receipt commit."""

    CREATED = "created"
    CACHED = "cached"
    CONFLICT = "conflict"
    IDEMPOTENCY_CONFLICT = "idempotency_conflict"


@dataclass(frozen=True)
class CommitSubmissionResult:
    """Result returned after checking idempotency and the expected state snapshot."""

    status: CommitSubmissionStatus
    report: DiagnosticReport | None = None


@dataclass(frozen=True)
class SubmissionReceipt:
    """A persisted report plus the logical request fingerprint that created it."""

    request_fingerprint: str | None
    report: DiagnosticReport

    def matches(self, *, request_fingerprint: str, submission_key: str) -> bool:
        """Return whether this receipt represents the same logical request."""

        return _fingerprints_match(
            stored_fingerprint=self.request_fingerprint,
            request_fingerprint=request_fingerprint,
            submission_key=submission_key,
        )


class LearningStateStore:
    """Persist anonymized learning state in an independent SQLite table."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def load(self, learner_id: str) -> LearningState:
        """Load one learner or return a neutral state when no row exists."""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT state_json FROM learner_states WHERE learner_id = ?",
                (learner_id,),
            ).fetchone()
        if row is None:
            return LearningState()
        return LearningState.model_validate_json(row[0])

    def load_anonymized_states(self) -> tuple[LearningState, ...]:
        """Load state payloads only; learner IDs and raw receipts never leave storage."""

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT state_json FROM learner_states ORDER BY updated_at, rowid"
            ).fetchall()
        return tuple(LearningState.model_validate_json(row[0]) for row in rows)

    def save(self, learner_id: str, state: LearningState) -> None:
        """Atomically insert or replace a validated learner state."""

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO learner_states (learner_id, state_json, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(learner_id) DO UPDATE SET
                    state_json = excluded.state_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (learner_id, state.model_dump_json()),
            )

    def load_submission_report(
        self, learner_id: str, submission_key: str
    ) -> DiagnosticReport | None:
        """Return a cached report for an idempotent submission, when present."""

        receipt = self.load_submission_receipt(learner_id, submission_key)
        return None if receipt is None else receipt.report

    def load_submission_receipt(
        self, learner_id: str, submission_key: str
    ) -> SubmissionReceipt | None:
        """Return a receipt and its request fingerprint, when present."""

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT request_fingerprint, report_json FROM submission_receipts
                WHERE learner_id = ? AND submission_key = ?
                """,
                (learner_id, submission_key),
            ).fetchone()
        if row is None:
            return None
        return SubmissionReceipt(
            request_fingerprint=row[0],
            report=DiagnosticReport.model_validate_json(row[1]),
        )

    def commit_submission(
        self,
        *,
        learner_id: str,
        submission_key: str,
        request_fingerprint: str,
        expected_state: LearningState,
        updated_state: LearningState,
        report: DiagnosticReport,
    ) -> CommitSubmissionResult:
        """Atomically save mastery and its receipt, or report a snapshot conflict."""

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            receipt_row = connection.execute(
                """
                SELECT request_fingerprint, report_json FROM submission_receipts
                WHERE learner_id = ? AND submission_key = ?
                """,
                (learner_id, submission_key),
            ).fetchone()
            if receipt_row is not None:
                stored_fingerprint = receipt_row[0]
                if not _fingerprints_match(
                    stored_fingerprint=stored_fingerprint,
                    request_fingerprint=request_fingerprint,
                    submission_key=submission_key,
                ):
                    return CommitSubmissionResult(
                        status=CommitSubmissionStatus.IDEMPOTENCY_CONFLICT
                    )
                return CommitSubmissionResult(
                    status=CommitSubmissionStatus.CACHED,
                    report=DiagnosticReport.model_validate_json(receipt_row[1]),
                )

            state_row = connection.execute(
                "SELECT state_json FROM learner_states WHERE learner_id = ?",
                (learner_id,),
            ).fetchone()
            current_state = (
                LearningState()
                if state_row is None
                else LearningState.model_validate_json(state_row[0])
            )
            if current_state != expected_state:
                return CommitSubmissionResult(status=CommitSubmissionStatus.CONFLICT)

            connection.execute(
                """
                INSERT INTO learner_states (learner_id, state_json, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(learner_id) DO UPDATE SET
                    state_json = excluded.state_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (learner_id, updated_state.model_dump_json()),
            )
            connection.execute(
                """
                INSERT INTO submission_receipts
                    (learner_id, submission_key, request_fingerprint, report_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    learner_id,
                    submission_key,
                    request_fingerprint,
                    report.model_dump_json(),
                ),
            )
            return CommitSubmissionResult(
                status=CommitSubmissionStatus.CREATED,
                report=report,
            )

    def reset(self, learner_id: str) -> None:
        """Delete one demo learner's state and idempotency receipts."""

        with self._connect() as connection:
            connection.execute(
                "DELETE FROM learner_states WHERE learner_id = ?", (learner_id,)
            )
            connection.execute(
                "DELETE FROM submission_receipts WHERE learner_id = ?", (learner_id,)
            )

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS learner_states (
                    learner_id TEXT PRIMARY KEY,
                    state_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS submission_receipts (
                    learner_id TEXT NOT NULL,
                    submission_key TEXT NOT NULL,
                    request_fingerprint TEXT,
                    report_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (learner_id, submission_key)
                )
                """
            )
            columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(submission_receipts)"
                ).fetchall()
            }
            if "request_fingerprint" not in columns:
                connection.execute(
                    "ALTER TABLE submission_receipts ADD COLUMN request_fingerprint TEXT"
                )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path, timeout=5.0)
        connection.execute("PRAGMA busy_timeout = 5000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()


def _fingerprints_match(
    *,
    stored_fingerprint: str | None,
    request_fingerprint: str,
    submission_key: str,
) -> bool:
    """Safely handle new receipts and pre-G3.5 content-key receipts."""

    if stored_fingerprint is not None:
        return stored_fingerprint == request_fingerprint
    return submission_key == request_fingerprint
