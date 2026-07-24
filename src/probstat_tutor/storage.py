"""Small SQLite repository for deterministic learner state."""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from probstat_tutor.schemas import DiagnosticReport, LearningState


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

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT report_json FROM submission_receipts
                WHERE learner_id = ? AND submission_key = ?
                """,
                (learner_id, submission_key),
            ).fetchone()
        if row is None:
            return None
        return DiagnosticReport.model_validate_json(row[0])

    def save_submission_report(
        self,
        learner_id: str,
        submission_key: str,
        report: DiagnosticReport,
    ) -> None:
        """Cache a report so repeated UI clicks do not update mastery twice."""

        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO submission_receipts
                    (learner_id, submission_key, report_json)
                VALUES (?, ?, ?)
                """,
                (learner_id, submission_key, report.model_dump_json()),
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
                    report_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (learner_id, submission_key)
                )
                """
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path)
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
