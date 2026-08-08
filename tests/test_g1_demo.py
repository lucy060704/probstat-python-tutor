"""Tests for the reproducible five-run G1 offline demonstration."""

import asyncio
from pathlib import Path

import pytest

from probstat_tutor.demo import run_g1_offline_demo


def test_five_offline_journeys_pass_with_isolated_learners(tmp_path: Path) -> None:
    summary = asyncio.run(run_g1_offline_demo(tmp_path))

    assert summary.offline_mode is True
    assert summary.requested_runs == 5
    assert summary.passed_runs == 5
    assert summary.all_passed is True
    assert len({run.learner_id for run in summary.runs}) == 5
    assert all(run.history_count == 2 for run in summary.runs)
    assert all(run.passed for run in summary.runs)
    assert not (tmp_path / "sessions.sqlite3").exists()


def test_demo_rejects_nonpositive_repetition_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="至少为 1"):
        asyncio.run(run_g1_offline_demo(tmp_path, repetitions=0))
