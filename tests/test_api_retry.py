"""Tests for the transport-independent bounded API retry policy."""

import pytest

from probstat_tutor.api.retry import BoundedApiRetryPolicy


def test_retryable_post_uses_same_key_with_bounded_exponential_delays() -> None:
    policy = BoundedApiRetryPolicy(
        max_attempts=3,
        base_delay_seconds=0.25,
        max_delay_seconds=0.4,
    )

    first = policy.decide(
        method="POST",
        completed_attempts=1,
        status_code=503,
        retryable=True,
        original_idempotency_key="idem_original_001",
        current_idempotency_key="idem_original_001",
    )
    second = policy.decide(
        method="POST",
        completed_attempts=2,
        status_code=500,
        retryable=True,
        original_idempotency_key="idem_original_001",
        current_idempotency_key="idem_original_001",
    )
    final = policy.decide(
        method="POST",
        completed_attempts=3,
        status_code=503,
        retryable=True,
        original_idempotency_key="idem_original_001",
        current_idempotency_key="idem_original_001",
    )

    assert first.should_retry is True
    assert first.delay_seconds == 0.25
    assert second.should_retry is True
    assert second.delay_seconds == 0.4
    assert final.should_retry is False


@pytest.mark.parametrize(
    ("status_code", "retryable", "original_key", "current_key"),
    [
        (409, False, "idem_original_001", "idem_original_001"),
        (422, False, "idem_original_001", "idem_original_001"),
        (503, False, "idem_original_001", "idem_original_001"),
        (503, True, None, None),
        (503, True, "idem_original_001", "idem_changed_002"),
    ],
)
def test_post_does_not_retry_unsafe_or_non_retryable_response(
    status_code: int,
    retryable: bool,
    original_key: str | None,
    current_key: str | None,
) -> None:
    decision = BoundedApiRetryPolicy().decide(
        method="POST",
        completed_attempts=1,
        status_code=status_code,
        retryable=retryable,
        original_idempotency_key=original_key,
        current_idempotency_key=current_key,
    )

    assert decision.should_retry is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_attempts": 4},
        {"base_delay_seconds": -0.1},
        {"base_delay_seconds": 1.0, "max_delay_seconds": 0.5},
    ],
)
def test_invalid_retry_policy_is_rejected(kwargs: dict[str, float | int]) -> None:
    with pytest.raises(ValueError):
        BoundedApiRetryPolicy(**kwargs)  # type: ignore[arg-type]
