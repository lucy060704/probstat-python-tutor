"""Deterministic tests for optional-model timeout, retry, and circuit behavior."""

import asyncio

import pytest

from probstat_tutor.reliability import (
    CircuitState,
    ModelCallFailedError,
    ModelCallTimeoutError,
    ModelCircuitOpenError,
    ModelReliabilityController,
)


def test_transient_failure_retries_once_then_closes_cleanly() -> None:
    calls = 0
    delays: list[float] = []

    async def operation() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary-provider-detail")
        return "safe-result"

    async def record_delay(delay: float) -> None:
        delays.append(delay)

    controller = ModelReliabilityController(
        timeout_seconds=1.0,
        max_attempts=2,
        retry_base_delay_seconds=0.25,
        failure_threshold=2,
        open_seconds=10.0,
        sleeper=record_delay,
    )

    result = asyncio.run(controller.run(operation))

    assert result == "safe-result"
    assert calls == 2
    assert delays == [0.25]
    assert controller.snapshot.state == CircuitState.CLOSED
    assert controller.snapshot.consecutive_failures == 0


def test_timeout_is_hard_and_attempt_count_is_bounded() -> None:
    calls = 0

    async def too_slow() -> None:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.05)

    controller = ModelReliabilityController(
        timeout_seconds=0.001,
        max_attempts=2,
        retry_base_delay_seconds=0.0,
        failure_threshold=3,
        open_seconds=10.0,
    )

    with pytest.raises(ModelCallTimeoutError, match="2 次尝试"):
        asyncio.run(controller.run(too_slow))

    assert calls == 2
    assert controller.snapshot.state == CircuitState.CLOSED
    assert controller.snapshot.consecutive_failures == 1


def test_circuit_opens_rejects_calls_and_half_open_success_resets() -> None:
    now = [0.0]
    calls = 0

    async def fail() -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("provider-unavailable")

    async def recover() -> str:
        nonlocal calls
        calls += 1
        return "recovered"

    controller = ModelReliabilityController(
        timeout_seconds=1.0,
        max_attempts=1,
        retry_base_delay_seconds=0.0,
        failure_threshold=2,
        open_seconds=10.0,
        clock=lambda: now[0],
    )

    with pytest.raises(ModelCallFailedError):
        asyncio.run(controller.run(fail))
    with pytest.raises(ModelCallFailedError):
        asyncio.run(controller.run(fail))
    assert controller.snapshot.state == CircuitState.OPEN

    with pytest.raises(ModelCircuitOpenError):
        asyncio.run(controller.run(recover))
    assert calls == 2

    now[0] = 10.0
    assert controller.snapshot.state == CircuitState.HALF_OPEN
    assert asyncio.run(controller.run(recover)) == "recovered"
    assert calls == 3
    assert controller.snapshot.state == CircuitState.CLOSED
    assert controller.snapshot.consecutive_failures == 0


def test_half_open_allows_exactly_one_in_flight_probe() -> None:
    now = [0.0]
    controller = ModelReliabilityController(
        timeout_seconds=1.0,
        max_attempts=1,
        retry_base_delay_seconds=0.0,
        failure_threshold=1,
        open_seconds=10.0,
        clock=lambda: now[0],
    )

    async def exercise() -> None:
        async def fail() -> None:
            raise RuntimeError("provider-unavailable")

        with pytest.raises(ModelCallFailedError):
            await controller.run(fail)
        now[0] = 10.0
        probe_started = asyncio.Event()
        release_probe = asyncio.Event()

        async def probe() -> str:
            probe_started.set()
            await release_probe.wait()
            return "recovered"

        first_probe = asyncio.create_task(controller.run(probe))
        await probe_started.wait()
        with pytest.raises(ModelCircuitOpenError):
            await controller.run(probe)
        release_probe.set()
        assert await first_probe == "recovered"

    asyncio.run(exercise())

    assert controller.snapshot.state == CircuitState.CLOSED


def test_cancelling_half_open_probe_reopens_without_counting_new_failure() -> None:
    now = [0.0]
    controller = ModelReliabilityController(
        timeout_seconds=1.0,
        max_attempts=1,
        retry_base_delay_seconds=0.0,
        failure_threshold=1,
        open_seconds=10.0,
        clock=lambda: now[0],
    )

    async def exercise() -> None:
        async def fail() -> None:
            raise RuntimeError("provider-unavailable")

        with pytest.raises(ModelCallFailedError):
            await controller.run(fail)
        now[0] = 10.0
        probe_started = asyncio.Event()

        async def probe() -> None:
            probe_started.set()
            await asyncio.Event().wait()

        task = asyncio.create_task(controller.run(probe))
        await probe_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())

    assert controller.snapshot.state == CircuitState.OPEN
    assert controller.snapshot.consecutive_failures == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timeout_seconds": 0.0},
        {"max_attempts": 4},
        {"retry_base_delay_seconds": -0.1},
        {"failure_threshold": 0},
        {"open_seconds": 0.0},
    ],
)
def test_invalid_reliability_limits_are_rejected(kwargs: dict[str, float | int]) -> None:
    defaults: dict[str, float | int] = {
        "timeout_seconds": 1.0,
        "max_attempts": 2,
        "retry_base_delay_seconds": 0.0,
        "failure_threshold": 2,
        "open_seconds": 10.0,
    }
    defaults.update(kwargs)

    with pytest.raises(ValueError):
        ModelReliabilityController(**defaults)  # type: ignore[arg-type]
