"""Bounded reliability controls for the optional online explanation layer."""

import asyncio
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeVar

ResultT = TypeVar("ResultT")
Clock = Callable[[], float]
Sleeper = Callable[[float], Awaitable[object]]


class CircuitState(StrEnum):
    """Observable state of the process-local optional-model circuit."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class ModelReliabilityError(RuntimeError):
    """Base error for a bounded optional-model call."""


class ModelCallTimeoutError(ModelReliabilityError):
    """Every permitted attempt reached the configured wall-clock timeout."""


class ModelCallFailedError(ModelReliabilityError):
    """Every permitted attempt failed before producing a valid result."""


class ModelCircuitOpenError(ModelReliabilityError):
    """The circuit rejected a call while the optional model was degraded."""


@dataclass(frozen=True)
class CircuitSnapshot:
    """Non-sensitive circuit status used by health checks and tests."""

    state: CircuitState
    consecutive_failures: int


class ModelReliabilityController:
    """Apply a hard timeout, finite retries, and a process-local circuit breaker."""

    def __init__(
        self,
        *,
        timeout_seconds: float,
        max_attempts: int,
        retry_base_delay_seconds: float,
        failure_threshold: int,
        open_seconds: float,
        clock: Clock = time.monotonic,
        sleeper: Sleeper = asyncio.sleep,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("模型超时必须大于 0 秒")
        if max_attempts not in range(1, 4):
            raise ValueError("模型调用尝试次数必须是 1、2 或 3")
        if retry_base_delay_seconds < 0:
            raise ValueError("模型重试等待不能为负数")
        if failure_threshold <= 0:
            raise ValueError("熔断失败阈值必须大于 0")
        if open_seconds <= 0:
            raise ValueError("熔断开启时间必须大于 0 秒")

        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.retry_base_delay_seconds = retry_base_delay_seconds
        self.failure_threshold = failure_threshold
        self.open_seconds = open_seconds
        self._clock = clock
        self._sleeper = sleeper
        self._lock = threading.Lock()
        self._consecutive_failures = 0
        self._opened_at: float | None = None
        self._half_open_probe_in_flight = False

    @property
    def snapshot(self) -> CircuitSnapshot:
        """Return state without exposing provider, prompt, learner, or exception data."""

        with self._lock:
            if self._opened_at is None:
                state = CircuitState.CLOSED
            elif self._clock() - self._opened_at < self.open_seconds:
                state = CircuitState.OPEN
            else:
                state = CircuitState.HALF_OPEN
            return CircuitSnapshot(
                state=state,
                consecutive_failures=self._consecutive_failures,
            )

    async def run(self, operation: Callable[[], Awaitable[ResultT]]) -> ResultT:
        """Run one logical model call; retries never include state persistence."""

        is_half_open_probe = self._acquire_permission()
        last_error: Exception | None = None
        all_attempts_timed_out = True
        try:
            for attempt_index in range(self.max_attempts):
                try:
                    result = await asyncio.wait_for(
                        operation(),
                        timeout=self.timeout_seconds,
                    )
                except TimeoutError as error:
                    last_error = error
                except Exception as error:
                    last_error = error
                    all_attempts_timed_out = False
                else:
                    self._record_success()
                    return result

                if attempt_index + 1 < self.max_attempts:
                    delay = self.retry_base_delay_seconds * (attempt_index + 1)
                    await self._sleeper(delay)
        except asyncio.CancelledError:
            self._release_cancelled_probe(is_half_open_probe)
            raise

        self._record_failure()
        if all_attempts_timed_out:
            raise ModelCallTimeoutError(
                f"可选模型在 {self.max_attempts} 次尝试中均超时"
            ) from last_error
        raise ModelCallFailedError(
            f"可选模型在 {self.max_attempts} 次尝试后仍不可用"
        ) from last_error

    def _acquire_permission(self) -> bool:
        with self._lock:
            if self._opened_at is None:
                return False
            if self._clock() - self._opened_at < self.open_seconds:
                raise ModelCircuitOpenError("可选模型熔断器仍处于开启状态")
            if self._half_open_probe_in_flight:
                raise ModelCircuitOpenError("可选模型正在执行半开探测")
            self._half_open_probe_in_flight = True
            return True

    def _record_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0
            self._opened_at = None
            self._half_open_probe_in_flight = False

    def _record_failure(self) -> None:
        with self._lock:
            self._consecutive_failures += 1
            if (
                self._half_open_probe_in_flight
                or self._consecutive_failures >= self.failure_threshold
            ):
                self._opened_at = self._clock()
            self._half_open_probe_in_flight = False

    def _release_cancelled_probe(self, is_half_open_probe: bool) -> None:
        if not is_half_open_probe:
            return
        with self._lock:
            self._opened_at = self._clock()
            self._half_open_probe_in_flight = False
