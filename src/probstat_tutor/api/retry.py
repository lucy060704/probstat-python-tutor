"""Transport-agnostic bounded retry policy for future API clients."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ApiRetryDecision:
    """One safe client action after an API response."""

    should_retry: bool
    delay_seconds: float = 0.0


class BoundedApiRetryPolicy:
    """Retry only explicitly retryable failures and never exceed three attempts."""

    def __init__(
        self,
        *,
        max_attempts: int = 3,
        base_delay_seconds: float = 0.25,
        max_delay_seconds: float = 1.0,
    ) -> None:
        if max_attempts not in range(1, 4):
            raise ValueError("API 最大尝试次数必须是 1、2 或 3")
        if base_delay_seconds < 0:
            raise ValueError("API 重试基础等待不能为负数")
        if max_delay_seconds < base_delay_seconds:
            raise ValueError("API 重试最大等待不能小于基础等待")
        self.max_attempts = max_attempts
        self.base_delay_seconds = base_delay_seconds
        self.max_delay_seconds = max_delay_seconds

    def decide(
        self,
        *,
        method: str,
        completed_attempts: int,
        status_code: int,
        retryable: bool,
        original_idempotency_key: str | None,
        current_idempotency_key: str | None,
    ) -> ApiRetryDecision:
        """Return a deterministic backoff decision without sending a request."""

        if completed_attempts <= 0:
            raise ValueError("已完成尝试次数必须大于 0")
        if completed_attempts >= self.max_attempts:
            return ApiRetryDecision(should_retry=False)
        if not retryable or status_code not in {500, 503}:
            return ApiRetryDecision(should_retry=False)
        if method.upper() == "POST" and (
            original_idempotency_key is None
            or current_idempotency_key != original_idempotency_key
        ):
            return ApiRetryDecision(should_retry=False)
        delay = min(
            self.base_delay_seconds * (2 ** (completed_attempts - 1)),
            self.max_delay_seconds,
        )
        return ApiRetryDecision(should_retry=True, delay_seconds=delay)
