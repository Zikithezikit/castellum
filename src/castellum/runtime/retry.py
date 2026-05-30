from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import Any


class RetryPolicy:
    def __init__(
        self,
        *,
        max_retries: int = 3,
        base_delay: float = 0.5,
        max_delay: float = 30.0,
        jitter: bool = True,
        retryable_exceptions: tuple[type[Exception], ...] | None = None,
    ) -> None:
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions or (
            OSError,
            TimeoutError,
            asyncio.TimeoutError,
            ConnectionError,
        )

    async def execute(self, fn: Callable[[], Awaitable[Any]]) -> Any:
        attempt = 0
        last_exc: Exception | None = None

        while attempt <= self.max_retries:
            try:
                return await fn()
            except self.retryable_exceptions as exc:
                last_exc = exc
                if attempt == self.max_retries:
                    break
                delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                if self.jitter:
                    delay += random.uniform(0, self.base_delay)
                await asyncio.sleep(delay)
                attempt += 1

        raise last_exc  # type: ignore[misc]
