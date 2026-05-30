from __future__ import annotations

import asyncio
import time


class RateLimiter:
    def __init__(self, *, tokens_per_minute: int) -> None:
        self._tpm = tokens_per_minute
        self._rate = tokens_per_minute / 60.0
        self._capacity = float(tokens_per_minute)
        self._tokens = self._capacity
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
            self._last_refill = now

            if self._tokens >= tokens:
                self._tokens -= tokens
                return

            deficit = tokens - self._tokens
            wait = deficit / self._rate
            self._tokens = 0

        await asyncio.sleep(wait)
        async with self._lock:
            self._tokens = max(0.0, self._tokens - tokens)


class ModelRateLimiter:
    def __init__(
        self,
        *,
        requests_per_minute: int | None = None,
        tokens_per_minute: int | None = None,
    ) -> None:
        self._rpm_limiter = RateLimiter(tokens_per_minute=requests_per_minute) if requests_per_minute else None
        self._tpm_limiter = RateLimiter(tokens_per_minute=tokens_per_minute) if tokens_per_minute else None

    async def acquire(self, tokens: int = 1) -> None:
        if self._rpm_limiter:
            await self._rpm_limiter.acquire(1)
        if self._tpm_limiter:
            await self._tpm_limiter.acquire(tokens)
