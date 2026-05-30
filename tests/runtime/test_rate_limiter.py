import time

import pytest
from castellum.runtime.rate_limiter import RateLimiter


@pytest.mark.asyncio
async def test_rate_limiter_allows_within_budget():
    limiter = RateLimiter(tokens_per_minute=6000)
    start = time.monotonic()
    for _ in range(5):
        await limiter.acquire(1)
    elapsed = time.monotonic() - start
    assert elapsed < 0.5


@pytest.mark.asyncio
async def test_rate_limiter_throttles_excess():
    limiter = RateLimiter(tokens_per_minute=60)
    await limiter.acquire(60)
    start = time.monotonic()
    await limiter.acquire(1)
    elapsed = time.monotonic() - start
    assert elapsed >= 0.9
