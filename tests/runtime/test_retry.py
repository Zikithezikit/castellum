import pytest
from castellum.runtime.retry import RetryPolicy


@pytest.mark.asyncio
async def test_retry_succeeds_eventually():
    calls = []

    async def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise OSError("transient")
        return "ok"

    policy = RetryPolicy(max_retries=3, base_delay=0.01, jitter=False)
    result = await policy.execute(flaky)
    assert result == "ok"
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_retry_exhausts_and_raises():
    async def always_fail():
        raise OSError("always")

    policy = RetryPolicy(max_retries=2, base_delay=0.01, jitter=False)
    with pytest.raises(OSError, match="always"):
        await policy.execute(always_fail)
